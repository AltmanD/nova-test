"""
性能监控模块 - 用于监控训练过程中的性能指标

1. 训练侧吞吐监控（rolling 统计）
2. CUDA allocator 监控
3. validation/checkpoint 事件标记
4. 通过开关控制是否开启
"""

import time
from collections import deque
import torch
import torch.distributed as dist
import sys


class PerfMonitor:
    """
    性能监控器 - 跟踪训练过程中的关键性能指标

    监控指标：
    - optimizer-step wall time rolling p50/p90/mean
    - rolling tokens/sec, EMA tokens/sec
    - rank step time min/mean/max, max/min ratio
    - CUDA allocator 统计
    - validation/checkpoint 事件标记
    """

    def __init__(self, enabled=True, window_size=50, ema_alpha=0.05,
                 log_interval=10, cuda_mem_interval=50):
        """
        初始化性能监控器

        Args:
            enabled: 是否启用监控
            window_size: rolling window 大小（optimizer steps）
            ema_alpha: EMA 平滑系数
            log_interval: 日志打印间隔（optimizer steps）
            cuda_mem_interval: CUDA 内存统计间隔（optimizer steps）
        """
        self.enabled = enabled

        if not self.enabled:
            return

        # Rolling window 配置
        self.window_size = window_size
        self.ema_alpha = ema_alpha
        self.log_interval = log_interval
        self.cuda_mem_interval = cuda_mem_interval

        # 时间跟踪
        self._perf_last_global_step = None
        self._perf_last_time = None
        self._step_times = deque(maxlen=window_size)
        self._tok_sec_ema = None

        # 事件跟踪
        self._event_stack = []  # 用于嵌套事件（虽然通常不会嵌套）

    def on_optimizer_step(self, global_step, num_gpus, batch_size_per_device,
                          context_length, accumulate_grad_batches, device=None):
        """
        在 optimizer step 完成后调用，记录性能指标

        Args:
            global_step: 当前 global_step
            num_gpus: GPU 数量
            batch_size_per_device: 每设备 batch size
            context_length: 上下文长度
            accumulate_grad_batches: 梯度累积步数
            device: 当前设备
        """
        if not self.enabled:
            return

        now = time.perf_counter()
        gs = int(global_step)

        # 初始化
        if self._perf_last_global_step is None:
            self._perf_last_global_step = gs
            self._perf_last_time = now
            return

        # 只在 global_step 增加时记录（optimizer step 完成）
        if gs > self._perf_last_global_step:
            step_dt = now - self._perf_last_time
            self._perf_last_time = now
            self._perf_last_global_step = gs
            self._step_times.append(step_dt)

            # 计算 tokens per optimizer step
            tokens_per_opt_step = (
                num_gpus
                * batch_size_per_device
                * context_length
                * accumulate_grad_batches
            )

            # Rolling window 统计
            window_time = sum(self._step_times)
            tok_sec_w = tokens_per_opt_step * len(self._step_times) / window_time

            # EMA 更新
            if self._tok_sec_ema is None:
                self._tok_sec_ema = tok_sec_w
            else:
                self._tok_sec_ema = self.ema_alpha * tok_sec_w + (1 - self.ema_alpha) * self._tok_sec_ema

            # Rank 间 step time 聚合
            local_t = torch.tensor(step_dt, device=device)
            t_min = local_t.clone()
            t_max = local_t.clone()
            t_sum = local_t.clone()
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(t_min, op=dist.ReduceOp.MIN)
                dist.all_reduce(t_max, op=dist.ReduceOp.MAX)
                dist.all_reduce(t_sum, op=dist.ReduceOp.SUM)
                world = dist.get_world_size()
            else:
                world = 1

            t_mean = t_sum / world
            skew = t_max / torch.clamp(t_min, min=1e-6)

            # 计算 p50/p90
            step_times_list = list(self._step_times)
            step_times_list.sort()
            n = len(step_times_list)
            p50_idx = int(n * 0.5)
            p90_idx = int(n * 0.9)
            step_p50 = step_times_list[p50_idx] if p50_idx < n else step_times_list[-1]
            step_p90 = step_times_list[p90_idx] if p90_idx < n else step_times_list[-1]

            # 打印日志（只在 rank0 且满足间隔时）
            is_rank_zero = not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0
            if is_rank_zero and gs % self.log_interval == 0:
                print(
                    f"[perf] step={gs} "
                    f"tok_sec_w{self.window_size}={tok_sec_w:,.0f} "
                    f"tok_sec_ema={self._tok_sec_ema:,.0f} "
                    f"step_ms_last={step_dt*1000:.1f} "
                    f"step_ms_p50={step_p50*1000:.1f} "
                    f"step_ms_p90={step_p90*1000:.1f} "
                    f"rank_min_ms={t_min.item()*1000:.1f} "
                    f"rank_mean_ms={t_mean.item()*1000:.1f} "
                    f"rank_max_ms={t_max.item()*1000:.1f} "
                    f"rank_skew={skew.item():.2f}",
                    flush=True,
                )

            # CUDA 内存统计
            if gs % self.cuda_mem_interval == 0 and is_rank_zero:
                self._log_cuda_memory_stats(gs)

    def _log_cuda_memory_stats(self, global_step):
        """记录 CUDA 内存统计"""
        if not torch.cuda.is_available():
            return

        try:
            stats = torch.cuda.memory_stats()
            allocated_gb = torch.cuda.memory_allocated() / 1024**3
            reserved_gb = torch.cuda.memory_reserved() / 1024**3
            inactive_split_gb = stats.get("inactive_split_bytes.all.current", 0) / 1024**3
            alloc_retries = stats.get("num_alloc_retries", 0)
            oom_count = stats.get("num_ooms", 0)

            print(
                f"[cuda_mem] step={global_step} "
                f"alloc={allocated_gb:.2f}GB "
                f"reserved={reserved_gb:.2f}GB "
                f"inactive_split={inactive_split_gb:.2f}GB "
                f"alloc_retries={alloc_retries} "
                f"ooms={oom_count}",
                flush=True,
            )
        except Exception as e:
            print(f"[cuda_mem] Failed to get stats: {e}", flush=True)

    def log_event_start(self, event_name, global_step=None):
        """
        记录事件开始

        Args:
            event_name: 事件名称（如 'validation', 'checkpoint'）
            global_step: 当前 global_step
        """
        if not self.enabled:
            return

        now = time.time()
        event_info = {
            'name': event_name,
            'start': now,
            'step': global_step,
        }
        self._event_stack.append(event_info)

        is_rank_zero = not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0
        if is_rank_zero:
            step_str = f" step={global_step}" if global_step is not None else ""
            print(f"[event] {event_name}_start{step_str} time={now:.3f}", flush=True)

    def log_event_end(self, event_name=None, global_step=None):
        """
        记录事件结束

        Args:
            event_name: 事件名称（如果不提供，则使用最后一个开始的事件）
            global_step: 当前 global_step
        """
        if not self.enabled:
            return

        now = time.time()
        if not self._event_stack:
            print(f"[event] Warning: No event to end for '{event_name}'", flush=True)
            return

        # 如果没有提供 event_name，使用最后一个开始的事件
        if event_name is None:
            event_info = self._event_stack.pop()
            event_name = event_info['name']
        else:
            # 查找匹配的事件
            event_info = None
            for i in range(len(self._event_stack) - 1, -1, -1):
                if self._event_stack[i]['name'] == event_name:
                    event_info = self._event_stack.pop(i)
                    break
            if event_info is None:
                print(f"[event] Warning: No matching event '{event_name}' found", flush=True)
                return

        start_time = event_info['start']
        duration = now - start_time
        start_step = event_info['step']
        step_str = f" step={start_step}" if start_step is not None else ""

        is_rank_zero = not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0
        if is_rank_zero:
            print(
                f"[event] {event_name}_end{step_str} time={now:.3f} duration={duration:.3f}s",
                flush=True,
            )

    def get_summary(self):
        """
        获取监控摘要信息

        Returns:
            dict: 包含各项指标的摘要
        """
        if not self.enabled or len(self._step_times) == 0:
            return {}

        step_times_list = list(self._step_times)
        step_times_list.sort()
        n = len(step_times_list)

        return {
            'window_size': self.window_size,
            'num_samples': n,
            'step_time_mean': sum(step_times_list) / n,
            'step_time_p50': step_times_list[int(n * 0.5)],
            'step_time_p90': step_times_list[int(n * 0.9)],
            'tok_sec_ema': self._tok_sec_ema,
        }
