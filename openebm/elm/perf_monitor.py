"""
性能监控模块 - 用于定位 GPU 利用率随训练下降问题。

覆盖两份排查文档中的监控需求：
1. 训练侧 optimizer-step rolling 吞吐、rank straggler、data wait
2. CUDA allocator 统计
3. validation/checkpoint 事件标记
4. 预训练 nanochat 数据管线：parquet read、tokenizer encode、best-fit scan、GPU copy、buffer 状态
5. 原因判定提示：按监控量映射到可能原因，便于日志侧直接观察

所有监控均受 enabled 总开关控制；不开启时只保留轻量 no-op。
"""

import os
import time
from collections import defaultdict, deque

import torch
import torch.distributed as dist


class PerfMonitor:
    """
    性能监控器。

    主要用于判定 gpu_util_possible_causes_after_sft.md 中保留的原因：
    - 预训练 nanochat 数据管线慢性退化或 rank straggler
    - 跨节点 DDP/NCCL/straggler
    - allocator 配置/碎片化差异
    - validation/checkpoint 对共享存储、I/O queue、CUDA allocator 的阶跃扰动
    """

    def __init__(
        self,
        enabled=True,
        window_size=50,
        ema_alpha=0.05,
        log_interval=10,
        cuda_mem_interval=50,
        data_log_interval=50,
        cause_log_interval=200,
        monitor_data_pipeline=True,
        run_context=None,
    ):
        self.enabled = bool(enabled)

        # 即使 disabled，也保留属性，避免调用方需要判断存在性。
        self.window_size = int(window_size)
        self.ema_alpha = float(ema_alpha)
        self.log_interval = int(log_interval)
        self.cuda_mem_interval = int(cuda_mem_interval)
        self.data_log_interval = int(data_log_interval)
        self.cause_log_interval = int(cause_log_interval)
        self.monitor_data_pipeline = bool(monitor_data_pipeline)
        self.run_context = run_context or {}

        self._perf_last_global_step = None
        self._perf_last_time = None
        self._step_times = deque(maxlen=self.window_size)
        self._tok_sec_ema = None
        self._last_data_wait_s = None
        self._data_wait_times = deque(maxlen=self.window_size)

        self._event_stack = []
        self._last_event_end = None

        self._data_metrics = defaultdict(lambda: deque(maxlen=self.window_size))
        self._data_last_meta = {}
        self._data_last_report_step = -1
        self._cause_last_report_step = -1

        if self.enabled:
            self.log_run_context()

    @staticmethod
    def _rank_info():
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        return 0, 1

    @classmethod
    def _is_rank_zero(cls):
        rank, _ = cls._rank_info()
        return rank == 0

    @staticmethod
    def _percentile(sorted_values, q):
        if not sorted_values:
            return 0.0
        idx = min(int(len(sorted_values) * q), len(sorted_values) - 1)
        return sorted_values[idx]

    @staticmethod
    def _mean(values):
        return sum(values) / max(len(values), 1)

    @staticmethod
    def _as_ms(seconds):
        return float(seconds) * 1000.0

    def _dist_min_mean_max(self, value, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local = torch.tensor(float(value), device=device)
        t_min = local.clone()
        t_max = local.clone()
        t_sum = local.clone()
        rank, world = self._rank_info()
        if world > 1:
            dist.all_reduce(t_min, op=dist.ReduceOp.MIN)
            dist.all_reduce(t_max, op=dist.ReduceOp.MAX)
            dist.all_reduce(t_sum, op=dist.ReduceOp.SUM)
        return t_min.item(), (t_sum / world).item(), t_max.item(), world

    def log_run_context(self):
        """打印用于判定 SFT/预训练差异的运行上下文。"""
        if not self.enabled or not self._is_rank_zero():
            return

        context = {
            "dataset_name": self.run_context.get("dataset_name"),
            "num_nodes": self.run_context.get("num_nodes"),
            "num_gpus": self.run_context.get("num_gpus"),
            "float_precision": self.run_context.get("float_precision"),
            "compile_model": self.run_context.get("compile_model"),
            "compile_mode": self.run_context.get("compile_mode"),
            "optimizer": self.run_context.get("optimizer"),
            "peak_learning_rate": self.run_context.get("peak_learning_rate"),
            "weight_decay": self.run_context.get("weight_decay"),
            "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
            "NODE_COUNT": os.environ.get("NODE_COUNT", ""),
            "PROC_PER_NODE": os.environ.get("PROC_PER_NODE", ""),
            "WORLD_SIZE": os.environ.get("WORLD_SIZE", ""),
        }
        items = " ".join(f"{key}={value}" for key, value in context.items())
        print(f"[perf_context] {items}", flush=True)
        print(
            "[perf_causes] monitor_map="
            "data_pipeline(parquet_read_ms,tokenize_ms,bestfit_scan_ms,gpu_copy_ms,doc_buffer_len,data_wait_ms);"
            "ddp_straggler(rank_step_skew,rank_data_wait_skew,rank_data_pipeline_skew);"
            "allocator(cuda_reserved_gb,inactive_split_gb,alloc_retries,ooms);"
            "events(validation/checkpoint durations plus post_event_step);"
            "external_gpu(nvidia_smi util,power,clocks,pstate);"
            "external_system(pidstat/iostat cpu,iowait,disk,network)",
            flush=True,
        )

    def mark_batch_start(self, global_step=None):
        """在 train batch start 调用，估计上一个 batch 结束到当前 batch 开始之间的数据等待时间。"""
        if not self.enabled:
            return
        now = time.perf_counter()
        if hasattr(self, "_last_batch_end_time") and self._last_batch_end_time is not None:
            self._last_data_wait_s = max(0.0, now - self._last_batch_end_time)
            self._data_wait_times.append(self._last_data_wait_s)
        self._last_batch_start_time = now

    def mark_batch_end(self, global_step=None):
        if not self.enabled:
            return
        self._last_batch_end_time = time.perf_counter()

    def on_optimizer_step(
        self,
        global_step,
        num_gpus,
        batch_size_per_device,
        context_length,
        accumulate_grad_batches,
        device=None,
    ):
        """在 optimizer step 完成后记录 rolling 吞吐与 rank 间 skew。"""
        if not self.enabled:
            return

        now = time.perf_counter()
        gs = int(global_step)

        if self._perf_last_global_step is None:
            self._perf_last_global_step = gs
            self._perf_last_time = now
            return

        if gs <= self._perf_last_global_step:
            return

        step_dt = now - self._perf_last_time
        self._perf_last_time = now
        self._perf_last_global_step = gs
        self._step_times.append(step_dt)

        tokens_per_opt_step = (
            int(num_gpus)
            * int(batch_size_per_device)
            * int(context_length)
            * int(accumulate_grad_batches)
        )

        window_time = sum(self._step_times)
        tok_sec_w = tokens_per_opt_step * len(self._step_times) / max(window_time, 1e-9)

        if self._tok_sec_ema is None:
            self._tok_sec_ema = tok_sec_w
        else:
            self._tok_sec_ema = self.ema_alpha * tok_sec_w + (1 - self.ema_alpha) * self._tok_sec_ema

        step_min, step_mean, step_max, _ = self._dist_min_mean_max(step_dt, device=device)
        step_skew = step_max / max(step_min, 1e-6)

        step_times_list = sorted(self._step_times)
        step_p50 = self._percentile(step_times_list, 0.5)
        step_p90 = self._percentile(step_times_list, 0.9)

        data_wait_last = self._last_data_wait_s or 0.0
        data_wait_min, data_wait_mean, data_wait_max, _ = self._dist_min_mean_max(data_wait_last, device=device)
        data_wait_skew = data_wait_max / max(data_wait_min, 1e-6)

        if self._is_rank_zero() and gs % self.log_interval == 0:
            post_event = ""
            if self._last_event_end is not None:
                post_steps = gs - int(self._last_event_end.get("step") or gs)
                post_event = f" post_event={self._last_event_end['name']}+{post_steps}steps"
            print(
                f"[perf] step={gs} "
                f"tok_sec_w{self.window_size}={tok_sec_w:,.0f} "
                f"tok_sec_ema={self._tok_sec_ema:,.0f} "
                f"step_ms_last={self._as_ms(step_dt):.1f} "
                f"step_ms_p50={self._as_ms(step_p50):.1f} "
                f"step_ms_p90={self._as_ms(step_p90):.1f} "
                f"rank_step_min_ms={self._as_ms(step_min):.1f} "
                f"rank_step_mean_ms={self._as_ms(step_mean):.1f} "
                f"rank_step_max_ms={self._as_ms(step_max):.1f} "
                f"rank_step_skew={step_skew:.2f} "
                f"rank_data_wait_min_ms={self._as_ms(data_wait_min):.1f} "
                f"rank_data_wait_mean_ms={self._as_ms(data_wait_mean):.1f} "
                f"rank_data_wait_max_ms={self._as_ms(data_wait_max):.1f} "
                f"rank_data_wait_skew={data_wait_skew:.2f}"
                f"{post_event}",
                flush=True,
            )

        if gs % self.cuda_mem_interval == 0 and self._is_rank_zero():
            self._log_cuda_memory_stats(gs)

        if gs % self.data_log_interval == 0:
            self.log_data_pipeline_summary(gs, device=device)

        if gs % self.cause_log_interval == 0:
            self.log_cause_snapshot(gs, device=device)

    def _log_cuda_memory_stats(self, global_step):
        """记录 CUDA allocator 统计。"""
        if not torch.cuda.is_available():
            return

        try:
            stats = torch.cuda.memory_stats()
            allocated_gb = torch.cuda.memory_allocated() / 1024**3
            reserved_gb = torch.cuda.memory_reserved() / 1024**3
            inactive_split_gb = stats.get("inactive_split_bytes.all.current", 0) / 1024**3
            active_gb = stats.get("active_bytes.all.current", 0) / 1024**3
            requested_gb = stats.get("requested_bytes.all.current", 0) / 1024**3
            alloc_retries = stats.get("num_alloc_retries", 0)
            oom_count = stats.get("num_ooms", 0)

            print(
                f"[cuda_mem] step={global_step} "
                f"alloc={allocated_gb:.2f}GB "
                f"reserved={reserved_gb:.2f}GB "
                f"active={active_gb:.2f}GB "
                f"requested={requested_gb:.2f}GB "
                f"inactive_split={inactive_split_gb:.2f}GB "
                f"alloc_retries={alloc_retries} "
                f"ooms={oom_count}",
                flush=True,
            )
        except Exception as e:
            print(f"[cuda_mem] Failed to get stats: {e}", flush=True)

    def record_data_pipeline(self, metric, seconds, meta=None):
        """记录预训练 nanochat 数据管线分段耗时。"""
        if not self.enabled or not self.monitor_data_pipeline:
            return
        self._data_metrics[metric].append(float(seconds))
        if meta:
            self._data_last_meta.update(meta)

    def record_data_pipeline_meta(self, **meta):
        if not self.enabled or not self.monitor_data_pipeline:
            return
        self._data_last_meta.update(meta)

    def log_data_pipeline_summary(self, global_step, device=None):
        """低频打印数据管线 rolling 统计，并聚合 rank 间 max 用于定位 straggler。"""
        if not self.enabled or not self.monitor_data_pipeline:
            return
        gs = int(global_step)
        if self._data_last_report_step == gs:
            return
        self._data_last_report_step = gs

        metrics = [
            "parquet_open_s",
            "parquet_read_s",
            "tokenize_s",
            "buffer_refill_s",
            "bestfit_scan_s",
            "pack_rows_s",
            "cpu_copy_s",
            "gpu_copy_s",
            "yield_total_s",
        ]
        local_summary = {}
        for metric in metrics:
            values = list(self._data_metrics.get(metric, []))
            if not values:
                local_summary[metric] = 0.0
                continue
            sorted_values = sorted(values)
            local_summary[metric] = self._percentile(sorted_values, 0.9)

        # 用各指标 local p90 做 rank 聚合，rank_max 越大越能说明慢 rank。
        rank_parts = []
        for metric in metrics:
            local_value = local_summary[metric]
            t_min, t_mean, t_max, _ = self._dist_min_mean_max(local_value, device=device)
            if self._is_rank_zero():
                rank_parts.append(
                    f"{metric[:-2]}_p90_ms={self._as_ms(local_value):.1f} "
                    f"rank_max_ms={self._as_ms(t_max):.1f}"
                )

        if self._is_rank_zero():
            meta = self._data_last_meta
            print(
                f"[data_pipeline] step={gs} "
                f"split={meta.get('split', '')} "
                f"pq_idx={meta.get('pq_idx', '')} "
                f"rg_idx={meta.get('rg_idx', '')} "
                f"epoch={meta.get('epoch', '')} "
                f"doc_buffer_len={meta.get('doc_buffer_len', '')} "
                f"doc_buffer_mean_len={meta.get('doc_buffer_mean_len', '')} "
                f"crop_count={meta.get('crop_count', '')} "
                + " ".join(rank_parts),
                flush=True,
            )

    def log_event_start(self, event_name, global_step=None):
        """记录 validation/checkpoint 等事件开始。"""
        if not self.enabled:
            return

        now = time.time()
        event_info = {
            "name": event_name,
            "start": now,
            "step": int(global_step) if global_step is not None else None,
        }
        self._event_stack.append(event_info)

        if self._is_rank_zero():
            step_str = f" step={global_step}" if global_step is not None else ""
            print(f"[event] {event_name}_start{step_str} time={now:.3f}", flush=True)

    def log_event_end(self, event_name=None, global_step=None):
        """记录 validation/checkpoint 等事件结束。"""
        if not self.enabled:
            return

        now = time.time()
        if not self._event_stack:
            if self._is_rank_zero():
                print(f"[event] Warning: No event to end for '{event_name}'", flush=True)
            return

        if event_name is None:
            event_info = self._event_stack.pop()
            event_name = event_info["name"]
        else:
            event_info = None
            for i in range(len(self._event_stack) - 1, -1, -1):
                if self._event_stack[i]["name"] == event_name:
                    event_info = self._event_stack.pop(i)
                    break
            if event_info is None:
                if self._is_rank_zero():
                    print(f"[event] Warning: No matching event '{event_name}' found", flush=True)
                return

        duration = now - event_info["start"]
        end_step = int(global_step) if global_step is not None else event_info.get("step")
        self._last_event_end = {"name": event_name, "step": end_step, "time": now, "duration": duration}

        if self._is_rank_zero():
            step_str = f" step={end_step}" if end_step is not None else ""
            print(
                f"[event] {event_name}_end{step_str} time={now:.3f} duration={duration:.3f}s",
                flush=True,
            )

    def log_cause_snapshot(self, global_step, device=None):
        """按原因输出判定快照，便于直接 grep 观察。"""
        if not self.enabled:
            return
        gs = int(global_step)
        if self._cause_last_report_step == gs:
            return
        self._cause_last_report_step = gs

        if not self._is_rank_zero():
            return

        step_values = sorted(self._step_times)
        data_wait_values = sorted(self._data_wait_times)
        data_meta = self._data_last_meta
        step_p50 = self._percentile(step_values, 0.5)
        step_p90 = self._percentile(step_values, 0.9)
        data_wait_p90 = self._percentile(data_wait_values, 0.9)

        def metric_p90(name):
            values = sorted(self._data_metrics.get(name, []))
            return self._percentile(values, 0.9)

        print(
            f"[cause_snapshot] step={gs} "
            f"pretrain_data_pipeline="
            f"parquet_read_p90_ms={self._as_ms(metric_p90('parquet_read_s')):.1f},"
            f"tokenize_p90_ms={self._as_ms(metric_p90('tokenize_s')):.1f},"
            f"bestfit_scan_p90_ms={self._as_ms(metric_p90('bestfit_scan_s')):.1f},"
            f"gpu_copy_p90_ms={self._as_ms(metric_p90('gpu_copy_s')):.1f},"
            f"doc_buffer_len={data_meta.get('doc_buffer_len', '')}; "
            f"ddp_straggler=step_p50_ms={self._as_ms(step_p50):.1f},"
            f"step_p90_ms={self._as_ms(step_p90):.1f},"
            f"data_wait_p90_ms={self._as_ms(data_wait_p90):.1f}; "
            f"event_amplifier=last_event={self._last_event_end}; "
            f"allocator=see_cuda_mem_lines; "
            f"external=see_nvidia_smi_pidstat_iostat_logs",
            flush=True,
        )

    def get_summary(self):
        """获取监控摘要。"""
        if not self.enabled or len(self._step_times) == 0:
            return {}

        step_times_list = sorted(self._step_times)
        n = len(step_times_list)

        return {
            "window_size": self.window_size,
            "num_samples": n,
            "step_time_mean": self._mean(step_times_list),
            "step_time_p50": self._percentile(step_times_list, 0.5),
            "step_time_p90": self._percentile(step_times_list, 0.9),
            "tok_sec_ema": self._tok_sec_ema,
            "data_wait_mean": self._mean(list(self._data_wait_times)),
            "last_data_meta": dict(self._data_last_meta),
        }
