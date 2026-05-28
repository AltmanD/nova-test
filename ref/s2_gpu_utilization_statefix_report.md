# S2 GPU 利用率下降问题修复报告

形成日期：2026-05-28

## 结论

GPU 利用率下降的主因不是 SIGReg，也不是 crop 分支本身，而是训练迭代中每个 microbatch 都构造完整 dataloader `state_dict()`，导致反复深拷贝 `doc_buffer`。修复后，训练迭代只返回 lightweight state，checkpoint 保存仍使用完整 exact-resume state。

新日志显示修复有效：`state_dict_avg_ms` 从 `38.98ms` 降到 `0.00ms`，`dl_avg_ms` 从 `40.48ms` 降到 `1.34ms`，active dmon SM 平均值从约 `92%` 回升到约 `98%`。

## 问题表现

| 现象 | 修复前表现 |
| --- | --- |
| 吞吐下降 | `tok_s` 从 front `17730` 降到 late `16401` |
| step 变慢 | `step_ms` 从 front `29581ms` 升到 late `31977ms` |
| GPU 等数据 | `wait_avg_ms` 从 front `69.93ms` 升到 late `122.10ms` |
| dataloader 变慢 | `dl_avg_ms` 从 front `19.50ms` 升到 late `57.78ms` |
| GPU SM 下降 | active dmon SM 从 front `94.49%` 降到约 `91%` |

## 根因

`StatefulBestFitDataLoader.state_dict()` 会深拷贝 `doc_buffer`：

```python
"doc_buffer": [list(doc) for doc in self.doc_buffer]
```

这个 full state 对 checkpoint exact resume 是必要的，但训练迭代每个 microbatch 都调用它没有必要。随着 `doc_buffer` 中 tokenized docs 的长度分布变化，深拷贝成本持续上升，形成链条：

`state_dict_avg_ms` 上升 -> `dl_avg_ms` 上升 -> `wait_avg_ms` 上升 -> GPU 空等增加 -> `tok_s` 下降。

旧日志相关性验证：

| 相关性 | 数值 |
| --- | ---: |
| `state_dict_avg_ms ~ dl_avg_ms` | `0.998` |
| `state_dict_avg_ms ~ wait_avg_ms` | `0.998` |
| `dl_avg_ms ~ tok_s` | `-0.940` |

## Debug 方式

1. 在训练中加入 optimizer-step 粒度 profile 行：
   - `[profile_step]`: `tok_s`, `step_ms`, loss, microbatch 数。
   - `[profile_timing]`: data wait, forward, backward, optimizer。
   - `[profile_data]`: dataloader 内部 fetch/tokenize/packing/copy/state_dict。
   - `[profile_state]`: parquet 位置、doc buffer、packed/cropped 文档分布。
2. 用 `nvidia-smi dmon` 同步采集 GPU SM、功耗、显存。
3. 用 `profile_log_summary.py` 将日志切成 front/middle/late 三段，对比趋势。
4. 逐步细拆 dataloader 后确认：`tensor_materialize_ms`、`row_write_ms`、`crop_branch_ms` 都很小，`state_dict_ms` 才是主要增长项。

## 修改方案

### 1. 区分训练迭代 state 和 checkpoint state

文件：`nanochat/nanochat/dataloader.py`

- 保留 `state_dict()`：用于 checkpoint exact resume，继续包含完整 `doc_buffer`。
- 新增 `lightweight_state_dict()`：只包含流式位置，不复制 `doc_buffer`。
- `__iter__()` 中从 `self.state_dict()` 改为 `self.lightweight_state_dict()`。

核心语义：

```python
def lightweight_state_dict(self):
    return {
        "state_version": EXACT_RESUME_STATE_VERSION,
        "pq_idx": self.next_pq_idx,
        "rg_idx": self.next_rg_idx,
        "epoch": self.next_epoch,
        "doc_batch_index": self.next_doc_batch_index,
    }
```

### 2. 保留 checkpoint exact resume

文件：`openebm/elm/dataset.py`

`get_dataloader_state()` 仍调用 active loader 的 full `state_dict()`，因此保存 checkpoint 时仍能保存 `doc_buffer`，不牺牲恢复精度。

### 3. 增加可复用 profile 工具

相关文件：

- `openebm/elm/train.py`: 增加 `--profile_training_pipeline`。
- `openebm/elm/trainer.py`: 输出 `[profile_step]`、`[profile_timing]`、`[profile_data]`、`[profile_state]`。
- `openebm/elm/scripts/profile_log_summary.py`: 解析日志并输出 CSV/Markdown。
- `openebm/elm/runs/rjob/run_ebt_s2_profile_nosigreg_1node_8gpu.sh`: 1 node 8 GPU 评测脚本，默认 SIGReg disabled。

### 4. 增加回归测试

相关文件：

- `openebm/elm/tests/test_dataloader_lightweight_state.py`
- `openebm/elm/tests/test_profile_log_summary.py`

测试覆盖：

- full `state_dict()` 包含 `doc_buffer`；
- `lightweight_state_dict()` 不包含 `doc_buffer`；
- 训练迭代调用 `lightweight_state_dict()`，不再调用 full `state_dict()`；
- profile parser 能解析新增指标。

## 修复结果

| 指标 | 修复前整体均值 | 修复后整体均值 | 变化 |
| --- | ---: | ---: | ---: |
| `tok_s` | 17001 | 18444 | +8.49% |
| `step_ms` | 30880.28 | 28427.36 | -7.94% |
| `wait_avg_ms` | 98.50 | 46.60 | -52.69% |
| `dl_avg_ms` | 40.48 | 1.34 | -96.68% |
| `state_dict_avg_ms` | 38.98 | 0.00 | eliminated |

修复后三段稳定性：

| stage | `tok_s` | `step_ms` | `wait_avg_ms` | `dl_avg_ms` | `state_dict_avg_ms` |
| --- | ---: | ---: | ---: | ---: | ---: |
| front | 18454.22 | 28411.50 | 46.13 | 1.49 | 0.00 |
| middle | 18443.95 | 28428.51 | 46.87 | 1.29 | 0.00 |
| late | 18434.40 | 28442.29 | 46.80 | 1.25 | 0.00 |

GPU 侧 dmon：

| stage | 修复前 SM avg | 修复后 SM avg |
| --- | ---: | ---: |
| front | 94.49% | 97.86% |
| middle | 91.05% | 98.07% |
| late | 91.38% | 97.74% |

## 评估结论

这次修复已经解决 200 step 窗口内的 GPU 利用率持续降低问题。crop-heavy 分布仍存在，但新日志中 `cropped_tok` 增长没有继续推高 `dl_avg_ms`，因此当前不应把 crop 分支作为首要优化目标。

后续建议跑 `500-1000 step` 长窗口确认稳定性；若再次下降，再补 rank-level dataloader/wait 方差、NCCL/all-reduce 耗时和 per-GPU SM 分化。
