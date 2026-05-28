# S2 1Node8GPU Profiling 指标关注清单

## 运行边界

- 脚本：`openebm/elm/runs/rjob/run_ebt_s2_profile_nosigreg_1node_8gpu.sh`
- 默认资源：`NODE_COUNT=1`，`PROC_PER_NODE=8`
- SIGReg：关闭。脚本不传任何 `--sigreg_*` 参数，依赖 `train.py` 默认 `sigreg_lambda=0.0`
- 默认提前结束：`PROFILE_MAX_STEPS=220`，即大约 200 个 optimizer step 后自然结束并自动分析
- 调度步数：默认 `MAX_SCHEDULING_STEPS=1200`，避免把短 profiling run 的 LR schedule 压缩到 220 step
- 主日志：`logs_base_train/<date>/<RUN_NAME>_rank0.log`
- GPU 监控日志：`logs_base_train/<date>/<RUN_NAME>_rank0_nvidia_smi_dmon.log`
- 汇总输出：训练结束后自动生成 `<log_stem>_profile_summary/`

## 提前结束与即时分析

推荐方式是让短 profiling run 自然结束：

```bash
bash openebm/elm/runs/rjob/run_ebt_s2_profile_nosigreg_1node_8gpu.sh
```

默认 `PROFILE_MAX_STEPS=220`。如果 200 step 不够，可以覆盖：

```bash
PROFILE_MAX_STEPS=300 bash openebm/elm/runs/rjob/run_ebt_s2_profile_nosigreg_1node_8gpu.sh
```

如果已经明显看到利用率下降，可以直接在运行脚本的终端按 `Ctrl-C`，或对脚本进程发 `TERM`。脚本会：

- 停止 `nvidia-smi dmon`
- 保留已经写出的训练日志和 GPU 监控日志
- 自动运行 `profile_log_summary.py`，对部分日志生成 `profile_summary.md`

如果训练还在跑，也可以另开一个终端先分析当前已有日志：

```bash
python openebm/elm/scripts/profile_log_summary.py \
  --log logs_base_train/<date>/<RUN_NAME>_rank0.log
```

如果临时不想自动分析，可以设置：

```bash
ANALYZE_ON_EXIT=0 bash openebm/elm/runs/rjob/run_ebt_s2_profile_nosigreg_1node_8gpu.sh
```

## 第一优先级：GPU 利用率是否持续降低

看 `*_nvidia_smi_dmon.log`：

- `sm`：核心 GPU utilization。若持续下降，这是要解释的主现象。
- `mem`：显存带宽/内存利用。`sm` 低但 `mem` 高，可能偏 memory-bound 或数据搬运。
- `pwr`：功耗。`sm` 和 `pwr` 同时下降，通常表示 GPU 等待更多。
- `pclk` / `mclk`：核心/显存时钟。若降频，需要先排除功耗、温度或集群策略影响。
- `fb`：显存占用。若持续上涨，关注是否有显存泄漏或缓存增长。

## 第二优先级：吞吐是否真的下降

看训练主日志里的 `[profile_step]`：

- `tok_s`：optimizer-step 级别 tokens/sec，核心吞吐指标。
- `step_ms`：一个 optimizer step 的 wall time。`tok_s` 下降应对应 `step_ms` 上升。
- `mb`：聚合的 micro-batch 数，应接近 `accumulate_grad_batches=32`。
- `loss`：确认评测过程没有明显训练异常。

## 第三优先级：下降来自哪里

看 `[profile_timing]`：

- `wait_avg_ms` / `wait_max_ms`：GPU 等下一批数据的时间。持续上升时优先怀疑 dataloader、parquet 读取、tokenize 或 packing。
- `fw_avg_ms`：forward 时间。上升更偏模型计算路径或 shape/compile 行为。
- `bw_avg_ms`：backward 时间。上升更偏 autograd/MCMC/通信等待。
- `opt_ms`：optimizer step 时间。上升时关注 Muon/AdamW step 或 DDP 同步。

看 `[profile_data]`：

- `dl_avg_ms` / `dl_max_ms`：dataloader 单 batch 总耗时。
- `fetch_avg_ms`：parquet/row-group 读取耗时。
- `tok_avg_ms` / `tok_max_ms`：tokenizer 耗时。
- `pack_avg_ms` / `pack_max_ms`：best-fit packing 扫描耗时。
- `copy_avg_ms`：CPU 到 GPU copy 耗时。

看 `[profile_state]`：

- `pq` / `rg` / `doc_batch`：定位吞吐下降发生在哪些 parquet shard / row group。
- `docbuf_end` / `docbuf_max`：doc buffer 是否异常波动。
- `refill_calls`：补充 doc buffer 的次数。
- `packed_docs` / `cropped_docs`：文档 packing 行为。
- `packed_tok` / `cropped_tok`：token 利用和裁剪量；若 `cropped_tok` 后期升高且 `pack_avg_ms` 升高，重点看对应 shard 的文档长度分布。

## 快速判断矩阵

| 现象 | 优先怀疑 |
| --- | --- |
| `sm` 下降，`wait_avg_ms` 上升，`dl_avg_ms` 上升 | dataloader 供给不足 |
| `tok_avg_ms` 上升 | tokenization 变慢 |
| `pack_avg_ms` 上升，`cropped_tok` 上升 | best-fit packing 与文档长度分布问题 |
| `fw_avg_ms` / `bw_avg_ms` 上升，但 `wait_avg_ms` 不变 | 模型计算路径变慢 |
| `opt_ms` 上升 | optimizer step 或 DDP 同步 |
| `sm` 下降，`pclk` / `mclk` 下降 | 先排查降频、功耗或温度策略 |
| `fb` 持续上涨 | 显存占用增长或缓存行为 |

## 汇总命令

训练结束后脚本会自动运行：

```bash
python openebm/elm/scripts/profile_log_summary.py --log <rank0.log> --label <RUN_NAME>
```

也可以手动重跑：

```bash
python openebm/elm/scripts/profile_log_summary.py \
  --log logs_base_train/<date>/<RUN_NAME>_rank0.log
```

优先看生成的 `profile_summary.md` 中 front / middle / late 三段对比，确认 `tok_s`、`step_ms`、`wait_avg_ms`、`tok_avg_ms`、`pack_avg_ms`、`cropped_tok` 哪些指标从前段到后段发生了单调恶化。
