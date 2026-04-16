#!/bin/bash

################################################################################
# EBT 显存优化训练脚本 - Gradient Checkpointing + CPU Offload Optimizer + BF16
#
# 日期: 2026-04-16
# 策略:
#   1. 方案一: Gradient Checkpointing (--gradient_checkpointing)
#      - 不保存 transformer 中间激活值，backward 时重新计算
#      - 节省约 60-70% 激活值显存，代价是约 30% 额外算力
#      - 仅作用于 transformer block forward，不影响 MCMC 循环
#
#   2. 方案二: CPU Offload Optimizer (--cpu_offload_optimizer)
#      - 将 AdamW 的 m/v 状态存放在 CPU
#      - 对 medium 模型 (~405M 参数) 可节省约 3-4 GB 显存
#      - 代价: PCIe 连接机器训练速度下降 10-30%
#
#   3. 方案三: BF16 混合精度 (--float_precision bf16-mixed)
#      - 前向/反向传播使用 bfloat16，参数更新保持 float32 master copy
#      - 激活值显存减少约 50%，同时提升 Tensor Core 吞吐量
#      - BF16 动态范围与 FP32 相同，无需 loss scaling，训练更稳定
#      - 适合 Ampere+ 架构 (A100/H100/RTX 3090+)
#
# 使用场景:
#   - 单卡 24GB 显存 (如 RTX 3090/4090) 训练 medium/large 模型
#   - 多卡但每卡显存不足时
#   - 需要增大 batch size 或 context length 时
#
# 参考:
#   - NanoChat: scripts/base_train.py, nanochat/optim.py
#   - EBT 官方: https://github.com/alexiglad/EBT/blob/main/example_code/minimal_nlp_training_loop.py
################################################################################

### SLURM 配置 ###
#SBATCH --array=0
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --job-name=ebt-memory-opt
#SBATCH --output=logs/slurm/nlp/ebt-memory-opt_%A-%a.log

### 基础配置 ###
export RUN_NAME="ebt-d26-ctx2048-muon-adamw-memopt"

export MODEL_NAME="${RUN_NAME%%-*}"
export MODEL_SIZE="d26"

### 环境变量 ###
HOME="/mnt/shared-storage-user/puyuan/code/nanochat"
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"

# PyTorch 内存优化
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:1024"

# WandB 配置
export WANDB_API_KEY="968275bc822c87ac741ecce2f06cdfb54dbc1608"
export WANDB_MODE="offline"

mkdir -p logs/slurm/nlp/
module purge

################################################################################
# EBT 核心超参数 (严格遵循官方建议)
################################################################################

MCMC_STEP_SIZE=500.0
MCMC_STEP_SIZE_LR_MULTIPLIER=1500  # 3 × 500 (官方推荐比例)
MCMC_NUM_STEPS=2                    # 官方建议: 2 is very safe
EBT_TYPE="time_embed"
NORMALIZE_INITIAL_CONDITION=true
DENOISING_INITIAL_CONDITION="random_noise"
MCMC_STEP_SIZE_LEARNABLE=true
NO_MCMC_DETACH=false

################################################################################
# Batch 配置与训练步数自动计算
################################################################################

NUM_GPUS=8
DEVICE_BATCH_SIZE=1
GRAD_ACCUM=32
CONTEXT_LENGTH=2048

EFFECTIVE_BATCH_SIZE=$((NUM_GPUS * DEVICE_BATCH_SIZE * GRAD_ACCUM * CONTEXT_LENGTH))

TARGET_TOTAL_TOKENS=7340032000  # 约 7.34B tokens
MAX_STEPS=$(( TARGET_TOTAL_TOKENS / EFFECTIVE_BATCH_SIZE ))
MAX_SCHEDULING_STEPS=$MAX_STEPS

echo "自动计算的训练步数信息："
echo "  - 每步有效 Token 数: ${EFFECTIVE_BATCH_SIZE}"
echo "  - 目标总 Token 数:   ${TARGET_TOTAL_TOKENS}"
echo "  - 计算得出总 Steps:  ${MAX_STEPS}"

################################################################################
# 学习率配置
################################################################################

PEAK_LR=0.00025
WARM_UP_STEPS=0
WARM_UP_BASE_LR_DIVIDER=10
MIN_LR_SCALE=50

################################################################################
# 优化器配置
################################################################################

WEIGHT_DECAY=0.2
BETA1=0.8
BETA2=0.95
GRADIENT_CLIP_VAL=1.0

################################################################################
# 验证与数据加载配置
################################################################################

VAL_CHECK_INTERVAL=2000
LIMIT_VAL_BATCHES=50
NUM_WORKERS=8
SAVE_TOP_K=2

################################################################################
# 显存优化配置 (核心新增)
################################################################################
#
# 方案一: Gradient Checkpointing
#   - 启用: GRADIENT_CHECKPOINTING_FLAG="--gradient_checkpointing"
#   - 禁用: GRADIENT_CHECKPOINTING_FLAG=""
#
# 方案二: CPU Offload Optimizer
#   - 启用: CPU_OFFLOAD_FLAG="--cpu_offload_optimizer"
#   - 禁用: CPU_OFFLOAD_FLAG=""
#
# 方案三: BF16 混合精度
#   - 启用: FLOAT_PRECISION_FLAG="--float_precision bf16-mixed"
#   - 禁用: FLOAT_PRECISION_FLAG=""  (默认 32-true)
#
# 推荐组合:
#   - 仅 gradient checkpointing: 显存节省最多，速度影响最小
#   - GC + BF16: 激活值 + 参数显存双重节省，速度通常更快
#   - 三者同时启用: 最大显存节省，适合极端显存不足场景
################################################################################

# 方案一: Gradient Checkpointing (推荐优先启用)
GRADIENT_CHECKPOINTING_FLAG="--gradient_checkpointing"

# 方案二: CPU Offload Optimizer (显存仍不足时再启用)
CPU_OFFLOAD_FLAG="--cpu_offload_optimizer"

# 方案三: BF16 混合精度 (Ampere+ 架构推荐启用，可同时节省显存并加速)
FLOAT_PRECISION_FLAG="--float_precision bf16-mixed"

# 如需禁用某个方案，注释对应行并取消注释下面的空值行:
# GRADIENT_CHECKPOINTING_FLAG=""
# CPU_OFFLOAD_FLAG=""
# FLOAT_PRECISION_FLAG=""

################################################################################
# 优化器与 LR 调度配置
################################################################################

OPTION_FLAGS="--dynamic_wd --linear_warmdown --warmup_ratio 0.0 --warmdown_ratio 0.5 --final_lr_frac 0.0 --optimizer muon_adamw --muon_lr 0.02 --muon_momentum 0.95 --muon_ns_steps 5 --muon_beta2 0.95 --adamw_embedding_lr 0.3 --adamw_vocab_to_embed_lr 0.01 --adamw_scalar_lr 0.04 --adamw_dmodel_lr_scaling"

################################################################################
# torch.compile 配置
################################################################################

COMPILE_FLAGS="--compile_model --compile_mode full"

################################################################################
# WandB 配置
################################################################################

WANDB_FLAGS=""
# WANDB_FLAGS="--disable_wandb"

################################################################################
# 日志配置
################################################################################

current_time=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/${current_time}_${MODEL_NAME}_${MODEL_SIZE}_ctx${CONTEXT_LENGTH}_memopt_gpu${NUM_GPUS}.log"
mkdir -p logs

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

print_separator() {
    local char=${1:-"═"}
    local width=${2:-80}
    printf "${BLUE}"
    printf '%*s' "$width" | tr ' ' "$char"
    printf "${NC}\n"
}

print_header() {
    local title=$1
    echo ""
    print_separator "═"
    echo -e "${BOLD}${GREEN}  $title${NC}"
    print_separator "═"
}

print_kv() {
    local key=$1
    local value=$2
    local width=${3:-30}
    printf "  ${DIM}%-${width}s${NC} %s\n" "$key:" "$value"
}

################################################################################
# 显示配置
################################################################################

print_header "EBT 显存优化训练配置"

echo ""
echo -e "${CYAN}▶ 显存优化策略${NC}"
print_separator "─" 60
if [[ -n "$GRADIENT_CHECKPOINTING_FLAG" ]]; then
    echo "  ${GREEN}✓${NC} Gradient Checkpointing 已启用"
    echo "    - 不保存 transformer 中间激活值，backward 时重新计算"
    echo "    - 节省约 60-70% 激活值显存，代价约 30% 额外算力"
    echo "    - 仅作用于 transformer block，不影响 MCMC 循环"
else
    echo "  ${YELLOW}○${NC} Gradient Checkpointing 未启用"
fi
if [[ -n "$CPU_OFFLOAD_FLAG" ]]; then
    echo "  ${GREEN}✓${NC} CPU Offload Optimizer 已启用"
    echo "    - AdamW m/v 状态存放在 CPU，节省约 3-4 GB 显存"
    echo "    - 注意: PCIe 连接机器可能导致训练速度下降 10-30%"
else
    echo "  ${YELLOW}○${NC} CPU Offload Optimizer 未启用"
fi
if [[ -n "$FLOAT_PRECISION_FLAG" ]]; then
    echo "  ${GREEN}✓${NC} BF16 混合精度 已启用"
    echo "    - 前向/反向使用 bfloat16，激活值显存减少约 50%"
    echo "    - 无需 loss scaling，训练稳定性与 FP32 相当"
    echo "    - 需要 Ampere+ 架构 (A100/H100/RTX 3090+)"
else
    echo "  ${YELLOW}○${NC} BF16 混合精度 未启用 (使用默认 32-true)"
fi

echo ""
echo -e "${CYAN}▶ 优化器策略 (对齐 NanoChat)${NC}"
print_separator "─" 60
echo "  ${GREEN}✓${NC} Muon+AdamW 混合优化器 (nanochat/optim.py)"
echo "  ${GREEN}✓${NC} Transformer 矩阵参数 → Muon (LR=0.02)"
echo "  ${GREEN}✓${NC} Embedding/Scalar/Alpha → AdamW"
echo "  ${GREEN}✓${NC} Adam Beta: (${BOLD}${BETA1}, ${BETA2}${NC}) (对齐 NanoChat)"
echo "  ${GREEN}✓${NC} Weight Decay: ${BOLD}${WEIGHT_DECAY}${NC} + 动态衰减到 0"
echo "  ${GREEN}✓${NC} LR 调度: Linear Warmdown (后50%衰减到0)"

echo ""
echo -e "${CYAN}▶ EBT 核心参数 (官方推荐)${NC}"
print_separator "─" 60
print_kv "MCMC Step Size" "${MCMC_STEP_SIZE}"
print_kv "MCMC LR Multiplier" "${MCMC_STEP_SIZE_LR_MULTIPLIER} (3× step_size)"
print_kv "MCMC Num Steps" "${MCMC_NUM_STEPS}"
print_kv "EBT Type" "${EBT_TYPE}"

echo ""
echo -e "${CYAN}▶ 模型与 Batch 配置${NC}"
print_separator "─" 60
print_kv "Model Size" "${MODEL_SIZE}"
print_kv "Context Length" "${CONTEXT_LENGTH}"
print_kv "Device Batch Size" "${DEVICE_BATCH_SIZE}"
print_kv "Gradient Accumulation" "${GRAD_ACCUM}"
print_kv "Num GPUs" "${NUM_GPUS}"
print_kv "Effective Batch Size" "${EFFECTIVE_BATCH_SIZE} tokens/step"
print_kv "Max Steps" "${MAX_STEPS}"

echo ""
echo -e "${YELLOW}⚠ 重要说明${NC}"
echo "  1. Gradient Checkpointing 只在 training 模式下生效，eval 不受影响"
echo "  2. CPU Offload 会增加每步 CPU↔GPU 数据传输，NVLink 机器影响较小"
echo "  3. BF16 需要 Ampere+ 架构，Volta 架构请改用 16-mixed (需 loss scaling)"
echo "  4. 三个方案可以同时启用以最大化显存节省"
echo "  5. 如果仍然 OOM，考虑减小 DEVICE_BATCH_SIZE 或 CONTEXT_LENGTH"

echo ""
read -p "按 Enter 开始训练，或 Ctrl+C 取消..."

################################################################################
# 写入日志头
################################################################################

cat << LOG_HEADER > "${LOG_FILE}"
################################################################################
#                    EBT 显存优化训练日志
################################################################################
#
# Run Name:        ${RUN_NAME}_${current_time}
# Start Time:      $(date '+%Y-%m-%d %H:%M:%S')
# Log File:        ${LOG_FILE}
#
# 显存优化策略:
#   Gradient Checkpointing: ${GRADIENT_CHECKPOINTING_FLAG:-"未启用"}
#   CPU Offload Optimizer:  ${CPU_OFFLOAD_FLAG:-"未启用"}
#   BF16 混合精度:          ${FLOAT_PRECISION_FLAG:-"未启用 (32-true)"}
#
# 优化器策略 (对齐 NanoChat base_train.py):
#   1. Muon+AdamW 混合优化器 (nanochat/optim.py)
#   2. Adam Beta: (${BETA1}, ${BETA2})
#   3. Weight Decay: ${WEIGHT_DECAY} + 动态衰减到 0
#   4. LR 调度: Linear Warmdown (后50%衰减到0)
#
################################################################################

================================================================================
[SYSTEM INFO]
================================================================================
Hostname:         $(hostname)
User:             $(whoami)
Python:           $(python3 --version 2>&1)
PyTorch:          $(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "N/A")
CUDA Available:   $(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "N/A")
GPU Count:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo "N/A")
GPU Model:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "N/A")

================================================================================
[显存优化配置]
================================================================================
Gradient Checkpointing:   ${GRADIENT_CHECKPOINTING_FLAG:-"未启用"}
CPU Offload Optimizer:    ${CPU_OFFLOAD_FLAG:-"未启用"}
BF16 混合精度:            ${FLOAT_PRECISION_FLAG:-"未启用 (32-true)"}

================================================================================
[训练配置]
================================================================================
Model Size:               ${MODEL_SIZE}
Context Length:           ${CONTEXT_LENGTH}
Device Batch Size:        ${DEVICE_BATCH_SIZE}
Gradient Accumulation:    ${GRAD_ACCUM}
Effective Batch Size:     ${EFFECTIVE_BATCH_SIZE} tokens/step
Peak LR:                  ${PEAK_LR}
Weight Decay:             ${WEIGHT_DECAY}
Beta1/Beta2:              ${BETA1} / ${BETA2}
Max Steps:                ${MAX_STEPS}

================================================================================
[训练输出]
================================================================================

LOG_HEADER

################################################################################
# 自动重定向所有输出到日志文件
################################################################################
exec > >(tee -a "${LOG_FILE}") 2>&1

################################################################################
# 启动训练
################################################################################

echo ""
print_header "开始训练"
echo ""

set +e
torchrun --standalone --nproc_per_node=${NUM_GPUS} /mnt/shared-storage-user/puyuan/code/nova/nova/ebt/train.py \
--run_name ${RUN_NAME}_${current_time} \
--modality "NLP" \
--model_name ${MODEL_NAME} \
--model_size ${MODEL_SIZE} \
\
--pretokenize_dataset \
\
--normalize_initial_condition \
--ebt_type ${EBT_TYPE} \
--denoising_initial_condition ${DENOISING_INITIAL_CONDITION} \
--mcmc_step_size_learnable \
--mcmc_step_size ${MCMC_STEP_SIZE} \
--mcmc_step_size_lr_multiplier ${MCMC_STEP_SIZE_LR_MULTIPLIER} \
--mcmc_num_steps ${MCMC_NUM_STEPS} \
\
--context_length ${CONTEXT_LENGTH} \
\
--gpus "-1" \
\
--peak_learning_rate ${PEAK_LR} \
--batch_size_per_device ${DEVICE_BATCH_SIZE} \
--accumulate_grad_batches ${GRAD_ACCUM} \
--gradient_clip_val ${GRADIENT_CLIP_VAL} \
\
--weight_decay ${WEIGHT_DECAY} \
--beta1 ${BETA1} \
--beta2 ${BETA2} \
--min_lr_scale ${MIN_LR_SCALE} \
--max_steps ${MAX_STEPS} \
--max_scheduling_steps ${MAX_SCHEDULING_STEPS} \
--warm_up_steps ${WARM_UP_STEPS} \
--warm_up_base_lr_divider ${WARM_UP_BASE_LR_DIVIDER} \
\
--dataset_name "nanochat" \
--num_workers ${NUM_WORKERS} \
--val_check_interval ${VAL_CHECK_INTERVAL} \
--limit_val_batches ${LIMIT_VAL_BATCHES} \
--val_sanity 1 \
--validation_split_pct 0.0027 \
\
--wandb_project 'nlp_pretrain' \
--log_model_archi \
--set_matmul_precision "medium" \
--save_top_k_ckpts ${SAVE_TOP_K} \
--save_periodic_steps 1000 \
${WANDB_FLAGS} \
${OPTION_FLAGS} \
${COMPILE_FLAGS} \
${GRADIENT_CHECKPOINTING_FLAG} \
${CPU_OFFLOAD_FLAG} \
${FLOAT_PRECISION_FLAG}

TRAIN_EXIT_CODE=$?
set -e

################################################################################
# 训练结束处理
################################################################################

echo ""
print_header "训练结束"

if [[ $TRAIN_EXIT_CODE -eq 0 ]]; then
    echo -e "${GREEN}✓ 训练成功完成${NC}"
else
    echo -e "${RED}✗ 训练异常退出 (exit code: $TRAIN_EXIT_CODE)${NC}"

    if grep -q "CUDA out of memory\|OutOfMemoryError\|OOM" "${LOG_FILE}" 2>/dev/null; then
        echo -e "${YELLOW}⚠ 检测到 OOM - 建议尝试以下方案:${NC}"
        echo "  1. 确认 --gradient_checkpointing 已启用"
        echo "  2. 确认 --float_precision bf16-mixed 已启用"
        echo "  3. 确认 --cpu_offload_optimizer 已启用"
        echo "  4. 减小 DEVICE_BATCH_SIZE"
        echo "  5. 减小 CONTEXT_LENGTH"
    fi

    if grep -q "nan\|NaN\|inf\|Inf" "${LOG_FILE}" 2>/dev/null | head -5; then
        echo -e "${YELLOW}⚠ 检测到 NaN/Inf - 可能需要进一步降低学习率${NC}"
    fi
fi

echo ""
echo "日志文件: ${LOG_FILE}"
echo ""

################################################################################
# 监控建议
################################################################################

print_header "训练监控建议"
cat << MONITOR_HELP

实时监控命令 (在另一个终端运行):
  tail -f ${LOG_FILE} | grep -E "(train_loss|Alpha_MCMC|Global_LR)"

显存监控:
  watch -n 1 nvidia-smi --query-gpu=memory.used,memory.free --format=csv

关键指标检查:
  1. train_loss: 应该平稳下降，避免突然飙升
  2. Alpha_MCMC: MCMC 步长应该稳定在合理范围 (如 400-600)
  3. Global_LR: 学习率应该平滑衰减

显存优化效果预期:
  - Gradient Checkpointing: 激活值显存减少 60-70%，训练速度下降约 30%
  - CPU Offload Optimizer: 优化器状态显存减少 3-4 GB，PCIe 机器速度下降 10-30%
  - BF16 混合精度: 激活值/参数显存减少约 50%，Tensor Core 加速，速度通常提升

MONITOR_HELP

echo ""
