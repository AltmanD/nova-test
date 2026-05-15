#!/bin/bash

################################################################################
# EBT d26 profiling training script - 2 nodes x 8 GPUs
#
# Purpose:
#   Validate whether late-stage GPU utilization decline mainly comes from:
#   1. dataloader wait / parquet progress
#   2. tokenization / best-fit packing cost
#   3. forward-backward / optimizer step time
#
# Based on:
#   openebm/elm/runs/rjob/run_ebt_2node_8gpu.sh
#
# Usage:
#   Submit via rjob with the same environment as the base 2node8gpu script, or:
#   NODE_RANK=0 NODE_COUNT=1 MASTER_ADDR=127.0.0.1 PROC_PER_NODE=8 \
#   bash run_ebt_2node_8gpu_profile.sh
################################################################################

if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate /mnt/shared-storage-user/luyudong/conda_envs/ebt
    export LD_LIBRARY_PATH="/mnt/shared-storage-user/luyudong/conda_envs/ebt/lib:${LD_LIBRARY_PATH}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NOVA_HOME="${NOVA_HOME:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"
NANOCHAT_HOME="/mnt/shared-storage-user/luyudong/nanochat"
TRAIN_SCRIPT="${NOVA_HOME}/openebm/elm/train.py"

cd "${NOVA_HOME}"
export PYTHONPATH="${NOVA_HOME}:${PYTHONPATH}"

RUN_PREFIX="2node-8gpu-profile"

export MODEL_NAME="ebt"
export MODEL_SIZE="d26"

HOME="${NANOCHAT_HOME}"
export NANOCHAT_BASE_DIR="${HOME}/.cache/nanochat"

# Keep the same allocator setting as the base script so profiling stays comparable.
export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.6"
export WANDB_MODE="offline"

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NCCL_IB_HCA=$NCCL_IB_HCA"
echo "NCCL_IB_GID_INDEX=$NCCL_IB_GID_INDEX"
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_IB_TIMEOUT=60
export NCCL_IB_RETRY_CNT=20

NODE_COUNT=${NODE_COUNT:-2}
PROC_PER_NODE=${PROC_PER_NODE:-8}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
JOB_ID=${JOB_ID:-$$}
MASTER_PORT=$(( 20000 + 0x$(echo "${JOB_ID}" | md5sum | head -c 4) % 10000 ))

NUM_NODES=${NODE_COUNT}
GPUS_PER_NODE=${PROC_PER_NODE}
NUM_GPUS=$((NUM_NODES * GPUS_PER_NODE))
WORLD_SIZE=${NUM_GPUS}

echo "=== Distributed Config ==="
echo "NODE_RANK:      ${NODE_RANK}"
echo "NODE_COUNT:     ${NODE_COUNT}"
echo "MASTER_ADDR:    ${MASTER_ADDR}"
echo "MASTER_PORT:    ${MASTER_PORT}"
echo "PROC_PER_NODE:  ${PROC_PER_NODE}"
echo "WORLD_SIZE:     ${WORLD_SIZE}"

MCMC_STEP_SIZE=500.0
MCMC_STEP_SIZE_LR_MULTIPLIER=1500
MCMC_NUM_STEPS=2
EBT_TYPE="time_embed"
DENOISING_INITIAL_CONDITION="random_noise"
USE_SDPA_ATTENTION=false

DEVICE_BATCH_SIZE=1
GRAD_ACCUM=32
CONTEXT_LENGTH=2048

EFFECTIVE_BATCH_SIZE=$((NUM_GPUS * DEVICE_BATCH_SIZE * GRAD_ACCUM * CONTEXT_LENGTH))

# Shorter run for profiling. Enough to cross multiple shards / row groups.
MAX_STEPS=${MAX_STEPS_OVERRIDE:-1200}
MAX_SCHEDULING_STEPS=${MAX_STEPS}

PEAK_LR=0.00025
WARM_UP_STEPS=0
WARM_UP_BASE_LR_DIVIDER=10
MIN_LR_SCALE=50

WEIGHT_DECAY=0.2
BETA1=0.8
BETA2=0.95
GRADIENT_CLIP_VAL=1.0

# Make validation less intrusive but still present, so we can separate steady-state
# decline from periodic validation/checkpoint interruptions.
VAL_CHECK_INTERVAL=${VAL_CHECK_INTERVAL_OVERRIDE:-400}
LIMIT_VAL_BATCHES=${LIMIT_VAL_BATCHES_OVERRIDE:-10}
SAVE_TOP_K=1
SAVE_PERIODIC_STEPS=${SAVE_PERIODIC_STEPS_OVERRIDE:-400}

OPTION_FLAGS="--dynamic_wd --linear_warmdown --warmup_ratio 0.0057 --warmdown_ratio 0.65 --final_lr_frac 0.05 \
--optimizer muon_adamw --muon_lr 0.02 --muon_momentum 0.95 --muon_ns_steps 5 --muon_beta2 0.95 \
--adamw_embedding_lr 0.3 --adamw_vocab_to_embed_lr 0.01 --adamw_scalar_lr 0.04 --adamw_dmodel_lr_scaling \
--muon_momentum_warmup_steps 0"

# compile stays aligned with the base script; the profiling goal is to compare
# against current production behavior rather than introduce a new optimization path.
COMPILE_FLAGS="--compile_model --compile_mode transformer_only"

SDPA_FLAGS=""
if [[ "${USE_SDPA_ATTENTION}" == "true" ]]; then
    SDPA_FLAGS="--use_sdpa_attention"
fi

WANDB_FLAGS=""

TIMESTAMP=$(date +"%m%d_%H%M")
DATE_DIR=$(date +"%Y%m%d")
CONFIG_TAG="${MODEL_SIZE}_profile_ctx${CONTEXT_LENGTH}_bs$((NUM_GPUS * DEVICE_BATCH_SIZE * GRAD_ACCUM))_lr${PEAK_LR}_${NUM_NODES}nodes_${GPUS_PER_NODE}gpus"
export RUN_NAME="${RUN_PREFIX}_${TIMESTAMP}_${CONFIG_TAG}"

LOG_DIR="${NOVA_HOME}/logs_base_train/${DATE_DIR}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${RUN_NAME}_rank${NODE_RANK}.log"

echo "=== Profiling Run ==="
echo "RUN_NAME: ${RUN_NAME}"
echo "LOG_FILE: ${LOG_FILE}"
echo "MAX_STEPS: ${MAX_STEPS}"
echo "VAL_CHECK_INTERVAL: ${VAL_CHECK_INTERVAL}"
echo "LIMIT_VAL_BATCHES: ${LIMIT_VAL_BATCHES}"
echo "SAVE_PERIODIC_STEPS: ${SAVE_PERIODIC_STEPS}"
echo "PROFILE LOG KEYS:"
echo "  [profile_step]   optimizer-step summary (loss / tok_s / step_ms / micro-batches)"
echo "  [profile_timing] wait/forward/backward/optimizer timing summary"
echo "  [profile_data]   dataloader fetch/tokenize/packing/copy summary"
echo "  [profile_state]  parquet progress / doc_buffer / packed-cropped stats"

exec > >(tee -a "${LOG_FILE}") 2>&1

set +e

torchrun \
  --nnodes="${NUM_NODES}" \
  --nproc_per_node="${GPUS_PER_NODE}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  --rdzv_id="${JOB_ID}" \
  "${TRAIN_SCRIPT}" \
  --run_name "${RUN_NAME}" \
  --modality "NLP" \
  --model_name "${MODEL_NAME}" \
  --model_size "${MODEL_SIZE}" \
  \
  --pretokenize_dataset \
  --profile_training_pipeline \
  \
  --normalize_initial_condition \
  --ebt_type "${EBT_TYPE}" \
  --denoising_initial_condition "${DENOISING_INITIAL_CONDITION}" \
  --mcmc_step_size_learnable \
  --mcmc_step_size "${MCMC_STEP_SIZE}" \
  --mcmc_step_size_lr_multiplier "${MCMC_STEP_SIZE_LR_MULTIPLIER}" \
  --mcmc_num_steps "${MCMC_NUM_STEPS}" \
  \
  --context_length "${CONTEXT_LENGTH}" \
  \
  --gpus "-1" \
  \
  --peak_learning_rate "${PEAK_LR}" \
  --batch_size_per_device "${DEVICE_BATCH_SIZE}" \
  --accumulate_grad_batches "${GRAD_ACCUM}" \
  --gradient_clip_val "${GRADIENT_CLIP_VAL}" \
  \
  --weight_decay "${WEIGHT_DECAY}" \
  --beta1 "${BETA1}" \
  --beta2 "${BETA2}" \
  --min_lr_scale "${MIN_LR_SCALE}" \
  --max_steps "${MAX_STEPS}" \
  --max_scheduling_steps "${MAX_SCHEDULING_STEPS}" \
  --warm_up_steps "${WARM_UP_STEPS}" \
  --warm_up_base_lr_divider "${WARM_UP_BASE_LR_DIVIDER}" \
  \
  --dataset_name "nanochat" \
  --val_check_interval "${VAL_CHECK_INTERVAL}" \
  --limit_val_batches "${LIMIT_VAL_BATCHES}" \
  --val_sanity 1 \
  --validation_split_pct 0.0027 \
  \
  --wandb_project 'nlp_pretrain' \
  --disable_wandb \
  --log_every_n_steps 1 \
  --set_matmul_precision "medium" \
  --float_precision "bf16-true" \
  --manual_gc_collect_every_n_steps -1 \
  --save_top_k_ckpts "${SAVE_TOP_K}" \
  --save_periodic_steps "${SAVE_PERIODIC_STEPS}" \
  ${WANDB_FLAGS} \
  ${OPTION_FLAGS} \
  ${COMPILE_FLAGS} \
  ${SDPA_FLAGS}

TRAIN_EXIT_CODE=$?
set -e

echo ""
if [[ $TRAIN_EXIT_CODE -eq 0 ]]; then
    echo "Training finished successfully (rank ${NODE_RANK})"
else
    echo "Training exited abnormally (exit code: $TRAIN_EXIT_CODE, rank: ${NODE_RANK})"
fi

echo "Log file: ${LOG_FILE}"
