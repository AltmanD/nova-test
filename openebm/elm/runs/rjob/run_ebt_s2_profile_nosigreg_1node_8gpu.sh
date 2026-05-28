#!/bin/bash

################################################################################
# EBT d26 S2 profiling run - 1 node x 8 GPUs, SIGReg disabled
#
# Purpose:
#   Observe whether late-stage GPU utilization decline comes from dataloader
#   wait, tokenization/packing, model forward/backward, or optimizer step time.
#
# Usage:
#   bash openebm/elm/runs/rjob/run_ebt_s2_profile_nosigreg_1node_8gpu.sh
#
# Early-stop controls:
#   PROFILE_MAX_STEPS=220  # default short profiling run
#   Ctrl-C / TERM          # stop early and summarize the partial log
################################################################################

if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate /mnt/shared-storage-user/luyudong/conda_envs/ebt
    export LD_LIBRARY_PATH="/mnt/shared-storage-user/luyudong/conda_envs/ebt/lib:${LD_LIBRARY_PATH}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NOVA_HOME="${NOVA_HOME:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"
NANOCHAT_HOME="${NANOCHAT_HOME:-/mnt/shared-storage-user/luyudong/nanochat}"
TRAIN_SCRIPT="${NOVA_HOME}/openebm/elm/train.py"
SUMMARY_SCRIPT="${NOVA_HOME}/openebm/elm/scripts/profile_log_summary.py"

cd "${NOVA_HOME}"
export PYTHONPATH="${NOVA_HOME}:${PYTHONPATH}"

RUN_PREFIX="${RUN_PREFIX:-s2-profile-nosigreg-1node-8gpu-bf16mixed}"

export MODEL_NAME="${MODEL_NAME:-ebt}"
export MODEL_SIZE="${MODEL_SIZE:-d26}"

HOME="${NANOCHAT_HOME}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-${HOME}/.cache/nanochat}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-garbage_collection_threshold:0.6}"
export WANDB_MODE="${WANDB_MODE:-offline}"

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NCCL_IB_HCA=$NCCL_IB_HCA"
echo "NCCL_IB_GID_INDEX=$NCCL_IB_GID_INDEX"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT:-60}"
export NCCL_IB_RETRY_CNT="${NCCL_IB_RETRY_CNT:-20}"

NODE_COUNT=${NODE_COUNT:-1}
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

MCMC_STEP_SIZE=${MCMC_STEP_SIZE:-500.0}
MCMC_STEP_SIZE_LR_MULTIPLIER=${MCMC_STEP_SIZE_LR_MULTIPLIER:-1500}
MCMC_NUM_STEPS=${MCMC_NUM_STEPS:-2}
EBT_TYPE="${EBT_TYPE:-time_embed}"
DENOISING_INITIAL_CONDITION="${DENOISING_INITIAL_CONDITION:-random_noise}"
USE_SDPA_ATTENTION="${USE_SDPA_ATTENTION:-false}"

DEVICE_BATCH_SIZE=${DEVICE_BATCH_SIZE:-1}
GRAD_ACCUM=${GRAD_ACCUM:-32}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-2048}

EFFECTIVE_BATCH_SIZE=$((NUM_GPUS * DEVICE_BATCH_SIZE * GRAD_ACCUM * CONTEXT_LENGTH))

PROFILE_MAX_STEPS=${PROFILE_MAX_STEPS:-220}
MAX_STEPS=${MAX_STEPS_OVERRIDE:-${PROFILE_MAX_STEPS}}
MAX_SCHEDULING_STEPS=${MAX_SCHEDULING_STEPS_OVERRIDE:-1200}

PEAK_LR=${PEAK_LR:-0.0012}
WARM_UP_STEPS=${WARM_UP_STEPS:-0}
WARM_UP_BASE_LR_DIVIDER=${WARM_UP_BASE_LR_DIVIDER:-10}
MIN_LR_SCALE=${MIN_LR_SCALE:-50}

WEIGHT_DECAY=${WEIGHT_DECAY:-0.2}
BETA1=${BETA1:-0.8}
BETA2=${BETA2:-0.95}
GRADIENT_CLIP_VAL=${GRADIENT_CLIP_VAL:-1.0}

VAL_CHECK_INTERVAL=${VAL_CHECK_INTERVAL_OVERRIDE:-400}
LIMIT_VAL_BATCHES=${LIMIT_VAL_BATCHES_OVERRIDE:-10}
SAVE_TOP_K=${SAVE_TOP_K:-1}
SAVE_PERIODIC_STEPS=${SAVE_PERIODIC_STEPS_OVERRIDE:-400}

OPTION_FLAGS="--dynamic_wd --linear_warmdown --warmup_ratio 0.0057 --warmdown_ratio 0.65 --final_lr_frac 0.05 \
--optimizer muon_adamw --muon_lr 0.02 --muon_momentum 0.95 --muon_ns_steps 5 --muon_beta2 0.95 \
--adamw_embedding_lr 0.3 --adamw_vocab_to_embed_lr 0.01 --adamw_scalar_lr 0.04 --adamw_dmodel_lr_scaling \
--muon_momentum_warmup_steps 300 --ffn_dim_multiplier 2.67"

COMPILE_FLAGS="${COMPILE_FLAGS:---compile_model --compile_mode transformer_only}"

SDPA_FLAGS=""
if [[ "${USE_SDPA_ATTENTION}" == "true" ]]; then
    SDPA_FLAGS="--use_sdpa_attention"
fi

# Keep SIGReg disabled for this profiling run. The script lives on the SIGReg
# branch to match the current codebase, but does not pass any SIGReg flags, so
# train.py uses its default sigreg_lambda=0.0 path.
SIGREG_FLAGS=""
SIGREG_STATUS="disabled (no SIGReg CLI flags; train.py default sigreg_lambda=0.0)"

if [[ -z "${WANDB_FLAGS+x}" ]]; then
    WANDB_FLAGS="--disable_wandb"
fi

TIMESTAMP=$(date +"%m%d_%H%M")
DATE_DIR=$(date +"%Y%m%d")
CONFIG_TAG="${MODEL_SIZE}_profile_ctx${CONTEXT_LENGTH}_bs$((NUM_GPUS * DEVICE_BATCH_SIZE * GRAD_ACCUM))_lr${PEAK_LR}_${NUM_NODES}node_${GPUS_PER_NODE}gpus"
export RUN_NAME="${RUN_PREFIX}_${TIMESTAMP}_${CONFIG_TAG}"

LOG_DIR="${NOVA_HOME}/logs_base_train/${DATE_DIR}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${RUN_NAME}_rank${NODE_RANK}.log"
GPU_MON_INTERVAL=${GPU_MON_INTERVAL:-5}
GPU_MON_LOG="${LOG_DIR}/${RUN_NAME}_rank${NODE_RANK}_nvidia_smi_dmon.log"
ANALYZE_ON_EXIT=${ANALYZE_ON_EXIT:-1}

echo "=== Profiling Run ==="
echo "RUN_NAME: ${RUN_NAME}"
echo "LOG_FILE: ${LOG_FILE}"
echo "GPU_MON_LOG: ${GPU_MON_LOG}"
echo "MAX_STEPS: ${MAX_STEPS}"
echo "MAX_SCHEDULING_STEPS: ${MAX_SCHEDULING_STEPS}"
echo "VAL_CHECK_INTERVAL: ${VAL_CHECK_INTERVAL}"
echo "LIMIT_VAL_BATCHES: ${LIMIT_VAL_BATCHES}"
echo "SAVE_PERIODIC_STEPS: ${SAVE_PERIODIC_STEPS}"
echo "ANALYZE_ON_EXIT: ${ANALYZE_ON_EXIT}"
echo "SIGREG_STATUS: ${SIGREG_STATUS}"
echo "PROFILE LOG KEYS:"
echo "  [profile_step]   optimizer-step summary: loss / tok_s / step_ms / micro-batches"
echo "  [profile_timing] wait / forward / backward / optimizer timing"
echo "  [profile_data]   dataloader fetch / tokenize / packing / copy timing"
echo "  [profile_state]  parquet progress / doc_buffer / packed-cropped stats"

exec > >(tee -a "${LOG_FILE}") 2>&1

GPU_MON_PID=""
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi dmon -s pucm -d "${GPU_MON_INTERVAL}" -o DT > "${GPU_MON_LOG}" 2>&1 &
    GPU_MON_PID=$!
    echo "Started nvidia-smi dmon: pid=${GPU_MON_PID}, interval=${GPU_MON_INTERVAL}s"
else
    echo "nvidia-smi not found; GPU utilization monitor is disabled"
fi

cleanup_gpu_monitor() {
    if [[ -n "${GPU_MON_PID}" ]]; then
        kill "${GPU_MON_PID}" >/dev/null 2>&1 || true
        wait "${GPU_MON_PID}" >/dev/null 2>&1 || true
        GPU_MON_PID=""
    fi
}

summarize_profile_log() {
    if [[ "${ANALYZE_ON_EXIT}" != "1" ]]; then
        return
    fi
    if [[ "${NODE_RANK}" != "0" ]]; then
        return
    fi
    if [[ ! -f "${SUMMARY_SCRIPT}" ]]; then
        echo "Summary script not found: ${SUMMARY_SCRIPT}"
        return
    fi
    if [[ ! -s "${LOG_FILE}" ]]; then
        echo "Profile log is empty or missing: ${LOG_FILE}"
        return
    fi
    python "${SUMMARY_SCRIPT}" --log "${LOG_FILE}" --label "${RUN_NAME}" || true
}

finish_run() {
    local exit_code=$?
    trap - EXIT INT TERM
    cleanup_gpu_monitor
    echo ""
    if [[ $exit_code -eq 0 ]]; then
        echo "Training finished successfully (rank ${NODE_RANK})"
    elif [[ $exit_code -eq 130 || $exit_code -eq 143 ]]; then
        echo "Training interrupted intentionally (exit code: ${exit_code}, rank: ${NODE_RANK}); summarizing partial profile log"
    else
        echo "Training exited abnormally (exit code: ${exit_code}, rank: ${NODE_RANK})"
    fi
    echo "Log file: ${LOG_FILE}"
    echo "GPU monitor log: ${GPU_MON_LOG}"
    summarize_profile_log
    exit "${exit_code}"
}

trap finish_run EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

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
  --truncate_mcmc \
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
  --log_every_n_steps 1 \
  --set_matmul_precision "medium" \
  --float_precision "bf16-mixed" \
  --manual_gc_collect_every_n_steps -1 \
  --save_top_k_ckpts "${SAVE_TOP_K}" \
  --save_periodic_steps "${SAVE_PERIODIC_STEPS}" \
  ${WANDB_FLAGS} \
  ${OPTION_FLAGS} \
  ${COMPILE_FLAGS} \
  ${SDPA_FLAGS} \
  ${SIGREG_FLAGS}

TRAIN_EXIT_CODE=$?
exit "${TRAIN_EXIT_CODE}"
