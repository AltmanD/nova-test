#!/bin/bash

################################################################################
# EBT d26 profiling rjob submit script - 2 nodes x 8 GPUs
#
# Usage:
#   bash rjob_ebt_2node_8gpu_profile.sh
################################################################################

RUN_SCRIPT="/mnt/shared-storage-user/luyudong/nova/openebm/elm/runs/rjob/run_ebt_2node_8gpu_profile.sh"

echo "=== Submit profiling run ==="
CMD="bash -exc \"${RUN_SCRIPT}\""
JOB_NAME="ebt-d26-2node-8gpu-profile"

rjob submit \
  --name="${JOB_NAME}" \
  --gpu=8 \
  --memory=1000000 \
  --cpu=100 \
  --charged-group=narmodel_gpu \
  --private-machine=group \
  -P 2 \
  --image=registry.h.pjlab.org.cn/ailab-rlinfra-rlinfra_gpu/easyr1:lightrft-20260119 \
  --mount=gpfs://gpfs1/puyuan:/mnt/shared-storage-user/puyuan \
  --mount=gpfs://gpfs1/luyudong:/mnt/shared-storage-user/luyudong \
  -e DISTRIBUTED_JOB=true \
  --custom-resources brainpp.cn/fuse=1 \
  --custom-resources rdma/mlnx_shared=8 \
  --custom-resources mellanox.com/mlnx_rdma=1 \
  -- ${CMD}
