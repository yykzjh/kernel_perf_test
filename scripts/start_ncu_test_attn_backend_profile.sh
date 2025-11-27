#!/bin/bash

set -euo pipefail

export PYTHONPATH="./"
export FLASHINFER_CUDA_ARCH_LIST="10.0"
export CUDA_VISIBLE_DEVICES="0"

export NUM_PAGES=0
export PAGE_SIZE=32
export HEAD_DIM=128
export MAX_SEQ_LEN=262144
export NUM_TP_Q_HEADS=96
export NUM_TP_K_HEADS=8
export NUM_TP_V_HEADS=8
export SLIDING_WINDOW_SIZE=-1
export TORCH_DTYPE="bf16"
export NCU_PROFILE_DIR_PATH="/root/ncu_profiles"

export ITERATE_MAX_SEQ_LEN=128  # 单位：K
export ITERATE_MAX_BATCH_SIZE=512

# Create profile directory
if [ ! -d "${NCU_PROFILE_DIR_PATH}" ]; then
  mkdir -p "${NCU_PROFILE_DIR_PATH}"
fi
sub_dir_name=$(printf "page_size-%d-head_dim-%d-max_seq_len-%d-num_q_heads-%d-num_kv_heads-%d-dtype-%s" "${PAGE_SIZE}" "${HEAD_DIM}" "${MAX_SEQ_LEN}" "${NUM_TP_Q_HEADS}" "${NUM_TP_K_HEADS}" "${TORCH_DTYPE}")
sub_dir_path="${NCU_PROFILE_DIR_PATH}/${sub_dir_name}"
if [ ! -d "${sub_dir_path}" ]; then
  mkdir -p "${sub_dir_path}"
fi
# Iterate over seq_lens
seq_len_k=2
while [ "${seq_len_k}" -le "${ITERATE_MAX_SEQ_LEN}" ]; do
  export SEQ_LEN=$((seq_len_k * 1024))
  batch_size=1
  while [ "${batch_size}" -le "${ITERATE_MAX_BATCH_SIZE}" ]; do
    export BATCH_SIZE="${batch_size}"
    profile_file_path=$(printf "%s/seq_len-%d_batch_size-%03d.ncu-rep" "${sub_dir_path}" "${SEQ_LEN}" "${BATCH_SIZE}")
    # Try to run test
    python kernel_perf_test/tests/ncu_test_attn_backend_profile.py
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        # Run ncu profile
        ncu --set full --target-processes all --export "${profile_file_path}" --kernel-name regex:".*mha.*" python kernel_perf_test/tests/ncu_test_attn_backend_profile.py
    else
        break
    fi
    batch_size=$((batch_size + 1))
  done
  seq_len_k=$((seq_len_k * 2))
done


