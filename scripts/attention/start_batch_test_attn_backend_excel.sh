#!/bin/bash

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
export PERFORMANCE_FILE_DIR_PATH="/root/performance_results/"

export SEQ_LEN=128  # 单位：K
export BATCH_SIZE=512

nohup python -u kernel_perf_test/tests/attention/batch_test_attn_backend_excel.py > batch_test_attn_backend_excel.log 2>&1 &
