#!/bin/bash

export PYTHONPATH="./"
export FLASHINFER_CUDA_ARCH_LIST="10.0"
export CUDA_VISIBLE_DEVICES="0"

export NUM_PAGES=1024000
export PAGE_SIZE=32
export HEAD_DIM=128
export BATCH_SIZE=32
export SEQ_LEN=2048
export MAX_SEQ_LEN=8196
export NUM_TP_Q_HEADS=12
export NUM_TP_K_HEADS=1
export NUM_TP_V_HEADS=1
export SLIDING_WINDOW_SIZE=-1
export TORCH_DTYPE="bf16"
export TORCH_CUDA_PROFILER_DIR_PATH="./trace_files/"

python kernel_perf_test/tests/test_attn_backend.py
