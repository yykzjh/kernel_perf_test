#!/bin/bash

export PYTHONPATH="./"
export FLASHINFER_CUDA_ARCH_LIST="10.0"
export CUDA_VISIBLE_DEVICES="0"

export NUM_PAGES=0
export PAGE_SIZE=32
export HEAD_DIM=128
export BATCH_SIZE=256
export SEQ_LEN=2000
export MAX_SEQ_LEN=262144
export NUM_TP_Q_HEADS=96
export NUM_TP_K_HEADS=8
export NUM_TP_V_HEADS=8
export SLIDING_WINDOW_SIZE=-1
export TORCH_DTYPE="bf16"
export TORCH_CUDA_PROFILER_DIR_PATH="./trace_files/"

python kernel_perf_test/tests/single_test_attn_backend.py
