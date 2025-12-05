#!/bin/bash

export PYTHONPATH="./"
export FLASHINFER_CUDA_ARCH_LIST="10.0"
export CUDA_VISIBLE_DEVICES="0"


export MAX_SEQ_LEN=262144
export HIDDEN_SIZE=6144
export NUM_Q_HEADS=96
export NUM_KV_HEADS=8
export HEAD_DIM=128
export PAGE_SIZE=32
export NUM_PAGES=0
export LAYER_NUM=62
export PERFORMANCE_RESULTS_DIR_PATH="/root/performance_results/"
export TORCH_CUDA_PROFILER_DIR_PATH="/root/torch_cuda_profiler/qwen3_moe_attention_layer/"

export ITERATE_MAX_SEQ_LEN=128  # 单位：K
export ITERATE_MAX_BATCH_SIZE=512

nohup python -u kernel_perf_test/tests/attention/batch_test_qwen3_moe_attention_layer.py > batch_test_qwen3_moe_attention_layer.log 2>&1 &
