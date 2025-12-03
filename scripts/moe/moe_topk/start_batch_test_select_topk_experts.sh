#!/bin/bash

export PYTHONPATH="./"
export FLASHINFER_CUDA_ARCH_LIST="10.0"
export CUDA_VISIBLE_DEVICES="1"

export HIDDEN_SIZE=6144
export NUM_EXPERTS=160
export TOPK=8
export PERFORMANCE_RESULTS_DIR_PATH="/root/performance_results/"
export TORCH_CUDA_PROFILER_DIR_PATH="/root/torch_cuda_profiler/select_topk_experts/"

export ITERATE_MAX_BATCH_SIZE=512

nohup python -u kernel_perf_test/tests/moe/moe_topk/batch_test_select_topk_experts.py > batch_test_select_topk_experts.log 2>&1 &
