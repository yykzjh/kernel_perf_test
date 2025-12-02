#!/bin/bash

export PYTHONPATH="./"
export FLASHINFER_CUDA_ARCH_LIST="10.0"
export CUDA_VISIBLE_DEVICES="0"

export HIDDEN_SIZE=6144
export NUM_EXPERTS=160
export PERFORMANCE_RESULTS_DIR_PATH="/root/performance_results/"
export TORCH_CUDA_PROFILER_DIR_PATH="/root/torch_cuda_profiler/replicated_linear_moe_gate/"

export ITERATE_MAX_BATCH_SIZE=512

nohup python -u kernel_perf_test/tests/moe/moe_gate/batch_test_replicated_linear_moe_gate.py > batch_test_replicated_linear_moe_gate.log 2>&1 &
