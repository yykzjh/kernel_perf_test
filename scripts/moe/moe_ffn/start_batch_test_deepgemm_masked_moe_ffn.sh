#!/bin/bash

export PYTHONPATH="./"
export FLASHINFER_CUDA_ARCH_LIST="10.0"
export CUDA_VISIBLE_DEVICES="0"

export HIDDEN_SIZE=6144
export MOE_INTERMEDIATE_SIZE=2560
export PERFORMANCE_RESULTS_DIR_PATH="/root/performance_results/"
export TORCH_CUDA_PROFILER_DIR_PATH="/root/torch_cuda_profiler/"

export ITERATE_MAX_NUM_LOCAL_EXPERTS=192
export ITERATE_MAX_EXPECTED_M=512

nohup python -u kernel_perf_test/tests/moe/moe_ffn/batch_test_deepgemm_masked_moe_ffn.py > batch_test_deepgemm_masked_moe_ffn.log 2>&1 &
