from functools import partial
import os
import gc
import time
import random
from tqdm import tqdm
from types import SimpleNamespace
import warnings

# Ignore UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

import torch

from kernel_perf_test import utils
from kernel_perf_test.layers.moe.moe_ffn.deepgemm_masked_moe_ffn import DeepGemmMaskedMoEFfn


def parse_environment_variables() -> SimpleNamespace:
    """
    Parse all environment variables required for testing.

    Returns:
        A SimpleNamespace object containing all configuration parameters, supporting attribute access
    """
    # Testing configuration
    performance_results_dir_path = os.getenv("PERFORMANCE_RESULTS_DIR_PATH", None)
    if performance_results_dir_path is not None and not os.path.exists(performance_results_dir_path):
        os.makedirs(performance_results_dir_path, exist_ok=True)
    torch_cuda_profiler_dir_path = os.getenv("TORCH_CUDA_PROFILER_DIR_PATH", None)
    if torch_cuda_profiler_dir_path is not None and not os.path.exists(torch_cuda_profiler_dir_path):
        os.makedirs(torch_cuda_profiler_dir_path, exist_ok=True)
    # Module configuration
    iterate_max_num_local_experts = int(os.getenv("ITERATE_MAX_NUM_LOCAL_EXPERTS", "192"))
    iterate_max_expected_m = int(os.getenv("ITERATE_MAX_EXPECTED_M", "512"))
    hidden_size = int(os.getenv("HIDDEN_SIZE", "6144"))
    moe_intermediate_size = int(os.getenv("MOE_INTERMEDIATE_SIZE", "2560"))

    return SimpleNamespace(
        performance_results_dir_path=performance_results_dir_path,
        torch_cuda_profiler_dir_path=torch_cuda_profiler_dir_path,
        iterate_max_num_local_experts=iterate_max_num_local_experts,
        iterate_max_expected_m=iterate_max_expected_m,
        hidden_size=hidden_size,
        moe_intermediate_size=moe_intermediate_size,
    )


def test_main(args: SimpleNamespace):
    # Set random seed
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    # Initialize DeepGemmMaskedMoEFfn
    deepgemm_masked_moe_ffn = DeepGemmMaskedMoEFfn(
        E=args.num_local_experts,
        N=args.moe_intermediate_size * 2,
        K=args.hidden_size,
    )
    # Initialize masked_m tensor
    masked_m = torch.empty((args.num_local_experts,), device="cuda", dtype=torch.int)
    for i in range(args.num_local_experts):
        masked_m[i] = int(args.expected_m * random.uniform(0.7, 1.3))
    max_m = (masked_m.max().item() + 127) // 128 * 128
    # Initialize hidden_states_fp8 and hidden_states_scale tensors
    hidden_states_fp8 = torch.randn(
        (args.num_local_experts, max_m, args.hidden_size),
        dtype=torch.float32,
        device="cuda",
    ).to(torch.float8_e4m3fn)
    hidden_states_scale = (
        torch.randn(
            (args.num_local_experts, args.hidden_size // 512, max_m),
            dtype=torch.float32,
            device="cuda",
        )
        .to(torch.int32)
        .permute(0, 2, 1)
    )

    graph = utils.capture_graph(
        lambda: deepgemm_masked_moe_ffn(
            hidden_states_fp8=hidden_states_fp8,
            hidden_states_scale=hidden_states_scale,
            masked_m=masked_m,
            expected_m=args.expected_m,
        ),
        num_warmups=50,
    )

    # Define test function
    def test_func():
        graph.replay()

    # Profile DeepGemmMaskedMoEFfn
    deepgemm_masked_moe_ffn_time = utils.bench_kineto(
        test_func,
        num_warmups=50,
        num_tests=30,
        suppress_kineto_output=False,
        trace_path=(
            os.path.join(
                args.torch_cuda_profiler_dir_path,
                f"deepgemm_masked_moe_ffn_trace_num_local_experts-{args.num_local_experts}_expected_m-{args.expected_m}.json",
            )
            if args.torch_cuda_profiler_dir_path is not None
            else None
        ),
    )

    # Clean up
    del deepgemm_masked_moe_ffn
    del masked_m
    del hidden_states_fp8
    del hidden_states_scale

    return deepgemm_masked_moe_ffn_time


if __name__ == "__main__":
    # Parse environment variables
    args = parse_environment_variables()

    # Performance data
    latency_dict = {}
    # init expected_m_list
    latency_dict["expected_m"] = list(range(1, args.iterate_max_expected_m + 1))
    # Iterate over num_local_experts
    for num_local_experts in tqdm(range(1, args.iterate_max_num_local_experts + 1), desc="Testing num_local_experts"):
        args.num_local_experts = num_local_experts
        # init current num_local_experts_list
        latency_dict[f"num_local_experts: {args.num_local_experts}"] = []
        # Iterate over batch sizes
        for expected_m in range(1, args.iterate_max_expected_m + 1):
            args.expected_m = expected_m
            try:
                # Execute test in subprocess
                latency = test_main(args)
                # Synchronize and empty cache
                torch.cuda.synchronize()
                gc.collect()
                time.sleep(0.1)
                torch.cuda.empty_cache()
                # Save performance data
                latency_dict[f"num_local_experts: {args.num_local_experts}"].append(latency)
            except Exception as e:
                # Fill None values
                latency_dict[f"num_local_experts: {args.num_local_experts}"] += [None] * (
                    args.iterate_max_expected_m - expected_m + 1
                )
                # Synchronize and empty cache
                torch.cuda.synchronize()
                gc.collect()
                time.sleep(0.1)
                torch.cuda.empty_cache()
                print(f"Error: {e}", flush=True)
                break
    print(latency_dict, flush=True)
    # Save performance results to excel file
    utils.save_performance_results_to_excel(
        save_dir_path=args.performance_results_dir_path,
        file_name="deepgemm_masked_moe_ffn_performance",
        index_key="expected_m",
        latency=latency_dict,
    )
