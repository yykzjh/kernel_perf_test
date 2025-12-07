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
from kernel_perf_test.layers.norm.fused_add_rmsnorm import FusedAddRMSNorm


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
    hidden_size = int(os.getenv("HIDDEN_SIZE", "6144"))
    iterate_max_batch_size = int(os.getenv("ITERATE_MAX_BATCH_SIZE", "512"))

    return SimpleNamespace(
        performance_results_dir_path=performance_results_dir_path,
        torch_cuda_profiler_dir_path=torch_cuda_profiler_dir_path,
        hidden_size=hidden_size,
        iterate_max_batch_size=iterate_max_batch_size,
    )


def test_main(args: SimpleNamespace):
    # Set random seed
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.set_default_device("cuda")
    # Initialize test module
    fused_add_rmsnorm = FusedAddRMSNorm(
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
    )

    graph = utils.capture_graph(lambda: fused_add_rmsnorm(), num_warmups=50)

    # Define test function
    def test_func():
        graph.replay()

    # Profile
    fused_add_rmsnorm_time = utils.bench_kineto(
        test_func,
        num_warmups=50,
        num_tests=30,
        suppress_kineto_output=False,
        trace_path=(
            os.path.join(
                args.torch_cuda_profiler_dir_path,
                f"fused_add_rmsnorm_trace_batch_size-{args.batch_size}.json",
            )
            if args.torch_cuda_profiler_dir_path is not None
            else None
        ),
        position_shift=(1, 1),
    )

    # Clean up
    del fused_add_rmsnorm

    return fused_add_rmsnorm_time


if __name__ == "__main__":
    # Parse environment variables
    args = parse_environment_variables()

    # Performance data
    latency_dict = {}
    # Initialize batch_size_list
    latency_dict["batch_size"] = list(range(1, args.iterate_max_batch_size + 1))
    latency_dict["latency / us"] = []
    # Iterate over batch sizes
    for batch_size in tqdm(range(1, args.iterate_max_batch_size + 1), desc="Testing batch sizes"):
        args.batch_size = batch_size
        try:
            # Execute test in subprocess
            latency_us = test_main(args)
            # Synchronize and empty cache
            torch.cuda.synchronize()
            gc.collect()
            time.sleep(0.1)
            torch.cuda.empty_cache()
            # Save performance data
            latency_dict["latency / us"].append(latency_us)
        except Exception as e:
            # Fill None values
            latency_dict["latency / us"] += [None] * (args.iterate_max_batch_size - batch_size + 1)
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
        file_name="fused_add_rmsnorm_performance",
        index_key="batch_size",
        latency=latency_dict,
    )
