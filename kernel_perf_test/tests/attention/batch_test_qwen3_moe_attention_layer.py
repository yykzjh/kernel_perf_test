import os
import gc
import math
import time
import random
from tqdm import tqdm
import multiprocessing as mp
from types import SimpleNamespace

import torch
from flashinfer import testing

from kernel_perf_test.layers.attention.qwen3_moe_attention_layer import Qwen3MoeAttentionLayer
from kernel_perf_test import utils


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
    # Attention backend configuration
    max_seq_len = int(os.getenv("MAX_SEQ_LEN", "8196"))
    hidden_size = int(os.getenv("HIDDEN_SIZE", "6144"))
    num_q_heads = int(os.getenv("NUM_Q_HEADS", "96"))
    num_kv_heads = int(os.getenv("NUM_KV_HEADS", "8"))
    head_dim = int(os.getenv("HEAD_DIM", "128"))
    page_size = int(os.getenv("PAGE_SIZE", "32"))
    num_pages = int(os.getenv("NUM_PAGES", "0"))
    layer_num = int(os.getenv("LAYER_NUM", "62"))
    iterate_max_seq_len = int(os.getenv("ITERATE_MAX_SEQ_LEN", "128"))
    iterate_max_batch_size = int(os.getenv("ITERATE_MAX_BATCH_SIZE", "512"))

    return SimpleNamespace(
        torch_cuda_profiler_dir_path=torch_cuda_profiler_dir_path,
        performance_results_dir_path=performance_results_dir_path,
        max_seq_len=max_seq_len,
        hidden_size=hidden_size,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        num_pages=num_pages,
        layer_num=layer_num,
        iterate_max_seq_len=iterate_max_seq_len,
        iterate_max_batch_size=iterate_max_batch_size,
    )


def _worker_process(args_dict: dict):
    """Worker entry for subprocess execution."""
    from types import SimpleNamespace as _SNS  # local import for spawned process
    import warnings
    import traceback
    import sys

    # Ignore UserWarning
    warnings.filterwarnings("ignore", category=UserWarning)

    try:
        worker_args = _SNS(**args_dict)
        return test_main(worker_args)
    except Exception as e:
        # 打印完整的错误堆栈，然后返回可序列化的错误信息
        print("=" * 80, file=sys.stderr)
        print("ERROR in worker process:", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        # 返回可序列化的错误信息字符串
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        raise RuntimeError(error_msg)


def run_test_in_subprocess(args: SimpleNamespace):
    """Run test_main inside a spawned subprocess and return its metrics."""
    ctx = mp.get_context("spawn")
    args_dict = vars(args).copy()
    with ctx.Pool(processes=1) as pool:
        result = pool.apply(_worker_process, (args_dict,))
    return result


def test_main(args: SimpleNamespace):
    # Set random seed
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    testing.set_seed(42)
    # Set default device
    torch.set_default_device("cuda")
    # Initialize Qwen3MoeAttentionLayer
    qwen3_moe_attention_layer = Qwen3MoeAttentionLayer(
        seq_len=args.seq_len,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        num_q_heads=args.num_q_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        page_size=args.page_size,
        num_pages=args.num_pages,
        layer_num=args.layer_num,
    )
    # Capture graph
    graph = utils.capture_graph(
        lambda: qwen3_moe_attention_layer(),
        num_warmups=2,
    )

    # Define test function
    def test_func():
        graph.replay()

    # Profile
    qwen3_moe_attention_layer_time = utils.bench_kineto(
        test_func,
        num_warmups=2,
        num_tests=10,
        suppress_kineto_output=False,
        trace_path=(
            os.path.join(
                args.torch_cuda_profiler_dir_path,
                f"qwen3_moe_attention_layer_seq_len-{args.seq_len // 1024}K_batch_size-{args.batch_size}.json",
            )
            if args.torch_cuda_profiler_dir_path is not None
            else None
        ),
        position_shift=(1, 1),
    )

    # Clean up
    del qwen3_moe_attention_layer

    return qwen3_moe_attention_layer_time


if __name__ == "__main__":
    # Parse environment variables
    args = parse_environment_variables()

    # tqdm setting
    pbar = tqdm(
        total=math.log2(args.iterate_max_seq_len) * args.iterate_max_batch_size,
        desc="Testing different seq_lens and batch sizes of Qwen3MoeAttentionLayer",
    )

    # Performance data
    latency_dict = {}
    # Iterate over seq_lens
    exp_base = 1
    latency_dict["batch_size"] = list(range(1, args.iterate_max_batch_size + 1))
    while math.exp2(exp_base) <= args.iterate_max_seq_len:
        args.seq_len = int(math.exp2(exp_base) * 1024)
        latency_dict[f"seq_len: {args.seq_len} / us"] = []
        # Iterate over batch sizes
        for batch_size in range(1, args.iterate_max_batch_size + 1):
            args.batch_size = batch_size
            args.num_pages = 0
            try:
                # Execute test in subprocess
                latency_us = run_test_in_subprocess(args)
                # Save performance data
                latency_dict[f"seq_len: {args.seq_len} / us"].append(latency_us)
                # Synchronize and empty cache
                torch.cuda.synchronize()
                gc.collect()
                time.sleep(0.1)
                torch.cuda.empty_cache()
                pbar.update(1)
            except Exception as e:
                # Fill None values
                latency_dict[f"seq_len: {args.seq_len} / us"] += [None] * (args.iterate_max_batch_size - batch_size + 1)
                # Synchronize and empty cache
                torch.cuda.synchronize()
                gc.collect()
                time.sleep(0.1)
                torch.cuda.empty_cache()
                print(f"Error: {e}", flush=True)
                pbar.update(args.iterate_max_batch_size - batch_size + 1)
                break
        exp_base += 1
    pbar.close()
    print(latency_dict, flush=True)
    # Save performance results to excel file
    utils.save_performance_results_to_excel(
        save_dir_path=args.performance_results_dir_path,
        file_name="qwen3_moe_attention_layer_performance",
        index_key="batch_size",
        latency=latency_dict,
    )
