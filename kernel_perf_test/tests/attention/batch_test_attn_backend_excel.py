import os
import gc
import math
import time
import random
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from types import SimpleNamespace

import torch
from flashinfer import testing

from kernel_perf_test.layers.attention import TRTLLMMHAAttnBackend
from kernel_perf_test import utils


def parse_environment_variables() -> SimpleNamespace:
    """
    Parse all environment variables required for testing.

    Returns:
        A SimpleNamespace object containing all configuration parameters, supporting attribute access
    """
    # Testing configuration
    torch_cuda_profiler_dir_path = os.getenv("TORCH_CUDA_PROFILER_DIR_PATH", None)
    if torch_cuda_profiler_dir_path is not None and not os.path.exists(torch_cuda_profiler_dir_path):
        os.makedirs(torch_cuda_profiler_dir_path, exist_ok=True)
    performance_file_dir_path = os.getenv("PERFORMANCE_FILE_DIR_PATH", None)
    if performance_file_dir_path is not None and not os.path.exists(performance_file_dir_path):
        os.makedirs(performance_file_dir_path, exist_ok=True)
    # Attention backend configuration
    num_pages = int(os.getenv("NUM_PAGES", "0"))
    page_size = int(os.getenv("PAGE_SIZE", "1"))
    head_dim = int(os.getenv("HEAD_DIM", "128"))
    batch_size = int(os.getenv("BATCH_SIZE", "32"))
    seq_len = int(os.getenv("SEQ_LEN", "4096"))
    max_seq_len = int(os.getenv("MAX_SEQ_LEN", "8196"))
    num_tp_q_heads = int(os.getenv("NUM_TP_Q_HEADS", "12"))
    num_tp_k_heads = int(os.getenv("NUM_TP_K_HEADS", "1"))
    num_tp_v_heads = int(os.getenv("NUM_TP_V_HEADS", "1"))
    sliding_window_size = int(os.getenv("SLIDING_WINDOW_SIZE", "-1"))
    torch_dtype = os.getenv("TORCH_DTYPE", "fp8")

    return SimpleNamespace(
        torch_cuda_profiler_dir_path=torch_cuda_profiler_dir_path,
        performance_file_dir_path=performance_file_dir_path,
        num_pages=num_pages,
        page_size=page_size,
        head_dim=head_dim,
        batch_size=batch_size,
        seq_len=seq_len,
        max_seq_len=max_seq_len,
        num_tp_q_heads=num_tp_q_heads,
        num_tp_k_heads=num_tp_k_heads,
        num_tp_v_heads=num_tp_v_heads,
        sliding_window_size=sliding_window_size,
        torch_dtype=torch_dtype,
    )


def _worker_process(args_dict: dict):
    """Worker entry for subprocess execution."""
    from types import SimpleNamespace as _SNS  # local import for spawned process
    import warnings

    # Ignore UserWarning
    warnings.filterwarnings("ignore", category=UserWarning)

    worker_args = _SNS(**args_dict)
    return test_main(worker_args)


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
    # Determine the torch.dtype
    if isinstance(args.torch_dtype, str):
        if args.torch_dtype == "fp8":
            args.torch_dtype = torch.float8_e4m3fn
        elif args.torch_dtype == "fp16":
            args.torch_dtype = torch.float16
        elif args.torch_dtype == "fp32":
            args.torch_dtype = torch.float32
        elif args.torch_dtype == "bf16":
            args.torch_dtype = torch.bfloat16
        else:
            raise ValueError(f"Unsupported torch.dtype: {args.torch_dtype}")
    elif isinstance(args.torch_dtype, torch.dtype):
        pass
    else:
        raise ValueError(f"Unsupported type of torch.dtype: {type(args.torch_dtype)}")
    # Check num_pages
    if args.num_pages < args.batch_size * math.ceil(args.seq_len / args.page_size):
        args.num_pages = args.batch_size * math.ceil(args.seq_len / args.page_size)
    # Create attention backend
    attn_backend = TRTLLMMHAAttnBackend(
        num_pages=args.num_pages,
        page_size=args.page_size,
        head_dim=args.head_dim,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_seq_len=args.max_seq_len,
        num_tp_q_heads=args.num_tp_q_heads,
        num_tp_k_heads=args.num_tp_k_heads,
        num_tp_v_heads=args.num_tp_v_heads,
        sliding_window_size=args.sliding_window_size,
        torch_dtype=args.torch_dtype,
    )
    # Initialize q, k, v tensors
    q = torch.randn(
        (args.batch_size, args.num_tp_q_heads, args.head_dim),
        dtype=torch.float32,
        device=torch.device("cuda"),
    ).to(args.torch_dtype)
    k = torch.randn(
        (args.num_pages, args.num_tp_k_heads, args.page_size, args.head_dim),
        dtype=torch.float32,
        device=torch.device("cuda"),
    ).to(args.torch_dtype)
    v = torch.randn(
        (args.num_pages, args.num_tp_v_heads, args.page_size, args.head_dim),
        dtype=torch.float32,
        device=torch.device("cuda"),
    ).to(args.torch_dtype)
    torch.cuda.synchronize()

    # Define test function
    def test_func():
        with torch.no_grad():
            _ = attn_backend(q, k, v)

    # Flashinfer benchmark
    measured_times = testing.bench_gpu_time(
        test_func,
        dry_run_iters=50,
        repeat_iters=30,
        l2_flush=True,
        l2_flush_size_mb=256,
        l2_flush_device="cuda",
        sleep_after_run=False,
        enable_cupti=False,
        use_cuda_graph=False,
    )
    measured_times = testing.bench_gpu_time(
        test_func,
        dry_run_iters=50,
        repeat_iters=30,
        l2_flush=True,
        l2_flush_size_mb=256,
        l2_flush_device="cuda",
        sleep_after_run=False,
        enable_cupti=False,
        use_cuda_graph=True,
        num_iters_within_graph=10,
    )

    # Profile attention backend
    attn_backend_time = utils.bench_kineto(
        test_func,
        num_warmups=50,
        num_tests=30,
        suppress_kineto_output=False,
        trace_path=(
            os.path.join(args.torch_cuda_profiler_dir_path, "attn_backend_trace.json")
            if args.torch_cuda_profiler_dir_path is not None
            else None
        ),
    )

    # Calculate attention TFLOPS per second
    FLOPs = testing.attention_flops(
        batch_size=args.batch_size,
        qo_seqlen=1,
        kv_seqlen=args.seq_len,
        head_dim_qk=args.head_dim,
        head_dim_vo=args.head_dim,
        num_qo_heads=args.num_tp_q_heads,
        causal=False,
    )
    tflops_per_sec = testing.attention_tflops_per_sec(
        batch_size=args.batch_size,
        qo_seqlen=1,
        kv_seqlen=args.seq_len,
        head_dim_qk=args.head_dim,
        head_dim_vo=args.head_dim,
        num_qo_heads=args.num_tp_q_heads,
        causal=False,
        time=attn_backend_time / 1e3,
    )

    # Calculate attention TB per second
    tb_per_sec = testing.attention_tb_per_sec(
        batch_size=args.batch_size,
        qo_seqlen=1,
        kv_seqlen=args.seq_len,
        head_dim_qk=args.head_dim,
        head_dim_vo=args.head_dim,
        num_qo_heads=args.num_tp_q_heads,
        num_kv_heads=args.num_tp_k_heads,
        time=attn_backend_time / 1e3,
        q_dtype=args.torch_dtype,
        kv_dtype=args.torch_dtype,
        o_dtype=args.torch_dtype,
    )
    TBytes = tb_per_sec * attn_backend_time / 1e6

    # Clean up
    del attn_backend
    del q
    del k
    del v

    return attn_backend_time, FLOPs / 1e12, TBytes, tflops_per_sec, tb_per_sec


if __name__ == "__main__":
    # Parse environment variables
    args = parse_environment_variables()

    # Performance data
    latency_dict = {}
    tflops_dict = {}
    tbytes_dict = {}
    tflops_per_sec_dict = {}
    tb_per_sec_dict = {}
    pbar = tqdm(total=math.log2(args.seq_len), desc="Testing seq_lens")
    # Iterate over seq_lens
    exp_base = 1
    iterate_max_seq_len = args.seq_len
    iterate_max_batch_size = args.batch_size
    args.seq_len = 0
    args.batch_size = 0
    while math.exp2(exp_base) <= iterate_max_seq_len:
        args.seq_len = int(math.exp2(exp_base) * 1024)
        # init batch_size_list
        latency_dict["Batch Size"] = list(range(1, iterate_max_batch_size + 1))
        tflops_dict["Batch Size"] = list(range(1, iterate_max_batch_size + 1))
        tbytes_dict["Batch Size"] = list(range(1, iterate_max_batch_size + 1))
        tflops_per_sec_dict["Batch Size"] = list(range(1, iterate_max_batch_size + 1))
        tb_per_sec_dict["Batch Size"] = list(range(1, iterate_max_batch_size + 1))
        # init current seq_len_list
        latency_dict[f"Seq Len {args.seq_len}"] = []
        tflops_dict[f"Seq Len {args.seq_len}"] = []
        tbytes_dict[f"Seq Len {args.seq_len}"] = []
        tflops_per_sec_dict[f"Seq Len {args.seq_len}"] = []
        tb_per_sec_dict[f"Seq Len {args.seq_len}"] = []
        # Iterate over batch sizes
        for batch_size in range(1, iterate_max_batch_size + 1):
            args.batch_size = batch_size
            args.num_pages = 0
            try:
                # Execute test in subprocess
                latency, tflops, tbytes, tflops_per_sec, tb_per_sec = run_test_in_subprocess(args)
                # Synchronize and empty cache
                torch.cuda.synchronize()
                gc.collect()
                time.sleep(0.1)
                torch.cuda.empty_cache()
                # Save performance data
                latency_dict[f"Seq Len {args.seq_len}"].append(latency)
                tflops_dict[f"Seq Len {args.seq_len}"].append(tflops)
                tbytes_dict[f"Seq Len {args.seq_len}"].append(tbytes)
                tflops_per_sec_dict[f"Seq Len {args.seq_len}"].append(tflops_per_sec)
                tb_per_sec_dict[f"Seq Len {args.seq_len}"].append(tb_per_sec)
            except Exception as e:
                # Fill None values
                latency_dict[f"Seq Len {args.seq_len}"] += [None] * (iterate_max_batch_size - batch_size + 1)
                tflops_dict[f"Seq Len {args.seq_len}"] += [None] * (iterate_max_batch_size - batch_size + 1)
                tbytes_dict[f"Seq Len {args.seq_len}"] += [None] * (iterate_max_batch_size - batch_size + 1)
                tflops_per_sec_dict[f"Seq Len {args.seq_len}"] += [None] * (iterate_max_batch_size - batch_size + 1)
                tb_per_sec_dict[f"Seq Len {args.seq_len}"] += [None] * (iterate_max_batch_size - batch_size + 1)
                # Synchronize and empty cache
                torch.cuda.synchronize()
                gc.collect()
                time.sleep(0.1)
                torch.cuda.empty_cache()
                print(f"Error: {e}", flush=True)
                break
        exp_base += 1
        pbar.update(1)
    pbar.close()

    print(latency_dict, flush=True)
    print(tflops_dict, flush=True)
    print(tbytes_dict, flush=True)
    print(tflops_per_sec_dict, flush=True)
    print(tb_per_sec_dict, flush=True)
    # Save performance data
    utils.save_performance_results_to_excel(
        save_dir_path=args.performance_file_dir_path,
        file_name="attn_backend_performance",
        index_key="Batch Size",
        latency=latency_dict,
        tflops=tflops_dict,
        tbytes=tbytes_dict,
        tflops_per_sec=tflops_per_sec_dict,
        tb_per_sec=tb_per_sec_dict,
    )
