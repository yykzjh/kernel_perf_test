import os
import gc
import json
import math
import time
import random
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import matplotlib.pyplot as plt
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
    performance_figure_dir_path = os.getenv("PERFORMANCE_FIGURE_DIR_PATH", "./performance_figures")
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
        performance_figure_dir_path=performance_figure_dir_path,
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


def draw_performance_chart(
    args: SimpleNamespace,
    batch_size_list: list,
    latency_list: list,
    tflops_list: list,
    tbytes_list: list,
    tflops_per_sec_list: list,
    tb_per_sec_list: list,
) -> None:
    """
    Draw performance chart
    Args:
        args: Environment variables
        batch_size_list: List of batch sizes
        latency_list: List of latencies
        tflops_list: List of TFLOPS
        tbytes_list: List of TBytes
        tflops_per_sec_list: List of TFLOPS per second
        tb_per_sec_list: List of TB per second
    """
    save_subdir_name = f"page_size={args.page_size}-head_dim={args.head_dim}-max_seq_len={args.max_seq_len}-num_q_heads={args.num_tp_q_heads}-num_kv_heads={args.num_tp_k_heads}-dtype={args.torch_dtype}-seq_len={args.seq_len}"
    save_subdir_path = os.path.join(args.performance_figure_dir_path, save_subdir_name)
    os.makedirs(save_subdir_path, exist_ok=True)

    data_dict = {
        "batch_size": batch_size_list,
        "latency_us": latency_list,
        "tflops": tflops_list,
        "tbytes": tbytes_list,
        "tflops_per_sec": tflops_per_sec_list,
        "tb_per_sec": tb_per_sec_list,
    }
    with open(os.path.join(save_subdir_path, "performance_data.json"), "w", encoding="utf-8") as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=2)

    # Draw performance chart
    def plot_line(y_values, y_label, file_name):
        if not y_values:
            return
        plt.figure()
        plt.plot(
            batch_size_list,
            y_values,
            marker="o",
            markersize=4,
            linewidth=1.2,
        )
        plt.xlabel("Batch Size")
        plt.ylabel(y_label)
        plt.title(f"{y_label} vs Batch Size")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(save_subdir_path, file_name))
        plt.close()

    plot_line(latency_list, "Latency (us)", "latency_vs_batch_size.png")
    plot_line(tflops_list, "TFLOPs", "tflops_vs_batch_size.png")
    plot_line(tbytes_list, "TBytes", "tbytes_vs_batch_size.png")
    plot_line(tflops_per_sec_list, "TFLOPS", "tflops_per_sec_vs_batch_size.png")
    plot_line(tb_per_sec_list, "TBS", "tb_per_sec_vs_batch_size.png")

    ai_values = []
    perf_values = []
    colors = []
    for batch, total_tflops, total_tbytes, perf in zip(batch_size_list, tflops_list, tbytes_list, tflops_per_sec_list):
        if total_tbytes <= 0 or perf <= 0:
            continue
        ai = (total_tflops * 1e12) / (total_tbytes * 1e12)
        ai_values.append(ai)
        perf_values.append(perf)
        colors.append(batch)

    if ai_values and perf_values:
        peak_compute = 2.5e15  # FLOPS
        peak_bandwidth = 8e12  # B/s
        ai_min = max(min(ai_values) * 0.5, 1e-3)
        ai_max = max(ai_values) * 2
        ai_axis = np.logspace(math.log10(ai_min), math.log10(ai_max), num=256)
        mem_bound = (peak_bandwidth * ai_axis) / 1e12
        compute_bound = np.full_like(ai_axis, peak_compute / 1e12)

        plt.figure()
        plt.loglog(
            ai_axis,
            mem_bound,
            "--",
            linewidth=1.2,
            label="Memory Bound (8 TB/s)",
        )
        plt.loglog(
            ai_axis,
            compute_bound,
            "-",
            linewidth=1.2,
            label="Compute Bound (2500 TFLOPs/s)",
        )
        scatter = plt.scatter(
            ai_values,
            perf_values,
            c=colors,
            cmap="viridis",
            s=36,
            marker="o",
            edgecolor="black",
            linewidth=0.6,
            label="Measured Points",
        )
        plt.colorbar(scatter, label="Batch Size")
        plt.xlabel("Arithmetic Intensity (FLOPs / Byte)")
        plt.ylabel("Performance (TFLOPs/s)")
        plt.title("Roofline Analysis")
        plt.grid(True, which="both", linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_subdir_path, "roofline.png"))
        plt.close()


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

    pbar = tqdm(total=math.log2(args.seq_len), desc="Testing seq_lens")
    # Iterate over seq_lens
    exp_base = 1
    iterate_max_seq_len = args.seq_len
    iterate_max_batch_size = args.batch_size
    args.seq_len = 0
    args.batch_size = 0
    while math.exp2(exp_base) <= iterate_max_seq_len:
        args.seq_len = int(math.exp2(exp_base) * 1024)
        # Current seq_len performance data
        batch_size_list = []
        latency_list = []
        tflops_list = []
        tbytes_list = []
        tflops_per_sec_list = []
        tb_per_sec_list = []
        # Iterate over batch sizes
        for batch_size in range(1, iterate_max_batch_size + 1):
            args.batch_size = batch_size
            args.num_pages = 0
            try:
                # Execute test
                latency, tflops, tbytes, tflops_per_sec, tb_per_sec = run_test_in_subprocess(args)
                # Synchronize and empty cache
                torch.cuda.synchronize()
                gc.collect()
                time.sleep(0.1)
                torch.cuda.empty_cache()
                # Save performance data
                batch_size_list.append(batch_size)
                latency_list.append(latency)
                tflops_list.append(tflops)
                tbytes_list.append(tbytes)
                tflops_per_sec_list.append(tflops_per_sec)
                tb_per_sec_list.append(tb_per_sec)
            except Exception as e:
                # Synchronize and empty cache
                torch.cuda.synchronize()
                gc.collect()
                time.sleep(0.1)
                torch.cuda.empty_cache()
                print(f"Error: {e}", flush=True)
                break
        # Draw performance chart
        draw_performance_chart(
            args,
            batch_size_list,
            latency_list,
            tflops_list,
            tbytes_list,
            tflops_per_sec_list,
            tb_per_sec_list,
        )
        exp_base += 1
        pbar.update(1)
    pbar.close()
