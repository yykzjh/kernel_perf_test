from functools import partial
import os
import random
from types import SimpleNamespace

import torch

from kernel_perf_test.layers.attention import TRTLLMMHAAttnBackend
from kernel_perf_test import utils


def parse_environment_variables() -> SimpleNamespace:
    """
    Parse all environment variables required for testing.

    Returns:
        A SimpleNamespace object containing all configuration parameters, supporting attribute access
    """
    # Testing configuration
    torch_cuda_profiler_dir_path = os.getenv("TORCH_CUDA_PROFILER_DIR_PATH", "./")

    # Attention backend configuration
    num_pages = int(os.getenv("NUM_PAGES", "1024000"))
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


def test_main(args: SimpleNamespace):
    # Set random seed
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    # Determine the torch.dtype
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

    # Define test function
    def test_func():
        _ = attn_backend(q, k, v)

    # Benchmark attention backend
    avg_t, min_t, max_t = utils.bench(test_func, num_warmups=50, num_tests=30)
    print(f"avg_t={avg_t * 1e6:.2f} us, min_t={min_t * 1e6:.2f} us, max_t={max_t * 1e6:.2f} us", flush=True)

    # Profile attention backend
    attn_backend_t = utils.bench_kineto(
        test_func,
        kernel_names=("mha",),
        num_warmups=50,
        num_tests=30,
        suppress_kineto_output=True,
        barrier_comm_profiling=False,
        trace_path=os.path.join(args.torch_cuda_profiler_dir_path, "attn_backend_trace.json"),
        num_kernels_per_period=1,
    )
    print(
        f"attn_backend_t={attn_backend_t[0] * 1e6:.2f} us",
        flush=True,
    )


if __name__ == "__main__":
    # Parse environment variables
    args = parse_environment_variables()
    # Execute test
    test_main(args)
