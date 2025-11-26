import os
import gc
import math
import time
import random
import pandas as pd
from tqdm import tqdm
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

    # Benchmark attention backend
    _, _, _ = utils.bench(test_func, num_warmups=0, num_tests=10)

    # Clean up
    del attn_backend
    del q
    del k
    del v


if __name__ == "__main__":
    # Parse environment variables
    args = parse_environment_variables()

    try:
        # Execute test
        test_main(args)
    except Exception:
        exit(1)

    exit(0)
