import math
from typing import Optional

import torch
import torch.nn as nn
import flashinfer


class TRTLLMMHAAttnBackend(nn.Module):
    """TRTLLM MHA Attention Backend"""

    def __init__(
        self,
        num_pages: int,
        page_size: int,
        head_dim: int,
        batch_size: int,
        seq_len: int,
        max_seq_len: int,
        num_tp_q_heads: int,
        num_tp_k_heads: int,
        num_tp_v_heads: int,
        sliding_window_size: int,
        torch_dtype: torch.dtype,
    ):
        """
        Initialize the TRTLLM MHA Attention Backend
        Args:
            num_pages (int): Number of pages
            page_size (int): Number of tokens per page
            head_dim (int): Number of dimensions per head
            batch_size (int): Current batch size
            seq_len (int): Number of tokens per prompt in the current batch
            max_seq_len (int): Maximum number of tokens per prompt in the batch
            num_tp_q_heads (int): Number of token parallel query heads
            num_tp_k_heads (int): Number of token parallel key heads
            num_tp_v_heads (int): Number of token parallel value heads
            sliding_window_size (int): Sliding window size
            torch_dtype (torch.dtype): PyTorch dtype
        """
        super().__init__()
        # Get device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize the parameters
        self.num_pages = num_pages
        self.page_size = page_size
        self.head_dim = head_dim
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.max_seq_len = max_seq_len
        self.num_tp_q_heads = num_tp_q_heads
        self.num_tp_k_heads = num_tp_k_heads
        self.num_tp_v_heads = num_tp_v_heads
        self.sliding_window_size = sliding_window_size
        self.torch_dtype = torch_dtype
        # Check parameters
        assert self.seq_len <= self.max_seq_len, "Sequence length must be less than or equal to maximum sequence length"
        assert self.num_pages >= self.batch_size * math.ceil(
            self.seq_len / self.page_size
        ), "Number of pages must be greater than or equal to the product of batch size and sequence length"
        # Generate query tensor
        self.q = torch.randn(
            (self.batch_size, self.num_tp_q_heads, self.head_dim), dtype=torch.float32, device=self.device
        ).to(self.torch_dtype)
        # Generate kv cache tensor
        self.k_cache = torch.randn(
            (self.num_pages, self.num_tp_k_heads, self.page_size, self.head_dim),
            dtype=torch.float32,
            device=self.device,
        ).to(self.torch_dtype)
        self.v_cache = torch.randn(
            (self.num_pages, self.num_tp_v_heads, self.page_size, self.head_dim),
            dtype=torch.float32,
            device=self.device,
        ).to(self.torch_dtype)
        # Generate bmm scales
        self.bmm1_scale, self.bmm2_scale = 1.0, 1.0
        # Generate workspace buffer
        self.workspace_buffer = torch.zeros((512 * 1024 * 1024,), dtype=torch.uint8, device=self.device)
        # Generate page_table
        self.page_table = torch.randperm(self.num_pages, dtype=torch.int32, device=self.device)[
            : self.batch_size * self.seq_len // self.page_size
        ]
        self.page_table = (
            self.page_table.view(self.batch_size, self.seq_len // self.page_size)
            .repeat_interleave(self.page_size, dim=1)
            .contiguous()
        )
        # Generate cache_seqlens_int32
        self.cache_seqlens_int32 = torch.full((self.batch_size,), self.seq_len, dtype=torch.int32, device=self.device)

    def forward(self, q: torch.Tensor, k: Optional[torch.Tensor] = None, v: Optional[torch.Tensor] = None):
        """Forward pass for the TRTLLM MHA Attention Backend

        Args:
            q (torch.Tensor): Query tensor
            k (Optional[torch.Tensor]): Key tensor
            v (Optional[torch.Tensor]): Value tensor
        """
        # Convert query tensor dtype
        if self.torch_dtype == torch.float8_e4m3fn:
            q = q.to(torch.float8_e4m3fn)
        q = q.contiguous().view(-1, self.num_tp_q_heads, self.head_dim)
        # Get k_cache
        if k is not None:
            k_cache = k.view(-1, self.page_size, self.num_tp_k_heads, self.head_dim).permute(0, 2, 1, 3)
        else:
            k_cache = self.k_cache
        # Get v_cache
        if v is not None:
            v_cache = v.view(-1, self.page_size, self.num_tp_v_heads, self.head_dim).permute(0, 2, 1, 3)
        else:
            v_cache = self.v_cache
        kv_cache = (k_cache, v_cache)
        # Attention core
        o = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            query=q,
            kv_cache=kv_cache,
            workspace_buffer=self.workspace_buffer,
            block_tables=self.page_table,
            seq_lens=self.cache_seqlens_int32,
            max_seq_len=self.max_seq_len,
            bmm1_scale=self.bmm1_scale,
            bmm2_scale=self.bmm2_scale,
            window_left=self.sliding_window_size,
            sinks=None,
            out_dtype=self.torch_dtype,
            backend="auto",
        )
        return o.view(-1, self.num_tp_q_heads * self.head_dim)
