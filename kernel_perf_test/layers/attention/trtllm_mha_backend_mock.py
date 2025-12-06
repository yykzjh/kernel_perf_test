import math
from typing import Optional

import torch
import torch.nn as nn
import flashinfer

from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend


class TRTLLMMHAAttnBackendMock(nn.Module):
    """TRTLLM MHA Attention Backend Mock"""

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
        # Generate workspace buffer
        self.workspace_buffer = torch.zeros((512 * 1024 * 1024,), dtype=torch.uint8, device=self.device)
        # Generate page_table
        self.page_table = torch.randperm(self.num_pages, dtype=torch.int32, device=self.device)[
            : self.batch_size * math.ceil(self.seq_len / self.page_size)
        ]
        self.page_table = self.page_table.view(self.batch_size, math.ceil(self.seq_len / self.page_size)).contiguous()
        # Generate cache_seqlens_int32
        self.cache_seqlens_int32 = torch.full((self.batch_size,), self.seq_len, dtype=torch.int32, device=self.device)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        self.forward_batch = forward_batch

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Run forward for decode using TRTLLM MHA kernel."""
        cache_loc = forward_batch.out_cache_loc
        if save_kv_cache and k is not None:
            forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v, layer.k_scale, layer.v_scale)

        q = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
        k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        # shape conversion:
        # [num_pages, page_size, num_kv_heads, head_dim] -> [num_pages, num_kv_heads, page_size, head_dim]
        k_cache = k_cache.view(-1, self.page_size, layer.tp_k_head_num, layer.head_dim).permute(0, 2, 1, 3)
        v_cache = v_cache.view(-1, self.page_size, layer.tp_v_head_num, layer.head_dim).permute(0, 2, 1, 3)
        kv_cache = (k_cache, v_cache)

        # TODO: add support for quantization
        q_scale = 1.0
        k_scale = layer.k_scale_float if getattr(layer, "k_scale_float", None) is not None else 1.0
        bmm1_scale = q_scale * k_scale * layer.scaling
        bmm2_scale = 1.0
        # sink: additional value per head in the denominator of the softmax.
        attention_sink = kwargs.get("sinks", None)

        # Call TRT-LLM kernel
        # raw_out: like q, [bs, acc_q_len, num_q_heads, head_dim] but with output dtype
        o = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            query=q,
            kv_cache=kv_cache,
            workspace_buffer=self.workspace_buffer,
            block_tables=self.page_table,
            seq_lens=self.cache_seqlens_int32,
            max_seq_len=self.max_seq_len,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            window_left=self.sliding_window_size,
            # TODO: add attention_sink operation or nvfp4 scale factor if needed
            sinks=attention_sink,
            out_dtype=q.dtype,
        )

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)
