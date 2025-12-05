import math

import torch
import torch.nn as nn


from sglang.srt.models.qwen3_moe import Qwen3MoeAttention
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.dp_attention import DpPaddingMode
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode, CaptureHiddenMode
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool, MHATokenToKVPool

from kernel_perf_test.layers.attention.trtllm_mha_backend_mock import TRTLLMMHAAttnBackendMock


class Qwen3MoeAttentionLayer(nn.Module):
    def __init__(
        self,
        seq_len: int,
        max_seq_len: int,
        batch_size: int,
        hidden_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        num_pages: int,
        layer_num: int,
    ):
        """Initialize the Qwen3MoeAttentionLayer Layer

        Args:
            seq_len (int): Sequence length
            max_seq_len (int): Maximum sequence length
            batch_size (int): Batch size
            hidden_size (int): Hidden size
            num_q_heads (int): Number of query heads
            num_kv_heads (int): Number of key/value heads
            head_dim (int): Head dimension
            page_size (int): Page size
            num_pages (int): Number of pages
            layer_num (int): Layer number
        """
        super().__init__()
        # Get device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize the parameters
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_pages = num_pages
        self.page_size = page_size
        self.layer_num = layer_num
        self.max_seq_len = max_seq_len
        # Check parameters
        assert self.seq_len <= self.max_seq_len, "Sequence length must be less than or equal to maximum sequence length"
        if self.num_pages < self.batch_size * math.ceil(self.seq_len / self.page_size):
            self.num_pages = self.batch_size * math.ceil(self.seq_len / self.page_size)
        # Generate Qwen3MoeAttention init parameters
        self.qwen3_moe_attention_init_params = {
            "hidden_size": self.hidden_size,
            "num_heads": self.num_q_heads,
            "num_kv_heads": self.num_kv_heads,
            "layer_id": 0,
            "rope_theta": 10000000,
            "rope_scaling": None,
            "max_position_embeddings": self.max_seq_len,
            "head_dim": self.head_dim,
            "rms_norm_eps": 1e-06,
            "attention_bias": False,
            "quant_config": Fp8Config(
                is_checkpoint_fp8_serialized=True,
                activation_scheme="dynamic",
                ignored_layers=[],
                weight_block_size=[128, 128],
            ),
            "prefix": "model.layers.0.self_attn",
            "dual_chunk_attention_config": None,
            "alt_stream": torch.cuda.Stream(self.device),
        }
        # Initialize Qwen3MoeAttention
        self.qwen3_moe_attention = Qwen3MoeAttention(**self.qwen3_moe_attention_init_params)
        # Initialize positions
        self.positions = torch.zeros((self.batch_size,), device=self.device, dtype=torch.int64)
        # Initialize hidden_states
        self.hidden_states = torch.randn((self.batch_size, self.hidden_size), device=self.device, dtype=torch.bfloat16)
        # Initialize forward_batch
        self.forward_batch = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=self.batch_size,
            input_ids=torch.zeros((self.batch_size,), device=self.device, dtype=torch.int64),
            req_pool_indices=torch.zeros((self.batch_size,), device=self.device, dtype=torch.int32),
            seq_lens=torch.full((self.batch_size,), self.seq_len, device=self.device, dtype=torch.int32),
            out_cache_loc=torch.zeros((self.batch_size,), device=self.device, dtype=torch.int64),
            seq_lens_sum=self.seq_len * self.batch_size,
            orig_seq_lens=torch.full((self.batch_size,), self.seq_len, device=self.device, dtype=torch.int32),
            seq_lens_cpu=torch.full((self.batch_size,), self.seq_len, device="cpu", dtype=torch.int32),
            return_logprob=False,
            top_logprobs_nums=None,
            token_ids_logprobs=None,
            next_token_logits_buffer=torch.empty((self.batch_size, 151936), device=self.device, dtype=torch.float32),
            temp_scaled_logprobs=False,
            temperature=None,
            top_p_normalized_logprobs=False,
            top_p=None,
            positions=self.positions,
            extend_num_tokens=None,
            extend_seq_lens=None,
            extend_prefix_lens=None,
            extend_start_loc=None,
            extend_prefix_lens_cpu=None,
            extend_seq_lens_cpu=None,
            extend_logprob_start_lens_cpu=None,
            extend_input_logprob_token_ids_gpu=None,
            hidden_states=None,
            residual=None,
            model_specific_states=None,
            split_index=0,
            attn_attend_prefix_cache=None,
            num_prefix_chunks=None,
            prefix_chunk_idx=None,
            prefix_chunk_len=None,
            prefix_chunk_starts=None,
            prefix_chunk_seq_lens=None,
            prefix_chunk_cu_seq_lens=None,
            prefix_chunk_max_seq_lens=None,
            prefix_chunk_num_tokens=None,
            prefix_chunk_kv_indices=None,
            mha_return_lse=None,
            mm_inputs=None,
            encoder_cached=None,
            encoder_lens=None,
            encoder_lens_cpu=None,
            encoder_out_cache_loc=None,
            lora_ids=None,
            input_embeds=None,
            token_type_ids=None,
            sampling_info=None,
            req_to_token_pool=ReqToTokenPool(
                size=4096,
                max_context_len=self.max_seq_len,
                device="cuda",
                enable_memory_saver=False,
            ),
            token_to_kv_pool=MHATokenToKVPool(
                size=self.num_pages * self.page_size,
                page_size=self.page_size,
                dtype=torch.bfloat16,
                head_num=self.num_kv_heads,
                head_dim=self.head_dim,
                layer_num=self.layer_num,
                device="cuda",
                enable_memory_saver=False,
                start_layer=0,
                end_layer=self.layer_num,
                enable_kv_cache_copy=False,
            ),
            attn_backend=TRTLLMMHAAttnBackendMock(
                num_pages=self.num_pages,
                page_size=self.page_size,
                head_dim=self.head_dim,
                batch_size=self.batch_size,
                seq_len=self.seq_len,
                max_seq_len=self.max_seq_len,
                num_tp_q_heads=self.num_q_heads,
                num_tp_k_heads=self.num_kv_heads,
                num_tp_v_heads=self.num_kv_heads,
                sliding_window_size=-1,
                torch_dtype=torch.bfloat16,
            ),
            global_num_tokens_cpu=None,
            global_num_tokens_gpu=None,
            global_num_tokens_for_logprob_cpu=None,
            global_num_tokens_for_logprob_gpu=None,
            dp_padding_mode=DpPaddingMode.MAX_LEN,
            dp_local_start_pos=None,
            dp_local_num_tokens=None,
            global_dp_buffer_len=None,
            is_extend_in_batch=False,
            can_run_dp_cuda_graph=False,
            global_forward_mode=ForwardMode.DECODE,
            is_prefill_only=False,
            spec_info=None,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            padded_static_len=-1,
            num_token_non_padded=torch.tensor([self.batch_size], device=self.device, dtype=torch.int32),
            num_token_non_padded_cpu=None,
            mrope_positions=torch.zeros((3, self.batch_size), device=self.device, dtype=torch.int64),
        )

    def forward(self):
        return self.qwen3_moe_attention(
            positions=self.positions,
            hidden_states=self.hidden_states,
            forward_batch=self.forward_batch,
        )
