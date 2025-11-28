import torch
import torch.nn as nn

from sglang.srt.layers.quantization import deep_gemm_wrapper
from sglang.srt.layers.moe.ep_moe.kernels import silu_and_mul_masked_post_quant_fwd


class DeepGemmMaskedMoEFfn(nn.Module):
    """
    DeepGemmMaskedMoEFfn is a class that implements the MoE FFN layer using the DeepGemm library.
    """

    def __init__(
        self,
        E: int,
        N: int,
        K: int,
    ):
        """
        Initialize the DeepGemmMaskedMoEFfn layer.

        Args:
            E (int): The number of local experts (num_groups).
            N (int): The number of intermediate size.
            K (int): The number of hidden size.
        """
        super().__init__()
        self.E = E
        self.N = N
        self.K = K
        assert N % 2 == 0, "N must be even"
        # Initialize w13 and w2 with fp8 dtype
        self.w13_weight_fp8 = torch.randn((E, N, K), dtype=torch.float32, device="cuda").to(torch.float8_e4m3fn)
        self.w13_weight_scale = torch.randn((E, N // 128, K // 128), dtype=torch.float32, device="cuda")
        self.w2_weight_fp8 = torch.randn((E, K, N // 2), dtype=torch.float32, device="cuda").to(torch.float8_e4m3fn)
        self.w2_weight_scale = torch.randn((E, K // 128, N // 2 // 128), dtype=torch.float32, device="cuda")

    def forward(
        self,
        hidden_states_fp8: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        masked_m: torch.Tensor,
        expected_m: int,
    ) -> torch.Tensor:
        """
        Forward pass of the DeepGemmMaskedMoEFfn layer.

        Args:
            hidden_states_fp8 (torch.Tensor): The hidden states in fp8 dtype.
            hidden_states_scale (torch.Tensor): The hidden states scale.
            masked_m (torch.Tensor): The masked m.
            expected_m (int): The expected m.

        Returns:
            torch.Tensor: The output of the DeepGemmMaskedMoEFfn layer.
        """
        # GroupGemm-0
        num_groups, m, k = hidden_states_fp8.size()
        n = self.w13_weight_fp8.size(1)
        expected_m = min(expected_m, m)
        gateup_output = torch.empty((num_groups, m, n), device=hidden_states_fp8[0].device, dtype=torch.bfloat16)
        deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
            (hidden_states_fp8, hidden_states_scale),
            (self.w13_weight_fp8, self.w13_weight_scale),
            gateup_output,
            masked_m,
            expected_m,
        )

        # Act
        down_input = torch.empty(
            (
                gateup_output.shape[0],
                gateup_output.shape[1],
                gateup_output.shape[2] // 2,
            ),
            device=gateup_output.device,
            dtype=torch.float8_e4m3fn,
        )
        scale_block_size = 128
        down_input_scale = torch.empty(
            (
                gateup_output.shape[0],
                gateup_output.shape[1],
                gateup_output.shape[2] // 2 // scale_block_size,
            ),
            device=gateup_output.device,
            dtype=torch.float32,
        )
        silu_and_mul_masked_post_quant_fwd(
            gateup_output,
            down_input,
            down_input_scale,
            scale_block_size,
            masked_m,
            scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
        )
        del gateup_output

        # GroupGemm-1
        k = self.w2_weight_fp8.size(1)
        down_input_fp8 = (
            down_input,
            (
                down_input_scale
                if deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
                else deep_gemm_wrapper.get_mn_major_tma_aligned_tensor(down_input_scale)
            ),
        )
        down_output = torch.empty((num_groups, m, k), device=down_input.device, dtype=torch.bfloat16)
        deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
            down_input_fp8,
            (self.w2_weight_fp8, self.w2_weight_scale),
            down_output,
            masked_m,
            expected_m,
        )

        return down_output
