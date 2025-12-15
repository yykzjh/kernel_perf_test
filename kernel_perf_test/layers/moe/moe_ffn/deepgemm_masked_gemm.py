import random

import torch
import torch.nn as nn

from sglang.srt.layers.quantization import deep_gemm_wrapper


class DeepGEMMMaskedGemm(nn.Module):
    """
    DeepGEMMMaskedGemm is a class that implements the masked gemm layer using the DeepGemm library.
    """

    def __init__(
        self,
        num_local_experts: int,
        expected_m: int,
        N: int,
        K: int,
    ):
        """
        Initialize the DeepGEMMMaskedGemm layer.

        Args:
            num_local_experts (int): The number of local experts (num_groups).
            expected_m (int): The expected m.
            N (int): The number of output size.
            K (int): The number of input size.
        """
        super().__init__()
        self.num_local_experts = num_local_experts
        self.expected_m = expected_m
        self.N = N
        self.K = K
        # Initialize device
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # Initialize weight and bias with fp8 dtype
        self.weight_fp8 = torch.randn(
            (self.num_local_experts, self.N, self.K), dtype=torch.float32, device=self.device
        ).to(torch.float8_e4m3fn)
        self.weight_scale = torch.randn(
            (self.num_local_experts, self.N // 128, self.K // 128), dtype=torch.float32, device=self.device
        )
        # Initialize input and output tensors
        self.masked_m = torch.empty((self.num_local_experts,), device=self.device, dtype=torch.int)
        for i in range(self.num_local_experts):
            self.masked_m[i] = int(self.expected_m * random.uniform(0.7, 1.3))
        self.max_m = (self.masked_m.max().item() + 127) // 128 * 128
        # Initialize input_fp8 and input_scale tensors
        self.input_fp8 = torch.randn(
            (self.num_local_experts, self.max_m, self.K),
            dtype=torch.float32,
            device=self.device,
        ).to(torch.float8_e4m3fn)
        self.input_scale = (
            torch.randn(
                (self.num_local_experts, self.K // 512, self.max_m),
                dtype=torch.float32,
                device=self.device,
            )
            .to(torch.int32)
            .permute(0, 2, 1)
        )
        # Initialize output tensor
        self.output = torch.empty(
            (self.num_local_experts, self.max_m, self.N), device=self.device, dtype=torch.bfloat16
        )

    def forward(self):
        """Forward pass of the DeepGEMMMaskedGemm layer."""
        deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
            (self.input_fp8, self.input_scale),
            (self.weight_fp8, self.weight_scale),
            self.output,
            self.masked_m,
            self.expected_m,
        )
        return self.output
