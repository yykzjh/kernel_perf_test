from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from sglang.srt.layers.quantization.fp8_utils import deepgemm_w8a8_block_fp8_linear_with_fallback


class DeepGemmFp8BlockLinear(nn.Module):
    """
    DeepGemmFp8BlockLinear is a class that implements the fp8 block linear layer using the DeepGemm library.
    """

    def __init__(self, N: int, K: int, block_size: List[int] = [128, 128]):
        """
        Initialize the DeepGemmFp8BlockLinear layer.

        Args:
            N (int): The number of intermediate dim.
            K (int): The number of hidden dim.
            block_size (List[int]): The block size for the matrix multiplication.
        """
        self.N = N
        self.K = K
        self.block_size = block_size
        # Initialize weight and bias with fp8 dtype
        self.weight_fp8 = torch.randn((N, K), dtype=torch.float32, device="cuda").to(torch.float8_e4m3fn)
        self.weight_scale = torch.randn((N // 128, K // 128), dtype=torch.float32, device="cuda")
        self.bias = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DeepGemmFp8BlockLinear layer.

        Args:
            input (torch.Tensor): The input tensor dtype must be torch.bfloat16. Must be of shape (M, K).

        Returns:
            torch.Tensor: The output tensor. dtype must be torch.bfloat16. Must be of shape (M, N).
        """
        # invoke the deepgemm fp8 block linear kernel
        return deepgemm_w8a8_block_fp8_linear_with_fallback(
            input=input,
            weight=self.weight_fp8,
            block_size=self.block_size,
            weight_scale=self.weight_scale,
            input_scale=None,
            bias=self.bias,
        )
