import torch
import torch.nn as nn


from sglang.srt.layers.layernorm import RMSNorm


class FusedAddRMSNorm(nn.Module):
    def __init__(self, batch_size: int, hidden_size: int, eps: float = 1e-6):
        """Fused Add RMSNorm

        Args:
            batch_size (int): Batch size
            hidden_size (int): Hidden size
            eps (float, optional): Epsilon. Defaults to 1e-6.
        """
        super().__init__()
        # initialize parameters
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.eps = eps
        # Initialize RMSNorm
        self.rmsnorm = RMSNorm(self.hidden_size, eps=self.eps, var_hidden_size=None)
        # Initialize hidden_states
        self.hidden_states = torch.randn((self.batch_size, self.hidden_size), device="cuda", dtype=torch.bfloat16)
        # Initialize residual
        self.residual = torch.randn((self.batch_size, self.hidden_size), device="cuda", dtype=torch.bfloat16)

    def forward(self):
        return self.rmsnorm(self.hidden_states, self.residual)
