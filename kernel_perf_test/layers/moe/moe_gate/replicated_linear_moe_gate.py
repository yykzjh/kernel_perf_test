import torch
import torch.nn as nn
import torch.nn.functional as F


class TorchReplicatedLinear(nn.Module):
    """Minimal replicated linear used only for gate benchmark."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.skip_bias_add = skip_bias_add
        weight = torch.empty(
            output_size,
            input_size,
            dtype=params_dtype,
            device=device,
        )
        self.weight = nn.Parameter(weight)
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size, dtype=params_dtype, device=device))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        bias_for_linear = None if self.skip_bias_add else self.bias
        output = F.linear(x, self.weight, bias_for_linear)
        bias_for_return = self.bias if self.skip_bias_add else None
        return output, bias_for_return


class ReplicatedLinearMoEGate(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
    ):
        """Initialize ReplicatedLinearMoEGate."""
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gate = TorchReplicatedLinear(
            input_size=self.hidden_size,
            output_size=self.num_experts,
            bias=False,
            skip_bias_add=False,
            params_dtype=torch.bfloat16,
            device=device,
        )
        self.weight_bf16 = torch.randn(
            (self.num_experts, self.hidden_size),
            dtype=torch.bfloat16,
            device=device,
        )
        with torch.no_grad():
            self.gate.weight.copy_(self.weight_bf16)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        router_logits, _ = self.gate(hidden_states)
        return router_logits
