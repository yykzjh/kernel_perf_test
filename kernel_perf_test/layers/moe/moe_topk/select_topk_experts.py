import torch
import torch.nn as nn

from sglang.srt.layers.moe.topk import select_experts, ExpertLocationDispatchInfo, TopKConfig, StandardTopKOutput


class SelectTopKExperts(nn.Module):
    def __init__(self, topk: int, batch_size: int, hidden_size: int, num_experts: int):
        super().__init__()
        # Initialize parameters
        self.topk = topk
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        # Initialize hidden_states and router_logits
        self.hidden_states = torch.randn((self.batch_size, self.hidden_size), device="cuda", dtype=torch.bfloat16)
        self.router_logits = torch.randn((self.batch_size, self.num_experts), device="cuda", dtype=torch.bfloat16)
        # Initialize num_token_non_padded
        self.num_token_non_padded = torch.ones((1,), device="cuda", dtype=torch.int32) * self.batch_size
        # Initialize expert_location_dispatch_info
        self.num_redundant_experts = min(self.num_experts, 32)
        partial_logical_to_all_physical_map = torch.full(
            (self.num_experts, self.num_experts + self.num_redundant_experts), -1, device="cuda", dtype=torch.int64
        )
        partial_logical_to_all_physical_map[:, 0] = torch.arange(self.num_experts, device="cuda", dtype=torch.int64)
        partial_logical_to_all_physical_map[: self.num_redundant_experts, 1] = torch.arange(
            self.num_experts, self.num_experts + self.num_redundant_experts, device="cuda", dtype=torch.int64
        )
        partial_logical_to_all_physical_map_num_valid = torch.ones(
            (self.num_experts,), device="cuda", dtype=torch.int64
        )
        partial_logical_to_all_physical_map_num_valid[: self.num_redundant_experts] = 2
        self.expert_location_dispatch_info = ExpertLocationDispatchInfo(
            ep_dispatch_algorithm="fake",
            partial_logical_to_rank_dispatch_physical_map=None,
            partial_logical_to_all_physical_map=partial_logical_to_all_physical_map,
            partial_logical_to_all_physical_map_num_valid=partial_logical_to_all_physical_map_num_valid,
            num_physical_experts=self.num_experts + self.num_redundant_experts,
        )
        # Initialize topk_config
        self.topk_config = TopKConfig(
            top_k=self.topk,
            use_grouped_topk=False,
            topk_group=None,
            num_expert_group=None,
            renormalize=True,
            num_fused_shared_experts=0,
            custom_routing_function=None,
            correction_bias=None,
            torch_native=False,
            routed_scaling_factor=None,
            apply_routed_scaling_factor_on_output=False,
            output_format=None,
        )

    def forward(self) -> StandardTopKOutput:
        return select_experts(
            hidden_states=self.hidden_states,
            router_logits=self.router_logits,
            topk_config=self.topk_config,
            num_token_non_padded=self.num_token_non_padded,
            expert_location_dispatch_info=self.expert_location_dispatch_info,
        )
