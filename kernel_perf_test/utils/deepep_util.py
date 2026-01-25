def calc_low_latency_max_token_per_rank(max_generate_batch_size: int, tp_size: int) -> int:
    ll_num_max_token_per_rank = (max_generate_batch_size + tp_size - 1) // tp_size
    # deepgemm masked with max_m < 64 get incorrect result, related: https://github.com/deepseek-ai/DeepGEMM/issues/268
    matched_tokens = [64, 128]
    if ll_num_max_token_per_rank > 128:
        ll_num_max_token_per_rank = ((ll_num_max_token_per_rank + 127) // 128) * 128
        return ll_num_max_token_per_rank
    for t in matched_tokens:
        if ll_num_max_token_per_rank <= t:
            ll_num_max_token_per_rank = t
            return ll_num_max_token_per_rank
    return 128
    