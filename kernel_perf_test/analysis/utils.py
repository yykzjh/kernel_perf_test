def calculate_deepgemm_masked_gemm_tflops(E: int, expected_m: int, N: int, K: int, latency_us: float) -> float:
    """Calculate the TFLOPS of DeepGEMM Masked GEMM

    Args:
        E (int): The number of local experts (num_groups).
        expected_m (int): The expected number of masked elements.
        N (int): The number of output size.
        K (int): The number of input size.
        latency_us (float): The latency in microseconds.

    Returns:
        float: The TFLOPS.
    """
    return (2 * E * expected_m * N * K) / (latency_us * 1e6)
