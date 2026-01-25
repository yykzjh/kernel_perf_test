import os
import tqdm
import shutil
import random
import itertools
from typing import List, Tuple
import matplotlib.pyplot as plt
import warnings

# Ignore Warning
warnings.filterwarnings("ignore", category=Warning)

import torch
from deep_gemm import m_grouped_fp8_gemm_nt_masked

from kernel_perf_test import utils
from kernel_perf_test.analysis.utils import calculate_deepgemm_masked_gemm_tflops


def get_output_sub_dir_path(
    output_dir_path: str,
    E: int,
    expected_m: int,
    N: int,
    K: int,
    num_max_sms: int,
    num_peo_rounds: int,
) -> str:
    return os.path.join(
        output_dir_path,
        f"E-{E}_expected_m-{expected_m}_N-{N}_K-{K}_num_max_sms-{num_max_sms}_num_peo_rounds-{num_peo_rounds}",
    )


def analyse_deepgemm_single_sms_tflops(
    E: int,
    expected_m: int,
    N: int = 5120,
    K: int = 6144,
    num_start_sms: int = 1,
    num_max_sms: int = -1,
    num_peo_rounds: int = 2,
    num_warmups: int = 2,
    num_tests: int = 5,
    position_shift: Tuple[int, int] = (2, 0),
    draw_figure: bool = True,
    output_dir_path: str = "./deepgemm_masked_sms_tflops_analysis_result",
) -> int:
    """Analyse Single SMs tflops of DeepGEMM
    Args:
        E (int): The number of local experts (num_groups).
        expected_m (int): The expected number of masked elements.
        N (int): The number of output size.
        K (int): The number of input size.
        num_start_sms (int): The start number of SMs.
        num_max_sms (int): The maximum number of SMs.
        num_peo_rounds (int): The number of PEO rounds.
        draw_figure (bool): Whether to draw the figure.
        output_dir_path (str): The path to save the analysis result.
    Returns:
        int: The number of SMs that the TFLOPS is not improved.
    """
    if num_max_sms == -1:
        num_max_sms = utils.get_num_device_sms()
    # Make sure the analysis result directory exists
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path, exist_ok=True)
    output_sub_dir_path = get_output_sub_dir_path(
        output_dir_path=output_dir_path,
        E=E,
        expected_m=expected_m,
        N=N,
        K=K,
        num_max_sms=num_max_sms,
        num_peo_rounds=num_peo_rounds,
    )
    if os.path.exists(output_sub_dir_path):
        shutil.rmtree(output_sub_dir_path)
    os.makedirs(output_sub_dir_path, exist_ok=True)
    os.makedirs(os.path.join(output_sub_dir_path, "traces"), exist_ok=True)
    # Initialize device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # Initialize weight and bias with fp8 dtype
    weight_fp8 = torch.randn((E, N, K), dtype=torch.float32, device=device).to(torch.float8_e4m3fn)
    weight_scale = torch.randn((E, N // 128, K // 128), dtype=torch.float32, device=device)
    masked_m = torch.empty((E,), device=device, dtype=torch.int)
    for i in range(E):
        masked_m[i] = int(expected_m * random.uniform(0.7, 1.3))
    max_m = (masked_m.max().item() + 127) // 128 * 128
    # Initialize input_fp8 and input_scale tensors
    input_fp8 = torch.randn(
        (E, max_m, K),
        dtype=torch.float32,
        device=device,
    ).to(torch.float8_e4m3fn)
    input_scale = torch.randn(
        (E, K // 128, max_m),
        dtype=torch.float32,
        device=device,
    ).permute(0, 2, 1)
    # Initialize output tensor
    output = torch.empty((E, max_m, N), device=device, dtype=torch.bfloat16)
    # Initialize PEO
    assert E % num_peo_rounds == 0
    E_per_round = E // num_peo_rounds
    streams_list = [torch.cuda.Stream() for _ in range(num_peo_rounds - 1)]

    # Iterate over number of SMs
    normal_sms_list = []
    peo_naive_sms_list = []
    peo_overlap_sms_list = []
    tflops_normal_list = []
    tflops_peo_naive_list = []
    tflops_peo_overlap_list = []
    compare_normal_score_sum = 0
    compare_peo_naive_score_sum = 0
    real_num_max_sms = min(utils.get_num_device_sms(), max(num_max_sms, 1))
    for num_sms in range(num_start_sms, real_num_max_sms + 1):
        # Initialize compare scores
        compare_normal_score = 0
        compare_peo_naive_score = 0
        # Normal test function
        normal_sms_list.append(num_sms)
        with utils.configure_deep_gemm_num_sms(num_sms):

            def normal_test_func():
                m_grouped_fp8_gemm_nt_masked(
                    (input_fp8, input_scale),
                    (weight_fp8, weight_scale),
                    output,
                    masked_m,
                    expected_m,
                    disable_ue8m0_cast=True,
                )

            # Capture graph
            normal_graph = utils.capture_graph(normal_test_func, num_warmups=num_warmups)
            # Profile
            normal_latency_us = utils.bench_kineto(
                lambda: normal_graph.replay(),
                num_warmups=num_warmups,
                num_tests=num_tests,
                trace_path=os.path.join(output_sub_dir_path, "traces", f"normal_trace_{num_sms}.json"),
                position_shift=position_shift,
                suppress_kineto_output=True,
            )
            normal_tflops = calculate_deepgemm_masked_gemm_tflops(E, expected_m, N, K, normal_latency_us)
            tflops_normal_list.append(normal_tflops)
        # PEO Naive test function
        peo_naive_sms_list.append(num_sms)
        with utils.configure_deep_gemm_num_sms(num_sms):

            def peo_naive_test_func():
                for round_id in range(num_peo_rounds):
                    m_grouped_fp8_gemm_nt_masked(
                        (
                            input_fp8[round_id * E_per_round : (round_id + 1) * E_per_round],
                            input_scale[round_id * E_per_round : (round_id + 1) * E_per_round],
                        ),
                        (
                            weight_fp8[round_id * E_per_round : (round_id + 1) * E_per_round],
                            weight_scale[round_id * E_per_round : (round_id + 1) * E_per_round],
                        ),
                        output[round_id * E_per_round : (round_id + 1) * E_per_round],
                        masked_m[round_id * E_per_round : (round_id + 1) * E_per_round],
                        expected_m,
                        disable_ue8m0_cast=True,
                    )

            # Capture graph
            peo_naive_graph = utils.capture_graph(peo_naive_test_func, num_warmups=num_warmups)
            # Profile
            peo_naive_latency_us = utils.bench_kineto(
                lambda: peo_naive_graph.replay(),
                num_warmups=num_warmups,
                num_tests=num_tests,
                trace_path=os.path.join(output_sub_dir_path, "traces", f"peo_naive_trace_{num_sms}.json"),
                position_shift=position_shift,
                suppress_kineto_output=True,
            )
            peo_naive_tflops = calculate_deepgemm_masked_gemm_tflops(E, expected_m, N, K, peo_naive_latency_us)
            tflops_peo_naive_list.append(peo_naive_tflops)
        # PEO Overlap test function
        if num_sms % num_peo_rounds == 0:
            peo_overlap_sms_list.append(num_sms)
            with utils.configure_deep_gemm_num_sms(num_sms // num_peo_rounds):

                def peo_overlap_test_func():
                    for stream in streams_list:
                        stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(torch.cuda.current_stream()):
                        m_grouped_fp8_gemm_nt_masked(
                            (input_fp8[0:E_per_round], input_scale[0:E_per_round]),
                            (weight_fp8[0:E_per_round], weight_scale[0:E_per_round]),
                            output[0:E_per_round],
                            masked_m[0:E_per_round],
                            expected_m,
                            disable_ue8m0_cast=True,
                        )
                    for i, stream in enumerate(streams_list):
                        round_id = i + 1
                        with torch.cuda.stream(stream):
                            m_grouped_fp8_gemm_nt_masked(
                                (
                                    input_fp8[round_id * E_per_round : (round_id + 1) * E_per_round],
                                    input_scale[round_id * E_per_round : (round_id + 1) * E_per_round],
                                ),
                                (
                                    weight_fp8[round_id * E_per_round : (round_id + 1) * E_per_round],
                                    weight_scale[round_id * E_per_round : (round_id + 1) * E_per_round],
                                ),
                                output[round_id * E_per_round : (round_id + 1) * E_per_round],
                                masked_m[round_id * E_per_round : (round_id + 1) * E_per_round],
                                expected_m,
                                disable_ue8m0_cast=True,
                            )
                        torch.cuda.current_stream().wait_stream(stream)

                peo_overlap_graph = utils.capture_graph(peo_overlap_test_func, num_warmups=num_warmups)
                peo_overlap_latency_us = utils.bench_kineto(
                    lambda: peo_overlap_graph.replay(),
                    num_warmups=num_warmups,
                    num_tests=num_tests,
                    trace_path=os.path.join(output_sub_dir_path, "traces", f"peo_overlap_trace_{num_sms}.json"),
                    position_shift=position_shift,
                    suppress_kineto_output=True,
                )
                peo_overlap_tflops = calculate_deepgemm_masked_gemm_tflops(E, expected_m, N, K, peo_overlap_latency_us)
                tflops_peo_overlap_list.append(peo_overlap_tflops)
                # Calculate compare scores
                compare_normal_score = (peo_overlap_tflops - normal_tflops) / normal_tflops
                compare_peo_naive_score = (peo_overlap_tflops - peo_naive_tflops) / peo_naive_tflops
        # Update compare scores sum
        compare_normal_score_sum += compare_normal_score
        compare_peo_naive_score_sum += compare_peo_naive_score

    # Draw figure
    if draw_figure:
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.plot(
            normal_sms_list,
            tflops_normal_list,
            marker="o",
            markersize=4,
            linewidth=1.5,
            label="Normal TFLOPS",
        )
        ax.plot(
            peo_naive_sms_list,
            tflops_peo_naive_list,
            marker="s",
            markersize=4,
            linewidth=1.5,
            label=f"PEO Naive TFLOPS (rounds={num_peo_rounds})",
        )
        ax.plot(
            peo_overlap_sms_list,
            tflops_peo_overlap_list,
            marker="^",
            markersize=4,
            linewidth=1.5,
            label=f"PEO Overlap TFLOPS (rounds={num_peo_rounds})",
        )
        ax.set_xlabel("Number of SMs")
        ax.set_ylabel("TFLOPS")
        ax.set_title("TFLOPS vs Number of SMs")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="best")
        fig.tight_layout()
        figure_path = os.path.join(output_sub_dir_path, "tflops_vs_num_sms.png")
        fig.savefig(figure_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return compare_normal_score_sum, compare_peo_naive_score_sum, figure_path


def analyse_deepgemm_batch_sms_tflops(
    ep_size_list: List[int],
    batch_size_per_rank_list: List[int],
    N_list: List[int],
    K_list: List[int],
    num_topk: int = 8,
    num_experts: int = 160,
    num_start_sms: int = 1,
    num_max_sms: int = -1,
    num_peo_rounds: int = 2,
    draw_figure: bool = True,
    output_dir_path: str = "./deepgemm_masked_sms_tflops_analysis_result",
) -> None:
    """Analyse Batch SMs tflops of DeepGEMM
    Args:
        num_experts_list: List of number of local experts.
        ep_size_list: List of expert size.
        num_topk_list: List of number of top-k elements.
        batch_size_per_rank_list: List of batch size per rank.
        N_list: List of number of output size.
        K_list: List of number of input size.
        num_start_sms: The start number of SMs.
        num_max_sms: The maximum number of SMs.
        num_peo_rounds: The number of PEO rounds.
        draw_figure: Whether to draw the figure.
        output_dir_path: The path to save the analysis result.
    """

    msg_list = []
    for ep_size, batch_size_per_rank, N, K in tqdm.tqdm(
        itertools.product(ep_size_list, batch_size_per_rank_list, N_list, K_list)
    ):
        assert num_experts % ep_size == 0
        num_local_experts = num_experts // ep_size
        expected_m = max(batch_size_per_rank * num_topk * ep_size // num_experts, 1)
        compare_normal_score, compare_peo_naive_score, figure_path = analyse_deepgemm_single_sms_tflops(
            E=num_local_experts,
            expected_m=expected_m,
            N=N,
            K=K,
            num_start_sms=num_start_sms,
            num_max_sms=num_max_sms,
            num_peo_rounds=num_peo_rounds,
            num_warmups=2,
            num_tests=10,
            position_shift=(2, 0),
            draw_figure=draw_figure,
            output_dir_path=output_dir_path,
        )
        if compare_normal_score < 0 or compare_peo_naive_score < 0:
            msg_list.append(
                f"ep_size: {ep_size}, batch_size_per_rank: {batch_size_per_rank}, num_local_experts: {num_local_experts}, expected_m: {expected_m}, "
                f"N: {N}, K: {K}, compare_normal_score: {compare_normal_score}, compare_peo_naive_score: {compare_peo_naive_score}, figure_path: {figure_path}\n"
            )
    if msg_list:
        print("\n".join(msg_list))
    else:
        print("\nAll results are positive")


if __name__ == "__main__":
    analyse_deepgemm_batch_sms_tflops(
        ep_size_list=[8, 16],
        batch_size_per_rank_list=[1, 16, 32, 64, 96, 128],
        N_list=[5120, 6144],
        K_list=[6144, 2560],
        num_topk=8,
        num_experts=160,
        num_start_sms=54,
        num_max_sms=-1,
        num_peo_rounds=2,
        draw_figure=True,
        output_dir_path="./deepgemm_masked_sms_tflops_analysis_result",
    )
