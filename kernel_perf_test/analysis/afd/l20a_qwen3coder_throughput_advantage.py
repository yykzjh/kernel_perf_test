import os
import math
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from kernel_perf_test import utils


VALID_MIN_NUM_GPUS = 2
VALID_MAX_NUM_GPUS = 72
VALID_MAX_BATCH_SIZE = 512
VALID_SEQ_LENS = [2048, 4096, 8192, 16384, 32768, 65536, 131072]

NUM_TOPK = 8
NUM_LAYERS = 62
NUM_EXPERTS = 160
NUM_REDUNDANT_EXPERTS = 32
KERNEL_INTERVAL_US_WITH_CUDA_GRAPH = 0.5


def load_performance_results(
    performance_results_dir_path: str,
    seq_len: int,
) -> Dict[str, pd.DataFrame]:
    """Load the performance results from the excel files"""
    fused_add_rmsnorm_performance_df = utils.load_performance_results_from_excel(
        file_path=os.path.join(performance_results_dir_path, r"fused_add_rmsnorm_performance.xlsx"),
        sheet_name="latency",
        index_col="batch_size",
    )
    fused_add_rmsnorm_performance_table = fused_add_rmsnorm_performance_df["latency / us"]
    qwen3_moe_attention_layer_performance_df = utils.load_performance_results_from_excel(
        file_path=os.path.join(performance_results_dir_path, r"qwen3_moe_attention_layer_performance.xlsx"),
        sheet_name="latency",
        index_col="batch_size",
    )
    qwen3_moe_attention_layer_performance_table = qwen3_moe_attention_layer_performance_df[f"seq_len: {seq_len} / us"]
    replicated_linear_moe_gate_performance_df = utils.load_performance_results_from_excel(
        file_path=os.path.join(performance_results_dir_path, r"replicated_linear_moe_gate_performance.xlsx"),
        sheet_name="latency",
        index_col="batch_size",
    )
    replicated_linear_moe_gate_performance_table = replicated_linear_moe_gate_performance_df["latency / us"]
    select_topk_experts_performance_df = utils.load_performance_results_from_excel(
        file_path=os.path.join(performance_results_dir_path, r"select_topk_experts_performance.xlsx"),
        sheet_name="latency",
        index_col="batch_size",
    )
    select_topk_experts_performance_table = select_topk_experts_performance_df["latency / us"]
    deepgemm_masked_moe_ffn_performance_df = utils.load_performance_results_from_excel(
        file_path=os.path.join(performance_results_dir_path, r"deepgemm_masked_moe_ffn_performance.xlsx"),
        sheet_name="latency",
        index_col="expected_m",
    )
    deepgemm_masked_moe_ffn_performance_table = deepgemm_masked_moe_ffn_performance_df

    return {
        "fused_add_rmsnorm_performance_table": fused_add_rmsnorm_performance_table,
        "qwen3_moe_attention_layer_performance_table": qwen3_moe_attention_layer_performance_table,
        "replicated_linear_moe_gate_performance_table": replicated_linear_moe_gate_performance_table,
        "select_topk_experts_performance_table": select_topk_experts_performance_table,
        "deepgemm_masked_moe_ffn_performance_table": deepgemm_masked_moe_ffn_performance_table,
    }


def calculate_reference_tpot_us(
    performance_results: Dict[str, pd.DataFrame],
    batch_size_per_gpu: int,
    num_gpus: int,
) -> Tuple[float, float, float, float, int, int]:
    """Calculate the reference TPOT / us, attn time / us, ffn time / us, ffn gpu time / us, number of local experts, expected m"""
    # Calculate moe parameters
    num_local_experts = (NUM_EXPERTS + NUM_REDUNDANT_EXPERTS) // num_gpus
    if num_local_experts > len(performance_results["deepgemm_masked_moe_ffn_performance_table"].columns):
        raise ValueError(
            f"Invalid number of local experts: {num_local_experts}. Valid number of local experts are in range [1, {len(performance_results['deepgemm_masked_moe_ffn_performance_table'].columns)}]"
        )
    expected_m = max(1, (batch_size_per_gpu * num_gpus * NUM_TOPK) // (NUM_EXPERTS + NUM_REDUNDANT_EXPERTS))
    if expected_m not in performance_results["deepgemm_masked_moe_ffn_performance_table"].index:
        raise ValueError(
            f"Invalid expected_m: {expected_m}. Valid expected_m are in range [1, {max(performance_results['deepgemm_masked_moe_ffn_performance_table'].index)}]"
        )
    # Calculate the time per layer
    fused_add_rmsnorm_latency = (
        performance_results["fused_add_rmsnorm_performance_table"].iloc[batch_size_per_gpu - 1]
        + KERNEL_INTERVAL_US_WITH_CUDA_GRAPH
    )
    if pd.isna(fused_add_rmsnorm_latency):
        raise ValueError(f"Fused Add RMSNorm latency is NaN for batch size per GPU: {batch_size_per_gpu}")
    qwen3_moe_attention_layer_latency = (
        performance_results["qwen3_moe_attention_layer_performance_table"].iloc[batch_size_per_gpu - 1]
        + KERNEL_INTERVAL_US_WITH_CUDA_GRAPH
    )
    if pd.isna(qwen3_moe_attention_layer_latency):
        raise ValueError(f"Qwen3 MOE Attention Layer latency is NaN for batch size per GPU: {batch_size_per_gpu}")
    replicated_linear_moe_gate_latency = (
        performance_results["replicated_linear_moe_gate_performance_table"].iloc[batch_size_per_gpu - 1]
        + KERNEL_INTERVAL_US_WITH_CUDA_GRAPH
    )
    if pd.isna(replicated_linear_moe_gate_latency):
        raise ValueError(f"Replicated Linear MOE Gate latency is NaN for batch size per GPU: {batch_size_per_gpu}")
    select_topk_experts_latency = (
        performance_results["select_topk_experts_performance_table"].iloc[batch_size_per_gpu - 1]
        + KERNEL_INTERVAL_US_WITH_CUDA_GRAPH
    )
    if pd.isna(select_topk_experts_latency):
        raise ValueError(f"Select TopK Experts latency is NaN for batch size per GPU: {batch_size_per_gpu}")
    deepgemm_masked_moe_ffn_latency = (
        performance_results["deepgemm_masked_moe_ffn_performance_table"].loc[expected_m][
            f"num_local_experts: {num_local_experts}"
        ]
        + KERNEL_INTERVAL_US_WITH_CUDA_GRAPH
    )
    if pd.isna(deepgemm_masked_moe_ffn_latency):
        raise ValueError(f"DeepGemm Masked MOE FFN latency is NaN for batch size per GPU: {batch_size_per_gpu}")
    time_us_per_layer = (
        2 * fused_add_rmsnorm_latency
        + qwen3_moe_attention_layer_latency
        + replicated_linear_moe_gate_latency
        + select_topk_experts_latency
        + deepgemm_masked_moe_ffn_latency
    )
    # Calculate reference TPOT / us
    reference_tpot_us = time_us_per_layer * NUM_LAYERS - KERNEL_INTERVAL_US_WITH_CUDA_GRAPH
    # Calculate attn and ffn time / us
    attn_time_us = (
        2 * fused_add_rmsnorm_latency
        + qwen3_moe_attention_layer_latency
        + replicated_linear_moe_gate_latency
        + select_topk_experts_latency
    ) * NUM_LAYERS
    ffn_time_us = reference_tpot_us - attn_time_us
    # Calculate FFN GPU time / us
    ffn_gpu_time_us = ffn_time_us * num_gpus
    return (
        float(reference_tpot_us),
        float(attn_time_us),
        float(ffn_time_us),
        float(ffn_gpu_time_us),
        int(num_local_experts),
        int(expected_m),
    )


def calculate_afd_tpot_us(
    performance_results: Dict[str, pd.DataFrame],
    micro_batch_size_list: List[int],
    num_attn_gpus: int,
    num_ffn_gpus: int,
) -> Tuple[float, Tuple[float, ...], Tuple[float, ...], float, int, int]:
    """Calculate the AFD TPOT / us, attn time per micro batch, ffn time per micro batch, ffn gpu time / us, number of local experts, expected m per micro batch"""
    # Initialize moe parameters for each micro batch
    num_local_experts = (NUM_EXPERTS + NUM_REDUNDANT_EXPERTS) // num_ffn_gpus
    if num_local_experts > len(performance_results["deepgemm_masked_moe_ffn_performance_table"].columns):
        raise ValueError(
            f"Invalid number of local experts: {num_local_experts}. Valid number of local experts are in range [1, {len(performance_results['deepgemm_masked_moe_ffn_performance_table'].columns)}]"
        )
    expected_m_per_micro_batch_list = []
    for micro_batch_size in micro_batch_size_list:
        expected_m_current_micro_batch = max(
            1, (micro_batch_size * num_attn_gpus * NUM_TOPK) // (NUM_EXPERTS + NUM_REDUNDANT_EXPERTS)
        )
        expected_m_per_micro_batch_list.append(expected_m_current_micro_batch)
        if expected_m_current_micro_batch not in performance_results["deepgemm_masked_moe_ffn_performance_table"].index:
            raise ValueError(
                f"Invalid expected_m_current_micro_batch: {expected_m_current_micro_batch}. Valid expected_m are in range [1, {max(performance_results['deepgemm_masked_moe_ffn_performance_table'].index)}]"
            )
    # Calculate attn and ffn time per micro batch
    attn_time_per_micro_batch_list = []
    ffn_time_per_micro_batch_list = []
    for micro_batch_idx, micro_batch_size in enumerate(micro_batch_size_list):
        # Calculate the attn time with current micro batch
        fused_add_rmsnorm_latency = (
            performance_results["fused_add_rmsnorm_performance_table"].iloc[micro_batch_size - 1]
            + KERNEL_INTERVAL_US_WITH_CUDA_GRAPH
        )
        if pd.isna(fused_add_rmsnorm_latency):
            raise ValueError(f"Fused Add RMSNorm latency is NaN for micro batch size: {micro_batch_size}")
        qwen3_moe_attention_layer_latency = (
            performance_results["qwen3_moe_attention_layer_performance_table"].iloc[micro_batch_size - 1]
            + KERNEL_INTERVAL_US_WITH_CUDA_GRAPH
        )
        if pd.isna(qwen3_moe_attention_layer_latency):
            raise ValueError(f"Qwen3 MOE Attention Layer latency is NaN for micro batch size: {micro_batch_size}")
        replicated_linear_moe_gate_latency = (
            performance_results["replicated_linear_moe_gate_performance_table"].iloc[micro_batch_size - 1]
            + KERNEL_INTERVAL_US_WITH_CUDA_GRAPH
        )
        if pd.isna(replicated_linear_moe_gate_latency):
            raise ValueError(f"Replicated Linear MOE Gate latency is NaN for micro batch size: {micro_batch_size}")
        select_topk_experts_latency = (
            performance_results["select_topk_experts_performance_table"].iloc[micro_batch_size - 1]
            + KERNEL_INTERVAL_US_WITH_CUDA_GRAPH
        )
        if pd.isna(select_topk_experts_latency):
            raise ValueError(f"Select TopK Experts latency is NaN for micro batch size: {micro_batch_size}")
        attn_time_with_current_micro_batch = (
            2 * fused_add_rmsnorm_latency
            + qwen3_moe_attention_layer_latency
            + replicated_linear_moe_gate_latency
            + select_topk_experts_latency
        )
        attn_time_per_micro_batch_list.append(attn_time_with_current_micro_batch)
        # Calculate the ffn time with current micro batch
        deepgemm_masked_moe_ffn_latency = (
            performance_results["deepgemm_masked_moe_ffn_performance_table"].loc[
                expected_m_per_micro_batch_list[micro_batch_idx]
            ][f"num_local_experts: {num_local_experts}"]
            + KERNEL_INTERVAL_US_WITH_CUDA_GRAPH
        )
        if pd.isna(deepgemm_masked_moe_ffn_latency):
            raise ValueError(f"DeepGemm Masked MOE FFN latency is NaN for micro batch size: {micro_batch_size}")
        ffn_time_with_current_micro_batch = deepgemm_masked_moe_ffn_latency
        ffn_time_per_micro_batch_list.append(ffn_time_with_current_micro_batch)
    # Initialize tpot_us with non-cyclic time
    afd_tpot_us = attn_time_per_micro_batch_list[0] + ffn_time_per_micro_batch_list[-1]
    for i in range(1, len(attn_time_per_micro_batch_list)):
        afd_tpot_us += max(attn_time_per_micro_batch_list[i], ffn_time_per_micro_batch_list[i - 1])
    # Calculate tpot_us with cyclic time
    cyclic_time = 0.0
    for i in range(len(attn_time_per_micro_batch_list)):
        cyclic_time += max(
            attn_time_per_micro_batch_list[i],
            ffn_time_per_micro_batch_list[
                (i - 1 + len(attn_time_per_micro_batch_list)) % len(attn_time_per_micro_batch_list)
            ],
        )
    # Calculate total TPOT / us
    afd_tpot_us += cyclic_time * (NUM_LAYERS - 1)
    afd_tpot_us -= KERNEL_INTERVAL_US_WITH_CUDA_GRAPH
    # Calculate FFN GPU time / us
    ffn_gpu_time_us = sum(ffn_time_per_micro_batch_list) * NUM_LAYERS * num_ffn_gpus
    # Calculate non-overlap extent of Attn and FFN per micro batch
    attn_non_overlap_time_us = 0.0
    ffn_non_overlap_time_us = 0.0
    for i in range(len(attn_time_per_micro_batch_list)):
        attn_time_cur_micro_batch = attn_time_per_micro_batch_list[i]
        ffn_time_prev_micro_batch = ffn_time_per_micro_batch_list[
            (i - 1 + len(attn_time_per_micro_batch_list)) % len(attn_time_per_micro_batch_list)
        ]
        if attn_time_cur_micro_batch > ffn_time_prev_micro_batch:
            attn_non_overlap_time_us += attn_time_cur_micro_batch - ffn_time_prev_micro_batch
        else:
            ffn_non_overlap_time_us += ffn_time_prev_micro_batch - attn_time_cur_micro_batch
    attn_non_overlap_time_us *= NUM_LAYERS
    ffn_non_overlap_time_us *= NUM_LAYERS
    return (
        float(afd_tpot_us),
        tuple([float(x) for x in attn_time_per_micro_batch_list]),
        tuple([float(x) for x in ffn_time_per_micro_batch_list]),
        float(ffn_gpu_time_us),
        int(num_local_experts),
        sum(expected_m_per_micro_batch_list) // len(expected_m_per_micro_batch_list),
        float(attn_non_overlap_time_us),
        float(ffn_non_overlap_time_us),
    )


def calculate_reference_statistics(
    performance_results: Dict[str, pd.DataFrame],
    num_gpus: int,
) -> pd.DataFrame:
    """Calculate the statistics for the reference mode and return the statistics as a DataFrame"""
    reference_batch_size_per_gpu_list = []
    reference_attn_time_us_list = []  # Attn time / us
    reference_ffn_time_us_list = []  # FFN time / us
    reference_ffn_gpu_time_us_list = []  # FFN GPU time / us
    reference_num_local_experts_list = []  # Number of local experts
    reference_expected_m_list = []  # Expected m
    reference_tpot_us_list = []  # TPOT / us
    reference_throughput_list = []  # tokens_per_gpu(batch_size_per_gpu) / s
    # Iterate over the batch sizes
    for batch_size_per_gpu in range(1, VALID_MAX_BATCH_SIZE + 1):
        try:
            # Calculate reference TPOT / us
            (
                reference_tpot_us,
                attn_time_us,
                ffn_time_us,
                ffn_gpu_time_us,
                num_local_experts,
                expected_m,
            ) = calculate_reference_tpot_us(
                performance_results=performance_results,
                batch_size_per_gpu=batch_size_per_gpu,
                num_gpus=num_gpus,
            )
            # Calculate the throughput tokens per GPU per second
            throughput_tokens_per_gpu_per_second = batch_size_per_gpu * 1e6 / reference_tpot_us
            # Save the reference TPOT / us and throughput tokens per GPU per second
            reference_batch_size_per_gpu_list.append(batch_size_per_gpu)
            reference_attn_time_us_list.append(attn_time_us)
            reference_ffn_time_us_list.append(ffn_time_us)
            reference_ffn_gpu_time_us_list.append(ffn_gpu_time_us)
            reference_num_local_experts_list.append(num_local_experts)
            reference_expected_m_list.append(expected_m)
            reference_tpot_us_list.append(reference_tpot_us)
            reference_throughput_list.append(throughput_tokens_per_gpu_per_second)
        except Exception:
            continue
    # Convert the statistics to a DataFrame
    reference_statistics_df = pd.DataFrame(
        zip(
            reference_batch_size_per_gpu_list,
            reference_attn_time_us_list,
            reference_ffn_time_us_list,
            reference_ffn_gpu_time_us_list,
            reference_num_local_experts_list,
            reference_expected_m_list,
            reference_tpot_us_list,
            reference_throughput_list,
        ),
        columns=[
            "Batch Size per GPU",
            "Attn Time / us",
            "FFN Time / us",
            "FFN GPU Time / us",
            "Number of Local Experts",
            "Expected m",
            "TPOT / us",
            "Throughput / tokens/gpu/s",
        ],
    )
    return reference_statistics_df


def calculate_afd_statistics(
    performance_results: Dict[str, pd.DataFrame],
    num_gpus: int,
) -> pd.DataFrame:
    """Calculate the statistics for the AFD best model and return the statistics as a DataFrame"""
    afd_batch_size_per_gpu_list = []
    afd_num_attn_gpus_list = []
    afd_num_ffn_gpus_list = []
    afd_num_micro_batches_list = []
    afd_micro_batch_size_list_list = []
    afd_attn_time_per_micro_batch_list_list = []
    afd_ffn_time_per_micro_batch_list_list = []
    afd_ffn_gpu_time_us_list = []  # FFN GPU time / us
    afd_num_local_experts_list = []  # Number of local experts
    afd_expected_m_list = []  # Expected m
    afd_attn_non_overlap_time_us_list = []  # Attn non-overlap time / us
    afd_ffn_non_overlap_time_us_list = []  # FFN non-overlap time / us
    afd_tpot_us_list = []  # TPOT / us
    afd_throughput_list = []  # tokens_per_gpu(batch_size_per_gpu) / s
    # Iterate over the batch sizes
    for batch_size_per_gpu in range(1, VALID_MAX_BATCH_SIZE + 1):
        # Iterate over the deployment strategies
        for num_ffn_gpus in range(1, num_gpus):
            # Check if the number of experts is divisible by the number of FFN GPUs
            if (NUM_EXPERTS + NUM_REDUNDANT_EXPERTS) % num_ffn_gpus != 0:
                continue
            # Determine the number of attention GPUs
            num_attn_gpus = num_gpus - num_ffn_gpus
            # Calculate the real batch size per attention GPU
            real_batch_size_per_attention_gpu = math.ceil(batch_size_per_gpu * num_gpus / num_attn_gpus)
            # The number of micro batches: 3, 4
            for num_micro_batches in [3, 4]:
                # Calculate the micro batch sizes
                micro_batch_size_list = [real_batch_size_per_attention_gpu // num_micro_batches] * num_micro_batches
                # Distribute the remainder to the first few micro batches
                for i in range(real_batch_size_per_attention_gpu % num_micro_batches):
                    micro_batch_size_list[i] += 1
                try:
                    # Calculate the TPOT / us
                    (
                        afd_tpot_us,
                        attn_time_per_micro_batch_list,
                        ffn_time_per_micro_batch_list,
                        ffn_gpu_time_us,
                        num_local_experts,
                        expected_m,
                        attn_non_overlap_time_us,
                        ffn_non_overlap_time_us,
                    ) = calculate_afd_tpot_us(
                        performance_results=performance_results,
                        micro_batch_size_list=micro_batch_size_list,
                        num_attn_gpus=num_attn_gpus,
                        num_ffn_gpus=num_ffn_gpus,
                    )
                    # Calculate the throughput tokens per GPU per second
                    throughput_tokens_per_gpu_per_second = batch_size_per_gpu * 1e6 / afd_tpot_us
                    # Save statistics
                    afd_batch_size_per_gpu_list.append(batch_size_per_gpu)
                    afd_num_attn_gpus_list.append(num_attn_gpus)
                    afd_num_ffn_gpus_list.append(num_ffn_gpus)
                    afd_num_micro_batches_list.append(num_micro_batches)
                    afd_micro_batch_size_list_list.append(tuple(micro_batch_size_list))
                    afd_attn_time_per_micro_batch_list_list.append(attn_time_per_micro_batch_list)
                    afd_ffn_time_per_micro_batch_list_list.append(ffn_time_per_micro_batch_list)
                    afd_ffn_gpu_time_us_list.append(ffn_gpu_time_us)
                    afd_num_local_experts_list.append(num_local_experts)
                    afd_expected_m_list.append(expected_m)
                    afd_attn_non_overlap_time_us_list.append(attn_non_overlap_time_us)
                    afd_ffn_non_overlap_time_us_list.append(ffn_non_overlap_time_us)
                    afd_tpot_us_list.append(afd_tpot_us)
                    afd_throughput_list.append(throughput_tokens_per_gpu_per_second)
                except Exception:
                    continue
    # Convert the statistics to a DataFrame
    afd_statistics_df = pd.DataFrame(
        zip(
            afd_batch_size_per_gpu_list,
            afd_num_attn_gpus_list,
            afd_num_ffn_gpus_list,
            afd_num_micro_batches_list,
            afd_micro_batch_size_list_list,
            afd_attn_time_per_micro_batch_list_list,
            afd_ffn_time_per_micro_batch_list_list,
            afd_ffn_gpu_time_us_list,
            afd_num_local_experts_list,
            afd_expected_m_list,
            afd_attn_non_overlap_time_us_list,
            afd_ffn_non_overlap_time_us_list,
            afd_tpot_us_list,
            afd_throughput_list,
        ),
        columns=[
            "Batch Size per GPU",
            "Number of Attention GPUs",
            "Number of FFN GPUs",
            "Number of Micro Batches",
            "Micro Batch Size List",
            "Attention Time per Micro Batch",
            "FFN Time per Micro Batch",
            "FFN GPU Time / us",
            "Number of Local Experts",
            "Expected m",
            "Attn Non-overlap Time / us",
            "FFN Non-overlap Time / us",
            "TPOT / us",
            "Throughput / tokens/gpu/s",
        ],
    )
    return afd_statistics_df


def save_reference_and_afd_statistics_to_csv(
    reference_statistics_df: pd.DataFrame,
    afd_statistics_df: pd.DataFrame,
    current_analysis_result_subdir_path: str,
) -> None:
    """Save the reference and AFD statistics to the csv files"""

    # Process the statistics DataFrame
    def round_tuple_elements(tpl, decimals=2):
        if isinstance(tpl, tuple):
            return tuple(round(x, decimals) for x in tpl)
        return tpl

    # Round the tuple elements in the statistics DataFrame
    temp_reference_statistics_df = reference_statistics_df.round(2)
    temp_afd_statistics_df = afd_statistics_df.round(2)
    for col in reference_statistics_df.columns:
        first_valid_item = (
            temp_reference_statistics_df[col].dropna().iloc[0]
            if not temp_reference_statistics_df[col].dropna().empty
            else None
        )
        if isinstance(first_valid_item, tuple):
            temp_reference_statistics_df[col] = temp_reference_statistics_df[col].apply(round_tuple_elements)
    for col in afd_statistics_df.columns:
        first_valid_item = (
            temp_afd_statistics_df[col].dropna().iloc[0] if not temp_afd_statistics_df[col].dropna().empty else None
        )
        if isinstance(first_valid_item, tuple):
            temp_afd_statistics_df[col] = temp_afd_statistics_df[col].apply(round_tuple_elements)
    # Save the reference and AFD statistics to the csv files
    temp_reference_statistics_df.to_csv(
        os.path.join(current_analysis_result_subdir_path, "reference_statistics.csv"), index=False
    )
    temp_afd_statistics_df.to_csv(os.path.join(current_analysis_result_subdir_path, "afd_statistics.csv"), index=False)
    print(reference_statistics_df)
    print(afd_statistics_df)


def sample_dataframe(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    """Uniformly sample dataframe to reduce number of points"""
    if len(df) <= max_points:
        return df
    # Calculate step size for uniform sampling
    indices = [int(i * (len(df) - 1) / (max_points - 1)) for i in range(max_points)]
    return df.iloc[indices].copy()


def plot_throughput_comparison_vs_tpot(
    reference_statistics_df: pd.DataFrame,
    afd_best_statistics_df: pd.DataFrame,
    current_analysis_result_subdir_path: str,
    slo_tpot_us: int,
    max_points: int = 50,
) -> None:
    """Plot the reference and AFD statistics

    Args:
        reference_statistics_df: Reference statistics DataFrame
        afd_best_statistics_df: AFD best statistics DataFrame
        current_analysis_result_subdir_path: Path to save the plot
        slo_tpot_us: SLO TPOT in microseconds
        max_points (int, optional): Maximum number of points to plot for each curve (default: 50)
    """
    # Sort by TPOT / us for better line plotting
    reference_statistics_df = reference_statistics_df.sort_values("TPOT / us")
    afd_best_statistics_df = afd_best_statistics_df.sort_values("TPOT / us")

    # Apply sampling
    if len(reference_statistics_df) > max_points:
        reference_statistics_df = sample_dataframe(reference_statistics_df, max_points)
    if len(afd_best_statistics_df) > max_points:
        afd_best_statistics_df = sample_dataframe(afd_best_statistics_df, max_points)
    # Create figure and axis
    plt.figure(figsize=(16, 10))
    # Plot reference statistics
    plt.plot(
        reference_statistics_df["TPOT / us"],
        reference_statistics_df["Throughput / tokens/gpu/s"],
        marker="o",
        markersize=6,
        linewidth=2,
        label="Reference",
        color="blue",
    )
    # Plot AFD statistics
    plt.plot(
        afd_best_statistics_df["TPOT / us"],
        afd_best_statistics_df["Throughput / tokens/gpu/s"],
        marker="s",
        markersize=6,
        linewidth=2,
        label="AFD",
        color="red",
    )
    # Add annotations for AFD data points with alternating positions to avoid overlap
    annotation_positions = [(10, 10), (10, -20), (-80, 10), (-80, -20)]  # Alternating positions
    for idx, (_, row) in enumerate(afd_best_statistics_df.iterrows()):
        annotation_text = f"({int(row['Number of Attention GPUs'])}, {int(row['Number of FFN GPUs'])}, {int(row['Number of Micro Batches'])})"
        # Use alternating positions to reduce overlap
        pos = annotation_positions[idx % len(annotation_positions)]
        plt.annotate(
            annotation_text,
            xy=(row["TPOT / us"], row["Throughput / tokens/gpu/s"]),
            xytext=pos,
            textcoords="offset points",
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color="gray", lw=0.8),
        )
    # Add vertical dashed line at SLO TPOT
    plt.axvline(x=slo_tpot_us, color="green", linestyle="--", linewidth=2, label=f"SLO TPOT ({slo_tpot_us} us)")
    # Set labels and title
    plt.xlabel("TPOT / us", fontsize=14, fontweight="bold")
    plt.ylabel("Throughput / tokens/gpu/s", fontsize=14, fontweight="bold")
    plt.title("Reference vs AFD Throughput Comparison", fontsize=16, fontweight="bold")
    plt.legend(fontsize=12, loc="best")
    plt.grid(True, alpha=0.3)
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    # Save the figure
    output_path = os.path.join(current_analysis_result_subdir_path, "throughput_comparison_vs_tpot.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def filter_common_batch_sizes(
    reference_statistics_df: pd.DataFrame,
    afd_best_statistics_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Filter the common batch sizes in the two DataFrames"""
    # Get the intersection of the "Batch Size per GPU" columns in the two DataFrames
    reference_batch_sizes = set(reference_statistics_df["Batch Size per GPU"])
    afd_batch_sizes = set(afd_best_statistics_df["Batch Size per GPU"])
    common_batch_sizes = reference_batch_sizes.intersection(afd_batch_sizes)
    # Only keep the rows in the intersection
    filtered_reference_statistics_df = reference_statistics_df[
        reference_statistics_df["Batch Size per GPU"].isin(common_batch_sizes)
    ]
    filtered_afd_best_statistics_df = afd_best_statistics_df[
        afd_best_statistics_df["Batch Size per GPU"].isin(common_batch_sizes)
    ]
    # Sort by Batch Size per GPU
    filtered_reference_statistics_df = filtered_reference_statistics_df.sort_values("Batch Size per GPU")
    filtered_afd_best_statistics_df = filtered_afd_best_statistics_df.sort_values("Batch Size per GPU")
    return filtered_reference_statistics_df, filtered_afd_best_statistics_df


def plot_tpot_comparison_vs_batch_size(
    reference_statistics_df: pd.DataFrame,
    afd_best_statistics_df: pd.DataFrame,
    current_analysis_result_subdir_path: str,
    max_points: int = 50,
) -> None:
    """Plot the reference and AFD statistics

    Args:
        reference_statistics_df: Reference statistics DataFrame
        afd_best_statistics_df: AFD best statistics DataFrame
        current_analysis_result_subdir_path: Path to save the plot
        max_points (int, optional): Maximum number of points to plot for each curve (default: 50)
    """
    # Sort by Batch Size per GPU
    reference_statistics_df = reference_statistics_df.sort_values("Batch Size per GPU")
    afd_best_statistics_df = afd_best_statistics_df.sort_values("Batch Size per GPU")
    # Apply sampling
    if len(reference_statistics_df) > max_points:
        reference_statistics_df = sample_dataframe(reference_statistics_df, max_points)
    if len(afd_best_statistics_df) > max_points:
        afd_best_statistics_df = sample_dataframe(afd_best_statistics_df, max_points)
    # Create figure and axis
    plt.figure(figsize=(16, 10))
    # Plot reference statistics
    plt.plot(
        reference_statistics_df["Batch Size per GPU"],
        reference_statistics_df["TPOT / us"],
        marker="o",
        markersize=6,
        linewidth=2,
        label="Reference",
        color="blue",
    )
    # Plot AFD statistics
    plt.plot(
        afd_best_statistics_df["Batch Size per GPU"],
        afd_best_statistics_df["TPOT / us"],
        marker="s",
        markersize=6,
        linewidth=2,
        label="AFD",
        color="red",
    )
    # Set labels and title
    plt.xlabel("Batch Size per GPU", fontsize=14, fontweight="bold")
    plt.ylabel("TPOT / us", fontsize=14, fontweight="bold")
    plt.title("TPOT Comparison vs Batch Size per GPU", fontsize=16, fontweight="bold")
    # Add grid
    plt.grid(True, alpha=0.3, linestyle="--")
    # Add legend
    plt.legend(fontsize=12, loc="best")
    # Adjust layout
    plt.tight_layout()
    # Save the plot
    output_path = os.path.join(current_analysis_result_subdir_path, "tpot_comparison_vs_batch_size.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    # Close the plot
    plt.close()


def plot_proportion_change_vs_batch_size(
    filtered_reference_statistics_df: pd.DataFrame,
    filtered_afd_best_statistics_df: pd.DataFrame,
    current_analysis_result_subdir_path: str,
    max_points: int = 50,
) -> None:
    """Plot the proportion change of the reference and AFD statistics"""
    # Calculate metrics for each batch size
    batch_sizes = []
    afd_attn_increase_ratios = []  # AFD attention time increase ratio
    reference_ffn_to_attn_ratios = []  # Reference FFN to Attn ratio
    tpot_increase_to_attn_ratios = []  # TPOT increase to Attn ratio
    afd_attn_per_layer_increase_times = []
    reference_ffn_per_layer_times = []
    for _, ref_row in filtered_reference_statistics_df.iterrows():
        batch_size = ref_row["Batch Size per GPU"]
        ref_attn_time = ref_row["Attn Time / us"]
        ref_ffn_time = ref_row["FFN Time / us"]
        ref_tpot = ref_row["TPOT / us"]
        # Find the corresponding AFD row
        afd_row = filtered_afd_best_statistics_df[filtered_afd_best_statistics_df["Batch Size per GPU"] == batch_size]
        if len(afd_row) == 0:
            continue
        afd_row = afd_row.iloc[0]
        # Calculate the total attention time for AFD
        # Attention Time per Micro Batch is a tuple, sum up to get the attention time per layer
        afd_attn_time_per_micro_batch = afd_row["Attention Time per Micro Batch"]
        if isinstance(afd_attn_time_per_micro_batch, (list, tuple)):
            afd_attn_time_per_layer = sum(afd_attn_time_per_micro_batch)
        else:
            afd_attn_time_per_layer = afd_attn_time_per_micro_batch
        afd_attn_total_time = afd_attn_time_per_layer * NUM_LAYERS
        afd_tpot = afd_row["TPOT / us"]
        # Calculate 3 metrics
        # 1. AFD attention time increase ratio (percentage)
        afd_attn_increase_ratio = (afd_attn_total_time - ref_attn_time) / ref_attn_time * 100
        # 2. Reference FFN to Attn ratio (percentage)
        reference_ffn_to_attn_ratio = ref_ffn_time / ref_attn_time * 100
        # 3. TPOT increase to Attn ratio (percentage)
        tpot_increase_to_attn_ratio = (afd_tpot - ref_tpot) / ref_attn_time * 100
        # 4. AFD attn increase per layer (percentage)
        afd_attn_per_layer_increase_time = afd_attn_time_per_layer - ref_attn_time / NUM_LAYERS
        # 5. Reference FFN per layer (percentage)
        reference_ffn_per_layer_time = ref_ffn_time / NUM_LAYERS
        # Save the metrics
        batch_sizes.append(batch_size)
        afd_attn_increase_ratios.append(afd_attn_increase_ratio)
        reference_ffn_to_attn_ratios.append(reference_ffn_to_attn_ratio)
        tpot_increase_to_attn_ratios.append(tpot_increase_to_attn_ratio)
        afd_attn_per_layer_increase_times.append(afd_attn_per_layer_increase_time)
        reference_ffn_per_layer_times.append(reference_ffn_per_layer_time)
    # Create DataFrame for sampling
    metrics_df = pd.DataFrame(
        {
            "Batch Size per GPU": batch_sizes,
            "AFD Attn Increase Ratio": afd_attn_increase_ratios,
            "Reference FFN to Attn Ratio": reference_ffn_to_attn_ratios,
            "TPOT Increase to Attn Ratio": tpot_increase_to_attn_ratios,
            "AFD Attn per Layer Increase Time": afd_attn_per_layer_increase_times,
            "Reference FFN per Layer Time": reference_ffn_per_layer_times,
        }
    )
    # Sort by Batch Size per GPU
    metrics_df = metrics_df.sort_values("Batch Size per GPU")
    # Sample
    if len(metrics_df) > max_points:
        metrics_df = sample_dataframe(metrics_df, max_points)
    # Create figure
    plt.figure(figsize=(16, 10))
    # Plot 1:
    plt.plot(
        metrics_df["Batch Size per GPU"],
        metrics_df["AFD Attn Increase Ratio"],
        marker="o",
        markersize=6,
        linewidth=2,
        label="(AFD Attn - Ref Attn) / Ref Attn (%)",
        color="red",
    )
    plt.plot(
        metrics_df["Batch Size per GPU"],
        metrics_df["Reference FFN to Attn Ratio"],
        marker="s",
        markersize=6,
        linewidth=2,
        label="Ref FFN / Ref Attn (%)",
        color="blue",
    )
    plt.plot(
        metrics_df["Batch Size per GPU"],
        metrics_df["TPOT Increase to Attn Ratio"],
        marker="^",
        markersize=6,
        linewidth=2,
        label="(AFD TPOT - Ref TPOT) / Ref Attn (%)",
        color="green",
    )
    # Set labels and title
    plt.xlabel("Batch Size per GPU", fontsize=14, fontweight="bold")
    plt.ylabel("Percentage (%)", fontsize=14, fontweight="bold")
    plt.title("Proportion Changes vs Batch Size per GPU", fontsize=16, fontweight="bold")
    # Add grid
    plt.grid(True, alpha=0.3, linestyle="--")
    # Add legend
    plt.legend(fontsize=12, loc="best")
    # Adjust layout
    plt.tight_layout()
    # Save figure
    output_path = os.path.join(current_analysis_result_subdir_path, "proportion_change_vs_batch_size.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    # Close figure
    plt.close()
    # Plot 2:
    plt.figure(figsize=(16, 10))
    plt.plot(
        metrics_df["Batch Size per GPU"],
        metrics_df["AFD Attn per Layer Increase Time"],
        marker="o",
        markersize=6,
        linewidth=2,
        label="AFD Attn per Layer Increase Time",
        color="blue",
    )
    plt.plot(
        metrics_df["Batch Size per GPU"],
        metrics_df["Reference FFN per Layer Time"],
        marker="s",
        markersize=6,
        linewidth=2,
        label="Reference FFN per Layer Time",
        color="red",
    )
    plt.xlabel("Batch Size per GPU", fontsize=14, fontweight="bold")
    plt.ylabel("Time / us", fontsize=14, fontweight="bold")
    plt.title("AFD Attn and FFN per Layer Time Comparison vs Batch Size per GPU", fontsize=16, fontweight="bold")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend(fontsize=12, loc="best")
    plt.tight_layout()
    output_path = os.path.join(
        current_analysis_result_subdir_path, "afd_attn_and_ffn_per_layer_time_comparison_vs_batch_size.png"
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_moe_ffn_metrics_vs_batch_size(
    filtered_reference_statistics_df: pd.DataFrame,
    filtered_afd_best_statistics_df: pd.DataFrame,
    current_analysis_result_subdir_path: str,
    max_points: int = 50,
) -> None:
    """Plot the MOE FFN metrics vs batch size"""
    # Prepare data for all 4 plots
    batch_sizes = []
    ref_ffn_times = []
    afd_ffn_times = []
    ref_num_local_experts = []
    afd_num_local_experts = []
    ref_expected_m = []
    afd_expected_m = []
    ref_ffn_gpu_times = []
    afd_ffn_gpu_times = []
    for _, ref_row in filtered_reference_statistics_df.iterrows():
        batch_size = ref_row["Batch Size per GPU"]
        # Find the corresponding AFD row
        afd_row = filtered_afd_best_statistics_df[filtered_afd_best_statistics_df["Batch Size per GPU"] == batch_size]
        if len(afd_row) == 0:
            continue
        afd_row = afd_row.iloc[0]
        # Calculate AFD FFN time (sum of tuple * NUM_LAYERS)
        afd_ffn_time_per_micro_batch = afd_row["FFN Time per Micro Batch"]
        if isinstance(afd_ffn_time_per_micro_batch, (list, tuple)):
            afd_ffn_time_per_layer = sum(afd_ffn_time_per_micro_batch)
        else:
            afd_ffn_time_per_layer = afd_ffn_time_per_micro_batch
        afd_ffn_total_time = afd_ffn_time_per_layer * NUM_LAYERS
        # Collect data
        batch_sizes.append(batch_size)
        ref_ffn_times.append(ref_row["FFN Time / us"])
        afd_ffn_times.append(afd_ffn_total_time)
        ref_num_local_experts.append(ref_row["Number of Local Experts"])
        afd_num_local_experts.append(afd_row["Number of Local Experts"])
        ref_expected_m.append(ref_row["Expected m"])
        afd_expected_m.append(afd_row["Expected m"])
        ref_ffn_gpu_times.append(ref_row["FFN GPU Time / us"])
        afd_ffn_gpu_times.append(afd_row["FFN GPU Time / us"])
    # Create DataFrame for sampling
    metrics_df = pd.DataFrame(
        {
            "Batch Size per GPU": batch_sizes,
            "Ref FFN Time": ref_ffn_times,
            "AFD FFN Time": afd_ffn_times,
            "Ref Num Local Experts": ref_num_local_experts,
            "AFD Num Local Experts": afd_num_local_experts,
            "Ref Expected m": ref_expected_m,
            "AFD Expected m": afd_expected_m,
            "Ref FFN GPU Time": ref_ffn_gpu_times,
            "AFD FFN GPU Time": afd_ffn_gpu_times,
        }
    )
    # Sort by Batch Size per GPU
    metrics_df = metrics_df.sort_values("Batch Size per GPU")
    # Sample
    if len(metrics_df) > max_points:
        metrics_df = sample_dataframe(metrics_df, max_points)
    # Plot 1: FFN Time comparison
    plt.figure(figsize=(16, 10))
    plt.plot(
        metrics_df["Batch Size per GPU"],
        metrics_df["Ref FFN Time"],
        marker="o",
        markersize=6,
        linewidth=2,
        label="Reference FFN Time",
        color="blue",
    )
    plt.plot(
        metrics_df["Batch Size per GPU"],
        metrics_df["AFD FFN Time"],
        marker="s",
        markersize=6,
        linewidth=2,
        label="AFD FFN Time",
        color="red",
    )
    plt.xlabel("Batch Size per GPU", fontsize=14, fontweight="bold")
    plt.ylabel("FFN Time / us", fontsize=14, fontweight="bold")
    plt.title("FFN Time Comparison vs Batch Size per GPU", fontsize=16, fontweight="bold")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend(fontsize=12, loc="best")
    plt.tight_layout()
    output_path = os.path.join(current_analysis_result_subdir_path, "ffn_time_comparison_vs_batch_size.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    # Plot 2: Number of Local Experts comparison
    plt.figure(figsize=(16, 10))
    plt.plot(
        metrics_df["Batch Size per GPU"],
        metrics_df["Ref Num Local Experts"],
        marker="o",
        markersize=6,
        linewidth=2,
        label="Reference Number of Local Experts",
        color="blue",
    )
    plt.plot(
        metrics_df["Batch Size per GPU"],
        metrics_df["AFD Num Local Experts"],
        marker="s",
        markersize=6,
        linewidth=2,
        label="AFD Number of Local Experts",
        color="red",
    )
    plt.xlabel("Batch Size per GPU", fontsize=14, fontweight="bold")
    plt.ylabel("Number of Local Experts", fontsize=14, fontweight="bold")
    plt.title("Number of Local Experts Comparison vs Batch Size per GPU", fontsize=16, fontweight="bold")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend(fontsize=12, loc="best")
    plt.tight_layout()
    output_path = os.path.join(current_analysis_result_subdir_path, "num_local_experts_comparison_vs_batch_size.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    # Plot 3: Expected m comparison
    plt.figure(figsize=(16, 10))
    plt.plot(
        metrics_df["Batch Size per GPU"],
        metrics_df["Ref Expected m"],
        marker="o",
        markersize=6,
        linewidth=2,
        label="Reference Expected m",
        color="blue",
    )
    plt.plot(
        metrics_df["Batch Size per GPU"],
        metrics_df["AFD Expected m"],
        marker="s",
        markersize=6,
        linewidth=2,
        label="AFD Expected m",
        color="red",
    )
    plt.xlabel("Batch Size per GPU", fontsize=14, fontweight="bold")
    plt.ylabel("Expected m", fontsize=14, fontweight="bold")
    plt.title("Expected m Comparison vs Batch Size per GPU", fontsize=16, fontweight="bold")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend(fontsize=12, loc="best")
    plt.tight_layout()
    output_path = os.path.join(current_analysis_result_subdir_path, "expected_m_comparison_vs_batch_size.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    # Plot 4: FFN GPU Time comparison
    plt.figure(figsize=(16, 10))
    plt.plot(
        metrics_df["Batch Size per GPU"],
        metrics_df["Ref FFN GPU Time"],
        marker="o",
        markersize=6,
        linewidth=2,
        label="Reference FFN GPU Time",
        color="blue",
    )
    plt.plot(
        metrics_df["Batch Size per GPU"],
        metrics_df["AFD FFN GPU Time"],
        marker="s",
        markersize=6,
        linewidth=2,
        label="AFD FFN GPU Time",
        color="red",
    )
    plt.xlabel("Batch Size per GPU", fontsize=14, fontweight="bold")
    plt.ylabel("FFN GPU Time / us", fontsize=14, fontweight="bold")
    plt.title("FFN GPU Time Comparison vs Batch Size per GPU", fontsize=16, fontweight="bold")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend(fontsize=12, loc="best")
    plt.tight_layout()
    output_path = os.path.join(current_analysis_result_subdir_path, "ffn_gpu_time_comparison_vs_batch_size.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_afd_non_overlap_time_vs_batch_size(
    afd_best_statistics_df: pd.DataFrame,
    current_analysis_result_subdir_path: str,
    max_points: int = 50,
) -> None:
    """Plot the AFD non-overlap time vs batch size"""
    # Prepare data
    batch_sizes = []
    attn_non_overlap_ratios = []
    ffn_non_overlap_ratios = []
    for _, row in afd_best_statistics_df.iterrows():
        batch_size = row["Batch Size per GPU"]
        attn_non_overlap_time = row["Attn Non-overlap Time / us"]
        ffn_non_overlap_time = row["FFN Non-overlap Time / us"]
        tpot = row["TPOT / us"]
        # Calculate ratios as percentages
        attn_non_overlap_ratio = (attn_non_overlap_time / tpot) * 100
        ffn_non_overlap_ratio = (ffn_non_overlap_time / tpot) * 100
        batch_sizes.append(batch_size)
        attn_non_overlap_ratios.append(attn_non_overlap_ratio)
        ffn_non_overlap_ratios.append(ffn_non_overlap_ratio)
    # Create DataFrame for sampling
    metrics_df = pd.DataFrame(
        {
            "Batch Size per GPU": batch_sizes,
            "Attn Non-overlap Ratio": attn_non_overlap_ratios,
            "FFN Non-overlap Ratio": ffn_non_overlap_ratios,
        }
    )
    # Sort by Batch Size per GPU
    metrics_df = metrics_df.sort_values("Batch Size per GPU")
    # Sample
    if len(metrics_df) > max_points:
        metrics_df = sample_dataframe(metrics_df, max_points)
    # Create figure
    plt.figure(figsize=(16, 10))
    # Plot first line: Attn non-overlap ratio
    plt.plot(
        metrics_df["Batch Size per GPU"],
        metrics_df["Attn Non-overlap Ratio"],
        marker="o",
        markersize=6,
        linewidth=2,
        label="Attn Non-overlap Time / TPOT (%)",
        color="blue",
    )
    # Plot second line: FFN non-overlap ratio
    plt.plot(
        metrics_df["Batch Size per GPU"],
        metrics_df["FFN Non-overlap Ratio"],
        marker="s",
        markersize=6,
        linewidth=2,
        label="FFN Non-overlap Time / TPOT (%)",
        color="red",
    )
    # Set labels and title
    plt.xlabel("Batch Size per GPU", fontsize=14, fontweight="bold")
    plt.ylabel("Percentage (%)", fontsize=14, fontweight="bold")
    plt.title("AFD Non-overlap Time Ratio vs Batch Size per GPU", fontsize=16, fontweight="bold")
    # Add grid
    plt.grid(True, alpha=0.3, linestyle="--")
    # Add legend
    plt.legend(fontsize=12, loc="best")
    # Adjust layout
    plt.tight_layout()
    # Save figure
    output_path = os.path.join(current_analysis_result_subdir_path, "afd_non_overlap_time_vs_batch_size.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    # Close figure
    plt.close()


def analyze_l20a_qwen3coder_throughput_advantage(
    seq_len: int,
    num_gpus: int,
    slo_tpot_us: int = 40000,
    performance_results_dir_path: str = r"./test_data",
    l20a_qwen3coder_throughput_advantage_analysis_result_dir_path: str = r"./l20a_qwen3coder_throughput_advantage_analysis_result",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Analyse the throughput advantage of L20A

    Args:
        seq_len (int): Sequence length.
        num_gpus (int): Number of GPUs.
        slo_tpot_us (int, optional): Time per output token in microseconds. Defaults to 40000.
        performance_results_dir_path (str, optional): Path to the performance results directory. Defaults to r"./test_data".
        l20a_qwen3coder_throughput_advantage_analysis_result_dir_path (str, optional): Path to the l20a qwen3coder throughput advantage analysis result directory. Defaults to r"./l20a_qwen3coder_throughput_advantage_analysis_result".

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Tuple containing the filtered reference statistics DataFrame and the filtered AFD best statistics DataFrame.
    """
    # Check if the sequence length is valid
    if seq_len not in VALID_SEQ_LENS:
        raise ValueError(f"Invalid sequence length: {seq_len}. Valid sequence lengths are: {VALID_SEQ_LENS}")
    # Check if the number of GPUs is valid
    if num_gpus not in range(VALID_MIN_NUM_GPUS, VALID_MAX_NUM_GPUS + 1):
        raise ValueError(
            f"Invalid number of GPUs: {num_gpus}. Valid number of GPUs are: {VALID_MIN_NUM_GPUS} to {VALID_MAX_NUM_GPUS}"
        )
    # Check if the number of experts is divisible by the number of GPUs
    if (NUM_EXPERTS + NUM_REDUNDANT_EXPERTS) % num_gpus != 0:
        raise ValueError(
            f"Invalid number of GPUs: {num_gpus}. Number of experts ({NUM_EXPERTS + NUM_REDUNDANT_EXPERTS}) must be divisible by the number of GPUs ({num_gpus})"
        )

    # Load the performance DataFrame from the excel files
    try:
        performance_results = load_performance_results(
            performance_results_dir_path=performance_results_dir_path,
            seq_len=seq_len,
        )
        print(f"Performance results: {performance_results}")
    except Exception as e:
        print(f"Error loading performance results: {e}")
        return None

    # Analyze reference mode
    reference_statistics_df = calculate_reference_statistics(
        performance_results=performance_results,
        num_gpus=num_gpus,
    )

    # Analyze AFD best model
    afd_statistics_df = calculate_afd_statistics(
        performance_results=performance_results,
        num_gpus=num_gpus,
    )

    # Make sure the analysis result directory exists
    if not os.path.exists(l20a_qwen3coder_throughput_advantage_analysis_result_dir_path):
        os.makedirs(l20a_qwen3coder_throughput_advantage_analysis_result_dir_path, exist_ok=True)
    # Create the current analysis result subdirectory
    current_analysis_result_subdir_path = os.path.join(
        l20a_qwen3coder_throughput_advantage_analysis_result_dir_path,
        f"seq_len-{seq_len}_num_gpus-{num_gpus}_slo_tpot_us-{slo_tpot_us}",
    )
    # Remove the current analysis result subdirectory if it exists
    if os.path.exists(current_analysis_result_subdir_path):
        shutil.rmtree(current_analysis_result_subdir_path)
    os.makedirs(current_analysis_result_subdir_path, exist_ok=True)

    # Save the reference and AFD statistics to the csv files
    save_reference_and_afd_statistics_to_csv(
        reference_statistics_df=reference_statistics_df,
        afd_statistics_df=afd_statistics_df,
        current_analysis_result_subdir_path=current_analysis_result_subdir_path,
    )

    # Group by Batch Size per GPU and select the row with the maximum Throughput / tokens/gpu/s
    afd_best_statistics_df = afd_statistics_df.copy()
    afd_best_statistics_df: pd.DataFrame = afd_best_statistics_df.loc[
        afd_best_statistics_df.groupby("Batch Size per GPU")["Throughput / tokens/gpu/s"].idxmax()
    ]

    # Filter the common batch sizes
    filtered_reference_statistics_df, filtered_afd_best_statistics_df = filter_common_batch_sizes(
        reference_statistics_df=reference_statistics_df.copy(),
        afd_best_statistics_df=afd_best_statistics_df.copy(),
    )

    # Plot the reference and AFD statistics
    plot_throughput_comparison_vs_tpot(
        reference_statistics_df=reference_statistics_df,
        afd_best_statistics_df=afd_best_statistics_df,
        current_analysis_result_subdir_path=current_analysis_result_subdir_path,
        slo_tpot_us=slo_tpot_us,
        max_points=50,
    )

    # Plot TPOT comparison vs batch size
    plot_tpot_comparison_vs_batch_size(
        reference_statistics_df=reference_statistics_df,
        afd_best_statistics_df=afd_best_statistics_df,
        current_analysis_result_subdir_path=current_analysis_result_subdir_path,
        max_points=50,
    )

    # Plot proportion change vs batch size
    plot_proportion_change_vs_batch_size(
        filtered_reference_statistics_df=filtered_reference_statistics_df,
        filtered_afd_best_statistics_df=filtered_afd_best_statistics_df,
        current_analysis_result_subdir_path=current_analysis_result_subdir_path,
        max_points=50,
    )

    # Plot MOE FFN metrics vs batch size
    plot_moe_ffn_metrics_vs_batch_size(
        filtered_reference_statistics_df=filtered_reference_statistics_df,
        filtered_afd_best_statistics_df=filtered_afd_best_statistics_df,
        current_analysis_result_subdir_path=current_analysis_result_subdir_path,
        max_points=50,
    )

    # Plot AFD non-overlap time vs batch size
    plot_afd_non_overlap_time_vs_batch_size(
        afd_best_statistics_df=afd_best_statistics_df,
        current_analysis_result_subdir_path=current_analysis_result_subdir_path,
        max_points=100,
    )
    return filtered_reference_statistics_df, filtered_afd_best_statistics_df


if __name__ == "__main__":
    # analyze_l20a_qwen3coder_throughput_advantage(
    #     seq_len=2048,
    #     num_gpus=8,
    #     slo_tpot_us=40000,
    #     performance_results_dir_path=r"./test_data",
    #     l20a_qwen3coder_throughput_advantage_analysis_result_dir_path=r"./l20a_qwen3coder_throughput_advantage_analysis_result",
    # )

    num_gpus_list = []
    ref_max_throughput_list = []
    afd_max_throughput_list = []
    for num_gpus in range(1, 73):
        try:
            filtered_reference_statistics_df, filtered_afd_best_statistics_df = (
                analyze_l20a_qwen3coder_throughput_advantage(
                    seq_len=2048,
                    num_gpus=num_gpus,
                    slo_tpot_us=40000,
                    performance_results_dir_path=r"./test_data",
                    l20a_qwen3coder_throughput_advantage_analysis_result_dir_path=r"./l20a_qwen3coder_throughput_advantage_analysis_result",
                )
            )
            num_gpus_list.append(num_gpus)
            ref_max_throughput_list.append(
                float(
                    filtered_reference_statistics_df[filtered_reference_statistics_df["TPOT / us"] < 40000][
                        "Throughput / tokens/gpu/s"
                    ].max()
                )
            )
            afd_max_throughput_list.append(
                float(
                    filtered_afd_best_statistics_df[filtered_afd_best_statistics_df["TPOT / us"] < 40000][
                        "Throughput / tokens/gpu/s"
                    ].max()
                )
            )
        except Exception as e:
            continue
    print(f"Number of GPUs: {num_gpus_list}")
    print(f"Reference max throughput: {ref_max_throughput_list}")
    print(f"AFD max throughput: {afd_max_throughput_list}")
