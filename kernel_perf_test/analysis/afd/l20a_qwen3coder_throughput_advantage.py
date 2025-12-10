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
) -> float:
    """Calculate the TPOT / us for the reference model"""
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
    return reference_tpot_us


def calculate_afd_tpot_us(
    performance_results: Dict[str, pd.DataFrame],
    micro_batch_size_list: List[int],
    num_attn_gpus: int,
    num_ffn_gpus: int,
) -> Tuple[float, Tuple[float, ...], Tuple[float, ...]]:
    """Calculate the TPOT / us for the AFD best model"""
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
    return (
        float(afd_tpot_us),
        tuple([float(x) for x in attn_time_per_micro_batch_list]),
        tuple([float(x) for x in ffn_time_per_micro_batch_list]),
    )


def calculate_reference_mode_statistics(
    performance_results: Dict[str, pd.DataFrame],
    num_gpus: int,
) -> pd.DataFrame:
    """Calculate the statistics for the reference mode and return the statistics as a DataFrame"""
    reference_batch_size_per_gpu_list = []
    reference_tpot_us_list = []  # TPOT / us
    reference_throughput_list = []  # tokens_per_gpu(batch_size_per_gpu) / s
    # Iterate over the batch sizes
    for batch_size_per_gpu in range(1, VALID_MAX_BATCH_SIZE + 1):
        try:
            # Calculate reference TPOT / us
            reference_tpot_us = calculate_reference_tpot_us(
                performance_results=performance_results,
                batch_size_per_gpu=batch_size_per_gpu,
                num_gpus=num_gpus,
            )
            # Calculate the throughput tokens per GPU per second
            throughput_tokens_per_gpu_per_second = batch_size_per_gpu * 1e6 / reference_tpot_us
            # Save the reference TPOT / us and throughput tokens per GPU per second
            reference_batch_size_per_gpu_list.append(batch_size_per_gpu)
            reference_tpot_us_list.append(reference_tpot_us)
            reference_throughput_list.append(throughput_tokens_per_gpu_per_second)
        except Exception:
            continue
    # Sort the reference TPOT / us and throughput tokens per GPU per second
    sorted_reference_batch_size_per_gpu_list, sorted_reference_tpot_us_list, sorted_reference_throughput_list = zip(
        *sorted(
            zip(reference_batch_size_per_gpu_list, reference_tpot_us_list, reference_throughput_list),
            key=lambda x: x[1],
        )
    )
    # Convert the statistics to a DataFrame
    reference_statistics_df = pd.DataFrame(
        zip(
            sorted_reference_batch_size_per_gpu_list,
            sorted_reference_tpot_us_list,
            sorted_reference_throughput_list,
        ),
        columns=["Batch Size per GPU", "TPOT / us", "Throughput / tokens/gpu/s"],
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
                    afd_tpot_us_list.append(afd_tpot_us)
                    afd_throughput_list.append(throughput_tokens_per_gpu_per_second)
                except Exception:
                    continue
    # Sort the AFD statistics
    (
        sorted_afd_batch_size_per_gpu_list,
        sorted_afd_num_attn_gpus_list,
        sorted_afd_num_ffn_gpus_list,
        sorted_afd_num_micro_batches_list,
        sorted_afd_micro_batch_size_list_list,
        sorted_afd_attn_time_per_micro_batch_list_list,
        sorted_afd_ffn_time_per_micro_batch_list_list,
        sorted_afd_tpot_us_list,
        sorted_afd_throughput_list,
    ) = zip(
        *sorted(
            zip(
                afd_batch_size_per_gpu_list,
                afd_num_attn_gpus_list,
                afd_num_ffn_gpus_list,
                afd_num_micro_batches_list,
                afd_micro_batch_size_list_list,
                afd_attn_time_per_micro_batch_list_list,
                afd_ffn_time_per_micro_batch_list_list,
                afd_tpot_us_list,
                afd_throughput_list,
            ),
            key=lambda x: x[-2],
        )
    )
    # Convert the statistics to a DataFrame
    afd_statistics_df = pd.DataFrame(
        zip(
            sorted_afd_batch_size_per_gpu_list,
            sorted_afd_num_attn_gpus_list,
            sorted_afd_num_ffn_gpus_list,
            sorted_afd_num_micro_batches_list,
            sorted_afd_micro_batch_size_list_list,
            sorted_afd_attn_time_per_micro_batch_list_list,
            sorted_afd_ffn_time_per_micro_batch_list_list,
            sorted_afd_tpot_us_list,
            sorted_afd_throughput_list,
        ),
        columns=[
            "Batch Size per GPU",
            "Number of Attention GPUs",
            "Number of FFN GPUs",
            "Number of Micro Batches",
            "Micro Batch Size List",
            "Attention Time per Micro Batch",
            "FFN Time per Micro Batch",
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


def plot_reference_and_afd_statistics(
    reference_statistics_df: pd.DataFrame,
    afd_statistics_df: pd.DataFrame,
    current_analysis_result_subdir_path: str,
    slo_tpot_us: int,
    max_points: int = 50,
) -> None:
    """Plot the reference and AFD statistics

    Args:
        reference_statistics_df: Reference statistics DataFrame
        afd_statistics_df: AFD statistics DataFrame
        current_analysis_result_subdir_path: Path to save the plot
        slo_tpot_us: SLO TPOT in microseconds
        max_points: Maximum number of points to plot for each curve (default: 50)
    """
    # Copy the reference and AFD statistics DataFrame
    copied_reference_statistics_df = reference_statistics_df.copy()
    copied_afd_statistics_df = afd_statistics_df.copy()
    # Group by Batch Size per GPU and select the row with the maximum Throughput / tokens/gpu/s
    copied_afd_statistics_df: pd.DataFrame = copied_afd_statistics_df.loc[
        copied_afd_statistics_df.groupby("Batch Size per GPU")["Throughput / tokens/gpu/s"].idxmax()
    ]
    # Sort by TPOT / us for better line plotting
    copied_reference_statistics_df = copied_reference_statistics_df.sort_values("TPOT / us")
    copied_afd_statistics_df = copied_afd_statistics_df.sort_values("TPOT / us")

    # Sample data points if there are too many
    def sample_dataframe(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
        """Uniformly sample dataframe to reduce number of points"""
        if len(df) <= max_points:
            return df
        # Calculate step size for uniform sampling
        indices = [int(i * (len(df) - 1) / (max_points - 1)) for i in range(max_points)]
        return df.iloc[indices].copy()

    # Apply sampling
    if len(copied_reference_statistics_df) > max_points:
        copied_reference_statistics_df = sample_dataframe(copied_reference_statistics_df, max_points)
    if len(copied_afd_statistics_df) > max_points:
        copied_afd_statistics_df = sample_dataframe(copied_afd_statistics_df, max_points)
    # Create figure and axis
    plt.figure(figsize=(16, 10))
    # Plot reference statistics
    plt.plot(
        copied_reference_statistics_df["TPOT / us"],
        copied_reference_statistics_df["Throughput / tokens/gpu/s"],
        marker="o",
        markersize=6,
        linewidth=2,
        label="Reference",
        color="blue",
    )
    # Plot AFD statistics
    plt.plot(
        copied_afd_statistics_df["TPOT / us"],
        copied_afd_statistics_df["Throughput / tokens/gpu/s"],
        marker="s",
        markersize=6,
        linewidth=2,
        label="AFD",
        color="red",
    )
    # Add annotations for AFD data points with alternating positions to avoid overlap
    annotation_positions = [(10, 10), (10, -20), (-80, 10), (-80, -20)]  # Alternating positions
    for idx, (_, row) in enumerate(copied_afd_statistics_df.iterrows()):
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
    output_path = os.path.join(current_analysis_result_subdir_path, "throughput_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def analyze_l20a_qwen3coder_throughput_advantage(
    seq_len: int,
    num_gpus: int,
    slo_tpot_us: int = 40000,
    performance_results_dir_path: str = r"./test_data",
    l20a_qwen3coder_throughput_advantage_analysis_result_dir_path: str = r"./l20a_qwen3coder_throughput_advantage_analysis_result",
) -> None:
    """Analyze the throughput advantage of L20A

    Args:
        seq_len (int): Sequence length
        num_gpus (int): Number of GPUs
        slo_tpot_us (int, optional): Time per output token in microseconds. Defaults to 40000.
        performance_results_dir_path (str, optional): Path to the performance results directory. Defaults to r"./test_data".
        l20a_qwen3coder_throughput_advantage_analysis_result_dir_path (str, optional): Path to the l20a qwen3coder throughput advantage analysis result directory. Defaults to r"./l20a_qwen3coder_throughput_advantage_analysis_result".

    """
    # Calculate the time per layer
    time_us_per_layer = slo_tpot_us / NUM_LAYERS
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
        return

    # Analyze reference mode
    reference_statistics_df = calculate_reference_mode_statistics(
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

    # Plot the reference and AFD statistics
    plot_reference_and_afd_statistics(
        reference_statistics_df=reference_statistics_df,
        afd_statistics_df=afd_statistics_df,
        current_analysis_result_subdir_path=current_analysis_result_subdir_path,
        slo_tpot_us=slo_tpot_us,
    )


if __name__ == "__main__":
    analyze_l20a_qwen3coder_throughput_advantage(
        seq_len=2048,
        num_gpus=48,
        slo_tpot_us=40000,
        performance_results_dir_path=r"./test_data",
        l20a_qwen3coder_throughput_advantage_analysis_result_dir_path=r"./l20a_qwen3coder_throughput_advantage_analysis_result",
    )
