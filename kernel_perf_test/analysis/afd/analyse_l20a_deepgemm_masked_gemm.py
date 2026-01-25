import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from kernel_perf_test import utils
from kernel_perf_test.analysis.utils import calculate_deepgemm_masked_gemm_tflops


def plot_tflops_vs_expected_m(
    deepgemm_masked_gemm_performance_df: pd.DataFrame,
    current_analysis_result_dir_path: str,
    max_series: int = 10,
) -> None:
    """Plot TFLOPS vs expected_m

    Args:
        deepgemm_masked_gemm_performance_df: DataFrame with expected_m as index and num_local_experts as columns
        current_analysis_result_dir_path: Path to save the plot
        max_series: Maximum number of series (num_local_experts) to plot
    """
    # Extract num_local_experts values from column names
    num_local_experts_list = []
    for col in deepgemm_masked_gemm_performance_df.columns:
        if col.startswith("num_local_experts: "):
            num_local_experts = int(col.split(": ")[1])
            num_local_experts_list.append((col, num_local_experts))

    # Sort by num_local_experts value
    num_local_experts_list.sort(key=lambda x: x[1])

    # Uniform sampling of series
    if len(num_local_experts_list) > max_series:
        indices = np.linspace(0, len(num_local_experts_list) - 1, max_series, dtype=int)
        sampled_num_local_experts_list = [num_local_experts_list[i] for i in indices]
    else:
        sampled_num_local_experts_list = num_local_experts_list

    # Create figure
    plt.figure(figsize=(16, 10))

    # Get sorted expected_m values
    sorted_expected_m = sorted(deepgemm_masked_gemm_performance_df.index.tolist())

    # Plot each series
    for col_name, num_local_experts in sampled_num_local_experts_list:
        expected_m_values = []
        tflops_values = []

        for expected_m in sorted_expected_m:
            latency_us = deepgemm_masked_gemm_performance_df.loc[expected_m, col_name]
            if pd.notna(latency_us) and latency_us > 0:
                tflops = calculate_deepgemm_masked_gemm_tflops(
                    E=num_local_experts, M=expected_m, N=5120, K=6144, latency_us=latency_us
                )
                expected_m_values.append(expected_m)
                tflops_values.append(tflops)

        if expected_m_values:
            plt.plot(
                expected_m_values,
                tflops_values,
                marker="o",
                markersize=6,
                linewidth=2,
                label=f"num_local_experts: {num_local_experts}",
            )

    # Set labels and title
    plt.xlabel("expected_m", fontsize=14, fontweight="bold")
    plt.ylabel("TFLOPS", fontsize=14, fontweight="bold")
    plt.title("TFLOPS vs expected_m", fontsize=16, fontweight="bold")

    # Add grid
    plt.grid(True, alpha=0.3, linestyle="--")

    # Add legend
    plt.legend(fontsize=12, loc="best")

    # Adjust layout
    plt.tight_layout()

    # Save figure
    output_path = os.path.join(current_analysis_result_dir_path, "tflops_vs_expected_m.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_tflops_vs_num_local_experts(
    deepgemm_masked_gemm_performance_df: pd.DataFrame,
    current_analysis_result_dir_path: str,
) -> None:
    """Plot TFLOPS vs num_local_experts

    Args:
        deepgemm_masked_gemm_performance_df: DataFrame with expected_m as index and num_local_experts as columns
        current_analysis_result_dir_path: Path to save the plot
    """
    # Extract num_local_experts values from column names
    num_local_experts_list = []
    for col in deepgemm_masked_gemm_performance_df.columns:
        if col.startswith("num_local_experts: "):
            num_local_experts = int(col.split(": ")[1])
            num_local_experts_list.append((col, num_local_experts))

    # Sort by num_local_experts value
    num_local_experts_list.sort(key=lambda x: x[1])

    # Get all expected_m values and sort them
    expected_m_values_list = sorted(deepgemm_masked_gemm_performance_df.index.tolist())

    # Uniform sampling of expected_m series (use max_series=10 as default)
    max_series = 10
    if len(expected_m_values_list) > max_series:
        indices = np.linspace(0, len(expected_m_values_list) - 1, max_series, dtype=int)
        sampled_expected_m_list = [expected_m_values_list[i] for i in indices]
    else:
        sampled_expected_m_list = expected_m_values_list

    # Create figure
    plt.figure(figsize=(16, 10))

    # Plot each series
    for expected_m in sampled_expected_m_list:
        num_local_experts_values = []
        tflops_values = []

        for col_name, num_local_experts in num_local_experts_list:
            latency_us = deepgemm_masked_gemm_performance_df.loc[expected_m, col_name]
            if pd.notna(latency_us) and latency_us > 0:
                tflops = calculate_deepgemm_masked_gemm_tflops(
                    E=num_local_experts, M=expected_m, N=5120, K=6144, latency_us=latency_us
                )
                num_local_experts_values.append(num_local_experts)
                tflops_values.append(tflops)

        if num_local_experts_values:
            plt.plot(
                num_local_experts_values,
                tflops_values,
                marker="o",
                markersize=6,
                linewidth=2,
                label=f"expected_m: {expected_m}",
            )

    # Set labels and title
    plt.xlabel("num_local_experts", fontsize=14, fontweight="bold")
    plt.ylabel("TFLOPS", fontsize=14, fontweight="bold")
    plt.title("TFLOPS vs num_local_experts", fontsize=16, fontweight="bold")

    # Add grid
    plt.grid(True, alpha=0.3, linestyle="--")

    # Add legend
    plt.legend(fontsize=12, loc="best")

    # Adjust layout
    plt.tight_layout()

    # Save figure
    output_path = os.path.join(current_analysis_result_dir_path, "tflops_vs_num_local_experts.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_latency_vs_expected_m(
    deepgemm_masked_gemm_performance_df: pd.DataFrame,
    current_analysis_result_dir_path: str,
    max_series: int = 10,
) -> None:
    """Plot latency vs expected_m

    Args:
        deepgemm_masked_gemm_performance_df: DataFrame with expected_m as index and num_local_experts as columns
        current_analysis_result_dir_path: Path to save the plot
        max_series: Maximum number of series (num_local_experts) to plot
    """
    # Extract num_local_experts values from column names
    num_local_experts_list = []
    for col in deepgemm_masked_gemm_performance_df.columns:
        if col.startswith("num_local_experts: "):
            num_local_experts = int(col.split(": ")[1])
            num_local_experts_list.append((col, num_local_experts))

    # Sort by num_local_experts value
    num_local_experts_list.sort(key=lambda x: x[1])

    # Uniform sampling of series
    if len(num_local_experts_list) > max_series:
        indices = np.linspace(0, len(num_local_experts_list) - 1, max_series, dtype=int)
        sampled_num_local_experts_list = [num_local_experts_list[i] for i in indices]
    else:
        sampled_num_local_experts_list = num_local_experts_list

    # Create figure
    plt.figure(figsize=(16, 10))

    # Get sorted expected_m values
    sorted_expected_m = sorted(deepgemm_masked_gemm_performance_df.index.tolist())

    # Plot each series
    for col_name, num_local_experts in sampled_num_local_experts_list:
        expected_m_values = []
        latency_values = []

        for expected_m in sorted_expected_m:
            latency_us = deepgemm_masked_gemm_performance_df.loc[expected_m, col_name]
            if pd.notna(latency_us) and latency_us > 0:
                expected_m_values.append(expected_m)
                latency_values.append(latency_us)

        if expected_m_values:
            plt.plot(
                expected_m_values,
                latency_values,
                marker="o",
                markersize=6,
                linewidth=2,
                label=f"num_local_experts: {num_local_experts}",
            )

    # Set labels and title
    plt.xlabel("expected_m", fontsize=14, fontweight="bold")
    plt.ylabel("Latency / us", fontsize=14, fontweight="bold")
    plt.title("Latency vs expected_m", fontsize=16, fontweight="bold")

    # Add grid
    plt.grid(True, alpha=0.3, linestyle="--")

    # Add legend
    plt.legend(fontsize=12, loc="best")

    # Adjust layout
    plt.tight_layout()

    # Save figure
    output_path = os.path.join(current_analysis_result_dir_path, "latency_vs_expected_m.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_latency_vs_num_local_experts(
    deepgemm_masked_gemm_performance_df: pd.DataFrame,
    current_analysis_result_dir_path: str,
) -> None:
    """Plot latency vs num_local_experts

    Args:
        deepgemm_masked_gemm_performance_df: DataFrame with expected_m as index and num_local_experts as columns
        current_analysis_result_dir_path: Path to save the plot
    """
    # Extract num_local_experts values from column names
    num_local_experts_list = []
    for col in deepgemm_masked_gemm_performance_df.columns:
        if col.startswith("num_local_experts: "):
            num_local_experts = int(col.split(": ")[1])
            num_local_experts_list.append((col, num_local_experts))

    # Sort by num_local_experts value
    num_local_experts_list.sort(key=lambda x: x[1])

    # Get all expected_m values and sort them
    expected_m_values_list = sorted(deepgemm_masked_gemm_performance_df.index.tolist())

    # Uniform sampling of expected_m series (use max_series=10 as default)
    max_series = 10
    if len(expected_m_values_list) > max_series:
        indices = np.linspace(0, len(expected_m_values_list) - 1, max_series, dtype=int)
        sampled_expected_m_list = [expected_m_values_list[i] for i in indices]
    else:
        sampled_expected_m_list = expected_m_values_list

    # Create figure
    plt.figure(figsize=(16, 10))

    # Plot each series
    for expected_m in sampled_expected_m_list:
        num_local_experts_values = []
        latency_values = []

        for col_name, num_local_experts in num_local_experts_list:
            latency_us = deepgemm_masked_gemm_performance_df.loc[expected_m, col_name]
            if pd.notna(latency_us) and latency_us > 0:
                num_local_experts_values.append(num_local_experts)
                latency_values.append(latency_us)

        if num_local_experts_values:
            plt.plot(
                num_local_experts_values,
                latency_values,
                marker="o",
                markersize=6,
                linewidth=2,
                label=f"expected_m: {expected_m}",
            )

    # Set labels and title
    plt.xlabel("num_local_experts", fontsize=14, fontweight="bold")
    plt.ylabel("Latency / us", fontsize=14, fontweight="bold")
    plt.title("Latency vs num_local_experts", fontsize=16, fontweight="bold")

    # Add grid
    plt.grid(True, alpha=0.3, linestyle="--")

    # Add legend
    plt.legend(fontsize=12, loc="best")

    # Adjust layout
    plt.tight_layout()

    # Save figure
    output_path = os.path.join(current_analysis_result_dir_path, "latency_vs_num_local_experts.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def analyse_l20a_deepgemm_masked_gemm(
    performance_results_dir_path: str,
    l20a_deepgemm_masked_gemm_analysis_result_dir_path: str,
) -> None:
    """Analyse the performance of L20A DeepGEMM Masked Gemm

    Args:
        performance_results_dir_path (str): Path to the performance results directory.
        l20a_deepgemm_masked_gemm_analysis_result_dir_path (str): Path to the l20a deepgemm masked gemm analysis result directory.
    """
    # Load DeepGEMM Grouped Gemm performance results
    try:
        deepgemm_masked_gemm_performance_df = utils.load_performance_results_from_excel(
            file_path=os.path.join(performance_results_dir_path, r"deepgemm_masked_gemm_performance.xlsx"),
            sheet_name="latency",
            index_col="expected_m",
        )
    except Exception as e:
        print(f"Error loading DeepGEMM Masked Gemm performance results: {e}")
        return None
    # Make sure the analysis result directory exists
    if os.path.exists(l20a_deepgemm_masked_gemm_analysis_result_dir_path):
        shutil.rmtree(l20a_deepgemm_masked_gemm_analysis_result_dir_path)
    os.makedirs(l20a_deepgemm_masked_gemm_analysis_result_dir_path, exist_ok=True)
    # Plot TFLOPS vs expected_m
    plot_tflops_vs_expected_m(
        deepgemm_masked_gemm_performance_df=deepgemm_masked_gemm_performance_df,
        current_analysis_result_dir_path=l20a_deepgemm_masked_gemm_analysis_result_dir_path,
    )
    # Plot latency vs expected_m
    plot_latency_vs_expected_m(
        deepgemm_masked_gemm_performance_df=deepgemm_masked_gemm_performance_df,
        current_analysis_result_dir_path=l20a_deepgemm_masked_gemm_analysis_result_dir_path,
    )
    # Plot TFLOPS vs num_local_experts
    plot_tflops_vs_num_local_experts(
        deepgemm_masked_gemm_performance_df=deepgemm_masked_gemm_performance_df,
        current_analysis_result_dir_path=l20a_deepgemm_masked_gemm_analysis_result_dir_path,
    )
    # Plot latency vs num_local_experts
    plot_latency_vs_num_local_experts(
        deepgemm_masked_gemm_performance_df=deepgemm_masked_gemm_performance_df,
        current_analysis_result_dir_path=l20a_deepgemm_masked_gemm_analysis_result_dir_path,
    )


if __name__ == "__main__":
    analyse_l20a_deepgemm_masked_gemm(
        performance_results_dir_path=r"./test_data",
        l20a_deepgemm_masked_gemm_analysis_result_dir_path=r"./l20a_deepgemm_masked_gemm_analysis_result",
    )
