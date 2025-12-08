import os
import pandas as pd
from types import SimpleNamespace
from typing import Dict, Optional, Union


def _dict_to_dataframe(data: dict, index_key: str) -> pd.DataFrame:
    if not data:
        return pd.DataFrame()
    if index_key not in data:
        raise ValueError(f"Each performance dictionary must contain a '{index_key}' key.")
    ordered_columns = [index_key] + [col for col in data.keys() if col != index_key]
    frame = pd.DataFrame({col: data.get(col, []) for col in ordered_columns})
    return frame.reindex(columns=ordered_columns)


def save_performance_results_to_excel(
    save_dir_path: str,
    file_name: str,
    index_key: str,
    **results_dicts: Dict[str, dict],
):
    """Save performance data to excel file

    Args:
        save_dir_path (str): Save directory path
        file_name (str): File name
        index_key (str): Index key
        **results_dicts (Dict[str, dict]): Performance data
            - sheet_name (str): Sheet name
            - data (dict): Performance data
                - key (str): Key
                - value (list): Value
    """
    # Create save subdir path
    save_results_file_path = os.path.join(save_dir_path, f"{file_name}.xlsx")

    # Save performance data to excel file
    with pd.ExcelWriter(save_results_file_path, engine="openpyxl") as writer:
        for sheet_name, data in results_dicts.items():
            df = _dict_to_dataframe(data, index_key)
            df.to_excel(writer, sheet_name=sheet_name, index=False)


def load_performance_results_from_excel(
    file_path: str,
    sheet_name: Optional[Union[str, int]] = None,
    index_col: Optional[str] = None,
) -> pd.DataFrame:
    """Load performance data from Excel file to pandas.DataFrame

    Args:
        file_path (str): Excel file path
        sheet_name (Optional[Union[str, int]]): The sheet name or index to read.
            Default is None (read the first sheet)
        index_col (Optional[str]): The column name to use as the index.
            Default is None (use the first column as the index)

    Returns:
        pd.DataFrame: DataFrame containing performance data
    """
    # If sheet_name is not specified, default to reading the first sheet
    if sheet_name is None:
        sheet_name = 0

    # Read Excel file, first row as column names (header=0)
    df = pd.read_excel(
        file_path,
        sheet_name=sheet_name,
        header=0,  # First row as column names
    )

    # If index_col is not specified, use the first column as the index
    if index_col is None:
        # Get the column name of the first column (the name of the first cell in the first row)
        if len(df.columns) > 0:
            index_col = df.columns[0]
            df.set_index(index_col, inplace=True)
    else:
        # Use the specified column as the index
        if index_col not in df.columns:
            raise ValueError(f"The specified index column '{index_col}' does not exist in the Excel file")
        df.set_index(index_col, inplace=True)

    return df


if __name__ == "__main__":
    # Load performance results from excel file
    df = load_performance_results_from_excel(
        file_path="./test_data/deepgemm_masked_moe_ffn_performance.xlsx",
        sheet_name="latency",
        index_col="expected_m",
    )
    print(df)
    df = load_performance_results_from_excel(
        file_path="./test_data/fused_add_rmsnorm_performance.xlsx",
        sheet_name="latency",
        index_col="batch_size",
    )
    print(df)
    df = load_performance_results_from_excel(
        file_path="./test_data/qwen3_moe_attention_layer_performance.xlsx",
        sheet_name="latency",
        index_col="batch_size",
    )
    print(df)
