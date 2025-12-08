import os
import pandas as pd
from types import SimpleNamespace
from typing import Dict


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
