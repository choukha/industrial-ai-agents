# itisa/tools/data_processing_tool.py
from smolagents import tool
import pandas as pd
from pathlib import Path
import os
import json

# Ensure data and results directories exist
DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Define the path for the processed data
PROCESSED_DATA_PATH = DATA_DIR / "current_timeseries_data.pkl"
DATA_SUMMARY_PATH = RESULTS_DIR / "current_data_summary.json"

@tool
def load_and_describe_data(file_path: str, datetime_column: str = "timestamp") -> str:
    """
    Loads time series data from a CSV file, sets the datetime column as index,
    saves the processed DataFrame, and returns a summary.

    Args:
        file_path (str): The path to the CSV file. Assumed to be in the 'data/' directory relative to the project root.
        datetime_column (str): The name of the column containing datetime information. Defaults to "timestamp".

    Returns:
        str: A summary of the loaded data, including where it's saved, its shape, columns, and time range.
             Returns an error message if loading fails.
    """
    try:
        # Construct the full path if a relative path is given (e.g., from Gradio upload)
        # Gradio typically provides an absolute path for uploaded files.
        # If file_path is just a name, assume it's in DATA_DIR
        actual_file_path = Path(file_path)
        if not actual_file_path.is_absolute() and not actual_file_path.exists():
            actual_file_path = DATA_DIR / file_path

        if not actual_file_path.exists():
            return f"Error: File not found at {actual_file_path}. Please ensure the file is in the 'data' directory or provide a full path."

        df = pd.read_csv(actual_file_path)

        if datetime_column not in df.columns:
            return (f"Error: Datetime column '{datetime_column}' not found in the CSV. "
                    f"Available columns are: {', '.join(df.columns)}")

        df[datetime_column] = pd.to_datetime(df[datetime_column])
        df.set_index(datetime_column, inplace=True)
        df.sort_index(inplace=True) # Ensure data is sorted by time

        # Save the processed DataFrame
        df.to_pickle(PROCESSED_DATA_PATH)

        # Generate summary
        summary = {
            "source_file": str(actual_file_path.name),
            "processed_data_path": str(PROCESSED_DATA_PATH),
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "time_range": {
                "start": df.index.min().isoformat() if not df.empty else None,
                "end": df.index.max().isoformat() if not df.empty else None,
            },
            "missing_values_per_column": df.isnull().sum().to_dict(),
            "basic_stats_sample": df.describe().to_dict() if not df.empty else {}
        }

        with open(DATA_SUMMARY_PATH, 'w') as f:
            json.dump(summary, f, indent=4)

        return (f"Successfully loaded and processed data from '{actual_file_path.name}'.\n"
                f"Processed data saved to: {PROCESSED_DATA_PATH}\n"
                f"Data shape: {df.shape[0]} rows, {df.shape[1]} columns.\n"
                f"Time range: {summary['time_range']['start']} to {summary['time_range']['end']}.\n"
                f"Columns: {', '.join(df.columns)}.\n"
                f"A detailed summary has been saved to: {DATA_SUMMARY_PATH}")

    except Exception as e:
        return f"Error loading or processing data: {str(e)}"

