from smolagents import tool
import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
import json

@tool
def load_timeseries_data(file_path: str, datetime_column: str = "timestamp", resample_rule: Optional[str] = None) -> str:
    """
    Load time series data from a file (CSV, Excel, etc.).
    
    Args:
        file_path: Path to the file containing time series data
        datetime_column: Name of the column containing timestamps
        resample_rule: Optional rule for resampling data (e.g., '1h', '15min')
        
    Returns:
        A string describing the loaded data structure and summary
    """
    try:
        # Get file extension
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Load the data based on file type
        if file_ext == '.csv':
            df = pd.read_csv(file_path)
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        elif file_ext == '.json':
            df = pd.read_json(file_path)
        else:
            return f"Unsupported file format: {file_ext}. Please use CSV, Excel, or JSON files."
            
        # Convert datetime column
        df[datetime_column] = pd.to_datetime(df[datetime_column])
        df.set_index(datetime_column, inplace=True)
        
        # Resample if specified
        if resample_rule:
            df = df.resample(resample_rule).mean()
        
        # Generate a dataset name from the file path
        dataset_name = Path(file_path).stem
        
        # Save the DataFrame as a pickle file for later use
        output_path = Path("data") / f"{dataset_name}.pkl"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_pickle(str(output_path))
        
        # Create summary
        summary = {
            "dataset_name": dataset_name,
            "file_path": str(output_path),
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "start_date": df.index.min().strftime("%Y-%m-%d %H:%M:%S"),
            "end_date": df.index.max().strftime("%Y-%m-%d %H:%M:%S"),
            "duration": str(df.index.max() - df.index.min()),
            "sample": df.head(3).to_dict(orient="records")
        }
        
        # Save summary to a JSON file
        summary_path = Path("data") / f"{dataset_name}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, default=str)
        
        return f"Successfully loaded data from {file_path}\n" + \
               f"Dataset name: {dataset_name}\n" + \
               f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n" + \
               f"Time range: {summary['start_date']} to {summary['end_date']}\n" + \
               f"Duration: {summary['duration']}\n" + \
               f"Columns: {', '.join(df.columns.tolist())}\n" + \
               f"Data saved to {output_path}\n" + \
               f"Summary saved to {summary_path}"
               
    except Exception as e:
        return f"Error loading data: {str(e)}"

@tool
def explore_dataset(dataset_name: str, sample_size: int = 5) -> str:
    """
    Explore and summarize a loaded time series dataset.
    
    Args:
        dataset_name: Name of the dataset to explore
        sample_size: Number of samples to show
        
    Returns:
        A string containing exploratory analysis of the dataset
    """
    try:
        # Load the DataFrame
        file_path = Path("data") / f"{dataset_name}.pkl"
        if not file_path.exists():
            return f"Dataset '{dataset_name}' not found. Please load it first using load_timeseries_data."
        
        df = pd.read_pickle(str(file_path))
        
        # Basic statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        stats = df[numeric_cols].describe().to_string()
        
        # Check for missing values
        missing_values = df.isna().sum().to_dict()
        missing_pct = ((df.isna().sum() / len(df)) * 100).round(2).to_dict()
        
        # Check for duplicate timestamps
        duplicate_timestamps = df.index.duplicated().sum()
        
        # Time series specific analysis
        time_stats = {
            "start_date": df.index.min().strftime("%Y-%m-%d %H:%M:%S"),
            "end_date": df.index.max().strftime("%Y-%m-%d %H:%M:%S"),
            "duration": str(df.index.max() - df.index.min()),
            "frequency": pd.infer_freq(df.index),
            "avg_time_delta": str(df.index.to_series().diff().mean()),
            "gaps": (df.index.to_series().diff() > pd.Timedelta(hours=1)).sum()
        }
        
        # Generate data samples
        head_samples = df.head(sample_size).to_string()
        tail_samples = df.tail(sample_size).to_string()
        
        # Create report
        report = f"Dataset: {dataset_name}\n\n"
        report += f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n\n"
        report += f"Time Range: {time_stats['start_date']} to {time_stats['end_date']}\n"
        report += f"Duration: {time_stats['duration']}\n"
        report += f"Avg Time Delta: {time_stats['avg_time_delta']}\n"
        report += f"Time Gaps > 1h: {time_stats['gaps']}\n\n"
        
        report += f"First {sample_size} samples:\n{head_samples}\n\n"
        report += f"Last {sample_size} samples:\n{tail_samples}\n\n"
        
        report += "Missing Values:\n"
        for col, count in missing_values.items():
            if count > 0:
                report += f"  {col}: {count} ({missing_pct[col]}%)\n"
        
        if duplicate_timestamps > 0:
            report += f"\nDuplicate timestamps: {duplicate_timestamps}\n"
            
        report += f"\nStatistics:\n{stats}\n"
        
        # Save the report
        report_path = Path("results") / f"{dataset_name}_exploration.txt"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, 'w') as f:
            f.write(report)
            
        return report
    
    except Exception as e:
        return f"Error exploring dataset: {str(e)}"

@tool
def generate_synthetic_data(
    output_name: str, 
    days: int = 30, 
    frequency: str = "1H", 
    columns: List[str] = ["temperature", "pressure", "humidity"],
    add_anomalies: bool = True
) -> str:
    """
    Generate synthetic time series data for testing.
    
    Args:
        output_name: Name of the output dataset
        days: Number of days of data to generate
        frequency: Frequency of data points (e.g., '1H' for hourly)
        columns: List of column names to generate
        add_anomalies: Whether to add synthetic anomalies
        
    Returns:
        A string describing the generated data
    """
    try:
        # Generate datetime index
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.Timedelta(days=days)
        date_range = pd.date_range(start=start_date, end=end_date, freq=frequency)
        
        # Number of data points
        n_points = len(date_range)
        
        # Create DataFrame
        df = pd.DataFrame(index=date_range)
        
        # Generate synthetic data for each column
        np.random.seed(42)  # For reproducibility
        
        for i, col in enumerate(columns):
            # Base signal with trend and seasonality
            t = np.linspace(0, 4*np.pi, n_points)
            base = 20 + 5 * np.sin(t + i*np.pi/4) + 0.1 * np.arange(n_points)
            # Add noise
            noise = 2 * np.random.randn(n_points)
            df[col] = base + noise
        
        # Add anomalies if specified
        anomaly_positions = []
        if add_anomalies:
            # Choose random positions for anomalies
            num_anomalies = min(10, n_points // 50)
            anomaly_indices = np.random.choice(range(n_points), size=num_anomalies, replace=False)
            
            for idx in anomaly_indices:
                # Add spike anomaly (random column)
                col = np.random.choice(columns)
                df.iloc[idx, df.columns.get_loc(col)] = df.iloc[idx, df.columns.get_loc(col)] + 15 * np.random.choice([-1, 1])
                anomaly_positions.append((idx, col))
                
                # Add contextual anomaly (plateau)
                if idx + 3 < n_points:
                    col = np.random.choice(columns)
                    plateau_val = df.iloc[idx, df.columns.get_loc(col)] * 1.5
                    for i in range(3):  # 3-point plateau
                        if idx + i < n_points:
                            df.iloc[idx + i, df.columns.get_loc(col)] = plateau_val
                            anomaly_positions.append((idx + i, col))
        
        # Save the DataFrame
        output_path = Path("data") / f"{output_name}.pkl"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_pickle(str(output_path))
        
        # Save as CSV as well
        csv_path = Path("data") / f"{output_name}.csv"
        df.to_csv(csv_path)
        
        # Create and save anomaly information
        if add_anomalies:
            anomaly_info = {"anomalies": []}
            for idx, col in anomaly_positions:
                anomaly_info["anomalies"].append({
                    "timestamp": date_range[idx].strftime("%Y-%m-%d %H:%M:%S"),
                    "column": col,
                    "index": int(idx)
                })
            
            anomaly_path = Path("data") / f"{output_name}_anomalies.json"
            with open(anomaly_path, 'w') as f:
                json.dump(anomaly_info, f, indent=2)
        
        return f"Generated synthetic time series with {n_points} points and {len(columns)} columns:\n" + \
               f"Columns: {', '.join(columns)}\n" + \
               f"Time range: {start_date} to {end_date}\n" + \
               f"Frequency: {frequency}\n" + \
               f"Added anomalies: {add_anomalies} ({len(anomaly_positions)} points)\n" + \
               f"Data saved to: {output_path} and {csv_path}" + \
               (f"\nAnomaly info saved to: {anomaly_path}" if add_anomalies else "")
    
    except Exception as e:
        return f"Error generating synthetic data: {str(e)}"