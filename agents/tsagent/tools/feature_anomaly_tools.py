from smolagents import tool
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import json
from typing import List, Dict, Optional, Union
import warnings

@tool
def extract_statistical_features(
    dataset_name: str,
    columns: Optional[List[str]] = None,
    window_size: int = 24
) -> str:
    """
    Extract statistical features from time series data.
    
    Args:
        dataset_name: Name of the dataset to analyze
        columns: List of columns to analyze (default: all numeric columns)
        window_size: Size of rolling window for feature extraction
        
    Returns:
        String describing the extracted features
    """
    try:
        # Load the DataFrame
        file_path = Path("data") / f"{dataset_name}.pkl"
        if not file_path.exists():
            return f"Dataset '{dataset_name}' not found. Please load it first using load_timeseries_data."
        
        df = pd.read_pickle(str(file_path))
        
        # Select columns to analyze
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            # Check if columns exist
            for col in columns:
                if col not in df.columns:
                    return f"Column '{col}' not found in dataset '{dataset_name}'"
        
        # Initialize results dictionary
        feature_results = {
            "dataset_name": dataset_name,
            "columns": columns,
            "window_size": window_size,
            "results": {}
        }
        
        # Extract features for each column
        for col in columns:
            series = df[col].dropna()
            
            if len(series) == 0:
                feature_results["results"][col] = {"error": "No valid data points in series"}
                continue
            
            # Basic statistics
            global_stats = {
                "mean": float(series.mean()),
                "median": float(series.median()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
                "range": float(series.max() - series.min()),
                "q1": float(series.quantile(0.25)),
                "q3": float(series.quantile(0.75)),
                "iqr": float(series.quantile(0.75) - series.quantile(0.25))
            }
            
            # Compute skewness and kurtosis if possible
            if len(series) > 2:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    global_stats["skewness"] = float(series.skew())
                    global_stats["kurtosis"] = float(series.kurtosis())
            
            # Rolling statistics
            rolling_mean = series.rolling(window=window_size).mean()
            rolling_std = series.rolling(window=window_size).std()
            
            rolling_stats = {
                "mean_std": float(rolling_mean.std()),
                "std_std": float(rolling_std.std() if len(rolling_std.dropna()) > 0 else 0),
                "max_mean": float(rolling_mean.max()),
                "min_mean": float(rolling_mean.min()),
                "max_std": float(rolling_std.max() if len(rolling_std.dropna()) > 0 else 0),
                "min_std": float(rolling_std.min() if len(rolling_std.dropna()) > 0 else 0)
            }
            
            # Trend analysis
            try:
                from scipy import stats as scipy_stats
                
                # Linear trend
                x = np.arange(len(series))
                mask = ~np.isnan(series)
                if np.sum(mask) > 1:
                    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x[mask], series[mask])
                    
                    trend_stats = {
                        "slope": float(slope),
                        "intercept": float(intercept),
                        "r_squared": float(r_value**2),
                        "p_value": float(p_value),
                        "std_err": float(std_err),
                        "is_significant": bool(p_value < 0.05),
                        "trend_direction": "upward" if slope > 0 else "downward",
                        "trend_strength": "strong" if r_value**2 > 0.7 else "moderate" if r_value**2 > 0.3 else "weak"
                    }
                else:
                    trend_stats = {"error": "Not enough valid data points for trend analysis"}
            except ImportError:
                trend_stats = {"error": "scipy not available for trend analysis"}
            
            # Autocorrelation analysis
            try:
                from statsmodels.tsa.stattools import acf, pacf
                
                # Calculate ACF and PACF
                max_lag = min(40, len(series) // 4)
                if max_lag > 0:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        acf_values = acf(series.dropna(), nlags=max_lag, fft=True)
                        pacf_values = pacf(series.dropna(), nlags=max_lag)
                    
                    # Find significant lags
                    confidence_level = 1.96 / np.sqrt(len(series.dropna()))
                    significant_acf = [i for i, val in enumerate(acf_values[1:], 1) if abs(val) > confidence_level]
                    significant_pacf = [i for i, val in enumerate(pacf_values[1:], 1) if abs(val) > confidence_level]
                    
                    autocorr_stats = {
                        "max_acf": float(max(abs(acf_values[1:])) if len(acf_values) > 1 else 0),
                        "max_pacf": float(max(abs(pacf_values[1:])) if len(pacf_values) > 1 else 0),
                        "significant_acf_lags": significant_acf[:5],  # Top 5 significant lags
                        "significant_pacf_lags": significant_pacf[:5],  # Top 5 significant lags
                        "has_autocorrelation": bool(len(significant_acf) > 0)
                    }
                else:
                    autocorr_stats = {"error": "Not enough data points for autocorrelation analysis"}
            except ImportError:
                autocorr_stats = {"error": "statsmodels not available for autocorrelation analysis"}
            
            # Combine all stats
            feature_results["results"][col] = {
                "global_stats": global_stats,
                "rolling_stats": rolling_stats,
                "trend": trend_stats,
                "autocorrelation": autocorr_stats
            }
        
        # Save results to file
        os.makedirs("results", exist_ok=True)
        result_path = f"results/{dataset_name}_features.json"
        with open(result_path, 'w') as f:
            json.dump(feature_results, f, indent=2)
        
        # Generate summary
        summary = f"Statistical features extracted for {len(columns)} columns of dataset '{dataset_name}'.\n"
        summary += f"Results saved to: {result_path}\n\n"
        
        # Add feature highlights for each column
        for col in columns:
            summary += f"Column: {col}\n"
            summary += "  " + "-" * 40 + "\n"
            
            # Check for errors
            if "error" in feature_results["results"][col]:
                summary += f"  Error: {feature_results['results'][col]['error']}\n"
                continue
            
            # Basic statistics
            global_stats = feature_results["results"][col]["global_stats"]
            summary += f"  Basic statistics: mean={global_stats['mean']:.2f}, "
            summary += f"std={global_stats['std']:.2f}, min={global_stats['min']:.2f}, "
            summary += f"max={global_stats['max']:.2f}, median={global_stats['median']:.2f}\n"
            
            # Trend
            trend = feature_results["results"][col]["trend"]
            if "error" not in trend:
                if trend["is_significant"]:
                    summary += f"  Trend: {trend['trend_strength']} {trend['trend_direction']} "
                    summary += f"(slope={trend['slope']:.4f}, r²={trend['r_squared']:.2f})\n"
                else:
                    summary += "  Trend: no significant trend detected\n"
            
            # Autocorrelation
            autocorr = feature_results["results"][col]["autocorrelation"]
            if "error" not in autocorr:
                if autocorr["has_autocorrelation"]:
                    summary += f"  Autocorrelation: detected at lags {autocorr['significant_acf_lags']}\n"
                else:
                    summary += "  Autocorrelation: no significant autocorrelation detected\n"
            
            summary += "\n"
        
        return summary
    
    except Exception as e:
        return f"Error extracting statistical features: {str(e)}"

@tool
def detect_anomalies_zscore(
    dataset_name: str,
    columns: Optional[List[str]] = None,
    window_size: int = 10,
    threshold: float = 3.0
) -> str:
    """
    Detect anomalies using Z-score method.
    
    Args:
        dataset_name: Name of the dataset to analyze
        columns: List of columns to analyze (default: all numeric columns)
        window_size: Size of rolling window for Z-score calculation
        threshold: Z-score threshold for anomaly detection
        
    Returns:
        String describing the detected anomalies
    """
    try:
        # Load the DataFrame
        file_path = Path("data") / f"{dataset_name}.pkl"
        if not file_path.exists():
            return f"Dataset '{dataset_name}' not found. Please load it first using load_timeseries_data."
        
        df = pd.read_pickle(str(file_path))
        
        # Select columns to analyze
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            # Check if columns exist
            for col in columns:
                if col not in df.columns:
                    return f"Column '{col}' not found in dataset '{dataset_name}'"
        
        # Initialize results
        anomaly_results = {
            "dataset_name": dataset_name,
            "method": "Z-score",
            "parameters": {
                "window_size": window_size,
                "threshold": threshold
            },
            "results": {}
        }
        
        # Process each column
        for col in columns:
            series = df[col]
            
            # Calculate rolling statistics
            rolling_mean = series.rolling(window=window_size, center=True).mean()
            rolling_std = series.rolling(window=window_size, center=True).std()
            
            # Calculate Z-scores
            z_scores = np.abs((series - rolling_mean) / rolling_std)
            
            # Handle NaN values
            z_scores = z_scores.fillna(0)
            
            # Identify anomalies
            anomalies = z_scores > threshold
            anomaly_indices = np.where(anomalies)[0]
            
            # Group consecutive anomalies into regions
            anomaly_regions = []
            if len(anomaly_indices) > 0:
                # Initialize with the first anomaly
                start_idx = anomaly_indices[0]
                current_region = [start_idx]
                
                # Group consecutive indices
                for i in range(1, len(anomaly_indices)):
                    if anomaly_indices[i] == anomaly_indices[i-1] + 1:
                        current_region.append(anomaly_indices[i])
                    else:
                        # Save the current region and start a new one
                        end_idx = current_region[-1]
                        anomaly_regions.append({
                            "start_idx": int(start_idx),
                            "end_idx": int(end_idx),
                            "start_time": df.index[start_idx].strftime("%Y-%m-%d %H:%M:%S"),
                            "end_time": df.index[end_idx].strftime("%Y-%m-%d %H:%M:%S"),
                            "duration": str(df.index[end_idx] - df.index[start_idx]),
                            "max_zscore": float(z_scores.iloc[current_region].max()),
                            "mean_value": float(series.iloc[current_region].mean()),
                            "point_count": len(current_region)
                        })
                        
                        # Start new region
                        start_idx = anomaly_indices[i]
                        current_region = [start_idx]
                
                # Don't forget the last region
                end_idx = current_region[-1]
                anomaly_regions.append({
                    "start_idx": int(start_idx),
                    "end_idx": int(end_idx),
                    "start_time": df.index[start_idx].strftime("%Y-%m-%d %H:%M:%S"),
                    "end_time": df.index[end_idx].strftime("%Y-%m-%d %H:%M:%S"),
                    "duration": str(df.index[end_idx] - df.index[start_idx]),
                    "max_zscore": float(z_scores.iloc[current_region].max()),
                    "mean_value": float(series.iloc[current_region].mean()),
                    "point_count": len(current_region)
                })
            
            # Store results
            anomaly_results["results"][col] = {
                "anomaly_count": int(anomalies.sum()),
                "anomaly_percentage": float((anomalies.sum() / len(anomalies)) * 100),
                "anomaly_regions": anomaly_regions
            }
        
        # Create visualization
        fig, axes = plt.subplots(len(columns), 1, figsize=(12, 5 * len(columns)), sharex=True)
        
        # Handle single axis case
        if len(columns) == 1:
            axes = [axes]
        
        # Plot each column
        for i, col in enumerate(columns):
            series = df[col]
            anomaly_data = anomaly_results["results"][col]
            
            # Plot time series
            axes[i].plot(df.index, series, label=col)
            
            # Highlight anomaly regions
            for region in anomaly_data["anomaly_regions"]:
                start_idx = region["start_idx"]
                end_idx = region["end_idx"]
                
                # Highlight region
                axes[i].axvspan(
                    df.index[start_idx],
                    df.index[end_idx],
                    color='red',
                    alpha=0.3
                )
                
                # Mark points
                axes[i].scatter(
                    df.index[start_idx:end_idx+1],
                    series.iloc[start_idx:end_idx+1],
                    color='red',
                    marker='o'
                )
            
            # Add title and labels
            axes[i].set_title(f"{col} - {anomaly_data['anomaly_count']} anomalies "
                           f"({anomaly_data['anomaly_percentage']:.2f}%)")
            axes[i].set_ylabel("Value")
            axes[i].grid(True)
            
            # Add legend if there are anomalies
            if anomaly_data["anomaly_count"] > 0:
                axes[i].legend([col, "Anomalies"])
        
        # Set common x-axis label
        axes[-1].set_xlabel("Time")
        
        # Set figure title
        fig.suptitle(f"Z-score Anomaly Detection (window={window_size}, threshold={threshold})",
                   fontsize=16)
        fig.tight_layout()
        plt.subplots_adjust(top=0.95)
        
        # Save the figure
        os.makedirs("results", exist_ok=True)
        plot_path = f"results/{dataset_name}_zscore_anomalies.png"
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Save results to file
        result_path = f"results/{dataset_name}_zscore_anomalies.json"
        with open(result_path, 'w') as f:
            json.dump(anomaly_results, f, indent=2, default=str)
        
        # Generate summary
        summary = f"Z-score anomaly detection completed for {len(columns)} columns.\n"
        summary += f"Parameters: window_size={window_size}, threshold={threshold}\n"
        summary += f"Results saved to: {result_path}\n"
        summary += f"Plot saved to: {plot_path}\n\n"
        
        total_anomalies = sum(anomaly_results["results"][col]["anomaly_count"] for col in columns)
        summary += f"Total anomalies detected: {total_anomalies}\n\n"
        
        # Add anomaly details for each column
        for col in columns:
            anomaly_data = anomaly_results["results"][col]
            summary += f"Column: {col}\n"
            summary += f"  Anomalies: {anomaly_data['anomaly_count']} points "
            summary += f"({anomaly_data['anomaly_percentage']:.2f}% of data)\n"
            
            if anomaly_data["anomaly_regions"]:
                summary += f"  Anomaly regions: {len(anomaly_data['anomaly_regions'])}\n"
                for i, region in enumerate(anomaly_data["anomaly_regions"]):
                    summary += f"    Region {i+1}: {region['start_time']} to {region['end_time']} "
                    summary += f"({region['point_count']} points, max Z-score: {region['max_zscore']:.2f})\n"
            
            summary += "\n"
        
        return summary
    
    except Exception as e:
        return f"Error detecting anomalies with Z-score method: {str(e)}"

@tool
def detect_anomalies_isolation_forest(
    dataset_name: str,
    columns: Optional[List[str]] = None,
    contamination: float = 0.05,
    n_estimators: int = 100
) -> str:
    """
    Detect anomalies using Isolation Forest algorithm.
    
    Args:
        dataset_name: Name of the dataset to analyze
        columns: List of columns to analyze (default: all numeric columns)
        contamination: Expected proportion of anomalies (0.0 to 0.5)
        n_estimators: Number of estimators for the isolation forest
        
    Returns:
        String describing the detected anomalies
    """
    try:
        # Import Isolation Forest
        from sklearn.ensemble import IsolationForest
        
        # Load the DataFrame
        file_path = Path("data") / f"{dataset_name}.pkl"
        if not file_path.exists():
            return f"Dataset '{dataset_name}' not found. Please load it first using load_timeseries_data."
        
        df = pd.read_pickle(str(file_path))
        
        # Select columns to analyze
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            # Check if columns exist
            for col in columns:
                if col not in df.columns:
                    return f"Column '{col}' not found in dataset '{dataset_name}'"
        
        # Initialize results
        anomaly_results = {
            "dataset_name": dataset_name,
            "method": "Isolation Forest",
            "parameters": {
                "contamination": contamination,
                "n_estimators": n_estimators
            },
            "results": {}
        }
        
        # Process each column
        for col in columns:
            series = df[col].fillna(df[col].mean())  # Fill NaN with mean for IF
            
            # Reshape data for Isolation Forest
            X = series.values.reshape(-1, 1)
            
            # Create and fit model
            model = IsolationForest(
                n_estimators=n_estimators,
                contamination=contamination,
                random_state=42
            )
            model.fit(X)
            
            # Get predictions and anomaly scores
            predictions = model.predict(X)
            scores = model.decision_function(X)
            
            # Convert to anomaly flags (-1 = anomaly, 1 = normal)
            anomalies = np.where(predictions == -1, 1, 0)
            anomaly_indices = np.where(anomalies == 1)[0]
            
            # Group consecutive anomalies into regions
            anomaly_regions = []
            if len(anomaly_indices) > 0:
                # Initialize with the first anomaly
                start_idx = anomaly_indices[0]
                current_region = [start_idx]
                
                # Group consecutive indices
                for i in range(1, len(anomaly_indices)):
                    if anomaly_indices[i] == anomaly_indices[i-1] + 1:
                        current_region.append(anomaly_indices[i])
                    else:
                        # Save the current region and start a new one
                        end_idx = current_region[-1]
                        anomaly_regions.append({
                            "start_idx": int(start_idx),
                            "end_idx": int(end_idx),
                            "start_time": df.index[start_idx].strftime("%Y-%m-%d %H:%M:%S"),
                            "end_time": df.index[end_idx].strftime("%Y-%m-%d %H:%M:%S"),
                            "duration": str(df.index[end_idx] - df.index[start_idx]),
                            "mean_score": float(-scores[current_region].mean()),  # Negative for clarity (higher = more anomalous)
                            "mean_value": float(series.iloc[current_region].mean()),
                            "point_count": len(current_region)
                        })
                        
                        # Start new region
                        start_idx = anomaly_indices[i]
                        current_region = [start_idx]
                
                # Don't forget the last region
                end_idx = current_region[-1]
                anomaly_regions.append({
                    "start_idx": int(start_idx),
                    "end_idx": int(end_idx),
                    "start_time": df.index[start_idx].strftime("%Y-%m-%d %H:%M:%S"),
                    "end_time": df.index[end_idx].strftime("%Y-%m-%d %H:%M:%S"),
                    "duration": str(df.index[end_idx] - df.index[start_idx]),
                    "mean_score": float(-scores[current_region].mean()),  # Negative for clarity (higher = more anomalous)
                    "mean_value": float(series.iloc[current_region].mean()),
                    "point_count": len(current_region)
                })
            
            # Store results
            anomaly_results["results"][col] = {
                "anomaly_count": int(np.sum(anomalies)),
                "anomaly_percentage": float((np.sum(anomalies) / len(anomalies)) * 100),
                "anomaly_regions": anomaly_regions
            }
        
        # Create visualization
        fig, axes = plt.subplots(len(columns), 1, figsize=(12, 5 * len(columns)), sharex=True)
        
        # Handle single axis case
        if len(columns) == 1:
            axes = [axes]
        
        # Plot each column
        for i, col in enumerate(columns):
            series = df[col]
            anomaly_data = anomaly_results["results"][col]
            
            # Plot time series
            axes[i].plot(df.index, series, label=col)
            
            # Highlight anomaly regions
            for region in anomaly_data["anomaly_regions"]:
                start_idx = region["start_idx"]
                end_idx = region["end_idx"]
                
                # Highlight region
                axes[i].axvspan(
                    df.index[start_idx],
                    df.index[end_idx],
                    color='red',
                    alpha=0.3
                )
                
                # Mark points
                axes[i].scatter(
                    df.index[start_idx:end_idx+1],
                    series.iloc[start_idx:end_idx+1],
                    color='red',
                    marker='o'
                )
            
            # Add title and labels
            axes[i].set_title(f"{col} - {anomaly_data['anomaly_count']} anomalies "
                           f"({anomaly_data['anomaly_percentage']:.2f}%)")
            axes[i].set_ylabel("Value")
            axes[i].grid(True)
            
            # Add legend if there are anomalies
            if anomaly_data["anomaly_count"] > 0:
                axes[i].legend([col, "Anomalies"])
        
        # Set common x-axis label
        axes[-1].set_xlabel("Time")
        
        # Set figure title
        fig.suptitle(f"Isolation Forest Anomaly Detection (contam={contamination}, estimators={n_estimators})",
                   fontsize=16)
        fig.tight_layout()
        plt.subplots_adjust(top=0.95)
        
        # Save the figure
        os.makedirs("results", exist_ok=True)
        plot_path = f"results/{dataset_name}_iforest_anomalies.png"
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Save results to file
        result_path = f"results/{dataset_name}_iforest_anomalies.json"
        with open(result_path, 'w') as f:
            json.dump(anomaly_results, f, indent=2, default=str)
        
        # Generate summary
        summary = f"Isolation Forest anomaly detection completed for {len(columns)} columns.\n"
        summary += f"Parameters: contamination={contamination}, n_estimators={n_estimators}\n"
        summary += f"Results saved to: {result_path}\n"
        summary += f"Plot saved to: {plot_path}\n\n"
        
        total_anomalies = sum(anomaly_results["results"][col]["anomaly_count"] for col in columns)
        summary += f"Total anomalies detected: {total_anomalies}\n\n"
        
        # Add anomaly details for each column
        for col in columns:
            anomaly_data = anomaly_results["results"][col]
            summary += f"Column: {col}\n"
            summary += f"  Anomalies: {anomaly_data['anomaly_count']} points "
            summary += f"({anomaly_data['anomaly_percentage']:.2f}% of data)\n"
            
            if anomaly_data["anomaly_regions"]:
                summary += f"  Anomaly regions: {len(anomaly_data['anomaly_regions'])}\n"
                for i, region in enumerate(anomaly_data["anomaly_regions"]):
                    summary += f"    Region {i+1}: {region['start_time']} to {region['end_time']} "
                    summary += f"({region['point_count']} points, mean score: {region['mean_score']:.2f})\n"
            
            summary += "\n"
        
        return summary
    
    except ImportError:
        return "Error: scikit-learn is required for Isolation Forest but is not installed."
    except Exception as e:
        return f"Error detecting anomalies with Isolation Forest: {str(e)}"

@tool
def analyze_anomalies(
    dataset_name: str,
    method: str = "zscore",
    reference_anomalies: Optional[str] = None
) -> str:
    """
    Analyze the results of anomaly detection and compare with reference anomalies if available.
    
    Args:
        dataset_name: Name of the dataset
        method: Method used for anomaly detection ('zscore' or 'iforest')
        reference_anomalies: Optional path to file with reference anomalies
        
    Returns:
        String with analysis of anomaly detection results
    """
    try:
        # Check method
        if method not in ["zscore", "iforest"]:
            return f"Invalid method: {method}. Must be one of: 'zscore', 'iforest'."
        
        # Load results
        result_path = Path("results") / f"{dataset_name}_{method}_anomalies.json"
        if not result_path.exists():
            return f"Results file not found: {result_path}. Please run anomaly detection first."
        
        with open(result_path, 'r') as f:
            results = json.load(f)
        
        # Check if we have reference anomalies
        reference_data = None
        if reference_anomalies:
            ref_path = Path(reference_anomalies)
            if ref_path.exists():
                with open(ref_path, 'r') as f:
                    reference_data = json.load(f)
        
        # Generate analysis
        analysis = f"Analysis of {method.upper()} anomaly detection results for '{dataset_name}':\n\n"
        
        # Summary statistics
        columns = list(results["results"].keys())
        total_anomalies = sum(results["results"][col]["anomaly_count"] for col in columns)
        total_points = sum(len(results["results"][col]["anomaly_regions"]) for col in columns)
        
        analysis += f"Method: {method.upper()}\n"
        analysis += f"Parameters: {results['parameters']}\n"
        analysis += f"Total anomalies: {total_anomalies} points across {total_points} regions\n\n"
        
        # Analysis by column
        for col in columns:
            col_results = results["results"][col]
            analysis += f"Column: {col}\n"
            analysis += f"  Anomalies: {col_results['anomaly_count']} points "
            analysis += f"({col_results['anomaly_percentage']:.2f}% of data)\n"
            
            # Analyze regions
            if col_results["anomaly_regions"]:
                regions = col_results["anomaly_regions"]
                
                # Calculate statistics
                durations = []
                for region in regions:
                    if isinstance(region["duration"], str):
                        # Parse duration string
                        duration_parts = region["duration"].split()
                        if "days" in region["duration"]:
                            days = int(duration_parts[0])
                            if ":" in region["duration"]:
                                time_parts = duration_parts[2].split(":")
                                hours = int(time_parts[0])
                                minutes = int(time_parts[1])
                                seconds = int(float(time_parts[2]))
                            else:
                                hours = minutes = seconds = 0
                        else:
                            days = 0
                            time_parts = duration_parts[0].split(":")
                            hours = int(time_parts[0])
                            minutes = int(time_parts[1])
                            seconds = int(float(time_parts[2]))
                        
                        # Convert to total seconds
                        total_seconds = days * 86400 + hours * 3600 + minutes * 60 + seconds
                        durations.append(total_seconds)
                
                # Calculate statistics if we have durations
                if durations:
                    avg_duration = sum(durations) / len(durations)
                    max_duration = max(durations)
                    min_duration = min(durations)
                    
                    # Format durations
                    def format_duration(seconds):
                        days = seconds // 86400
                        hours = (seconds % 86400) // 3600
                        minutes = (seconds % 3600) // 60
                        secs = seconds % 60
                        
                        if days > 0:
                            return f"{days} days, {hours:02d}:{minutes:02d}:{secs:02d}"
                        else:
                            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
                    
                    analysis += f"  Number of regions: {len(regions)}\n"
                    analysis += f"  Average duration: {format_duration(avg_duration)}\n"
                    analysis += f"  Longest region: {format_duration(max_duration)}\n"
                    analysis += f"  Shortest region: {format_duration(min_duration)}\n"
                    
                    # Point distribution
                    points = [region["point_count"] for region in regions]
                    avg_points = sum(points) / len(points)
                    analysis += f"  Average points per region: {avg_points:.1f}\n"
            
            # Compare with reference if available
            if reference_data and "anomalies" in reference_data:
                ref_anomalies = [a for a in reference_data["anomalies"] if a["column"] == col]
                if ref_anomalies:
                    analysis += f"  Reference anomalies: {len(ref_anomalies)}\n"
                    
                    # Convert detected anomalies to indices for comparison
                    detected_indices = set()
                    for region in col_results["anomaly_regions"]:
                        for idx in range(region["start_idx"], region["end_idx"] + 1):
                            detected_indices.add(idx)
                    
                    # Convert reference anomalies to indices
                    ref_indices = set(a["index"] for a in ref_anomalies)
                    
                    # Calculate metrics
                    true_positives = len(detected_indices.intersection(ref_indices))
                    false_positives = len(detected_indices - ref_indices)
                    false_negatives = len(ref_indices - detected_indices)
                    
                    if true_positives + false_positives > 0:
                        precision = true_positives / (true_positives + false_positives)
                    else:
                        precision = 0
                        
                    if true_positives + false_negatives > 0:
                        recall = true_positives / (true_positives + false_negatives)
                    else:
                        recall = 0
                        
                    if precision + recall > 0:
                        f1_score = 2 * (precision * recall) / (precision + recall)
                    else:
                        f1_score = 0
                    
                    analysis += f"  Evaluation metrics:\n"
                    analysis += f"    True Positives: {true_positives}\n"
                    analysis += f"    False Positives: {false_positives}\n"
                    analysis += f"    False Negatives: {false_negatives}\n"
                    analysis += f"    Precision: {precision:.2f}\n"
                    analysis += f"    Recall: {recall:.2f}\n"
                    analysis += f"    F1 Score: {f1_score:.2f}\n"
            
            analysis += "\n"
        
        # Save analysis to file
        analysis_path = Path("results") / f"{dataset_name}_{method}_analysis.txt"
        with open(analysis_path, 'w') as f:
            f.write(analysis)
        
        return analysis
    
    except Exception as e:
        return f"Error analyzing anomalies: {str(e)}"