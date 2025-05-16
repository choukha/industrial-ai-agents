# itisa/tools/analysis_tool.py
from smolagents import tool
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import seasonal_decompose
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json

# Ensure data and results directories exist
DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PROCESSED_DATA_PATH = DATA_DIR / "current_timeseries_data.pkl"

def _load_data() -> pd.DataFrame | None:
    """Helper function to load the processed data."""
    if PROCESSED_DATA_PATH.exists():
        return pd.read_pickle(PROCESSED_DATA_PATH)
    return None

def _save_plotly_fig(fig, filename_base: str) -> str:
    """Helper function to save a Plotly figure and return its path."""
    plot_path = RESULTS_DIR / f"{filename_base}.html"
    fig.write_html(str(plot_path))
    return str(plot_path)

@tool
def detect_anomalies_iforest_and_plot(column_name: str, contamination: float = 0.05, n_estimators: int = 100) -> str:
    """
    Detects anomalies in a specified column of the processed time series data
    using Isolation Forest, plots the series with anomalies highlighted, and returns a summary.

    Args:
        column_name (str): The name of the column to analyze for anomalies.
        contamination (float): The expected proportion of outliers in the data set. Defaults to 0.05.
        n_estimators (int): The number of base estimators in the ensemble. Defaults to 100.

    Returns:
        str: A summary of detected anomalies and the path to the HTML plot, or an error message.
    """
    df = _load_data()
    if df is None:
        return "Error: No processed data found. Please load data first."

    if column_name not in df.columns:
        return f"Error: Column '{column_name}' not found in the dataset. Available columns: {', '.join(df.columns)}"
    
    if not pd.api.types.is_numeric_dtype(df[column_name]):
        return f"Error: Column '{column_name}' is not numeric and cannot be used for anomaly detection."

    try:
        series = df[[column_name]].copy() # Use a copy to avoid SettingWithCopyWarning
        series.dropna(inplace=True) # Isolation Forest cannot handle NaNs

        if series.empty:
            return f"Error: Column '{column_name}' contains no valid data after dropping NaNs."

        model = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=42)
        series['anomaly'] = model.fit_predict(series[[column_name]])
        
        anomalies = series[series['anomaly'] == -1]
        num_anomalies = len(anomalies)
        anomaly_percentage = (num_anomalies / len(series)) * 100

        # Plotting
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=series.index, y=series[column_name], name='Normal Data', mode='lines'))
        fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies[column_name], name='Anomaly', mode='markers', marker=dict(color='red', size=8)))
        fig.update_layout(title_text=f"Anomaly Detection for '{column_name}' using Isolation Forest",
                          xaxis_title="Time", yaxis_title="Value")
        
        plot_file_path = _save_plotly_fig(fig, f"iforest_anomalies_{column_name}")

        summary = (f"Isolation Forest anomaly detection for column '{column_name}':\n"
                   f"Number of anomalies detected: {num_anomalies}\n"
                   f"Percentage of data considered anomalous: {anomaly_percentage:.2f}%\n"
                   f"Plot with anomalies highlighted saved to: {plot_file_path}")
        return summary

    except Exception as e:
        return f"Error detecting anomalies with Isolation Forest for column '{column_name}': {str(e)}"

@tool
def get_trend_seasonality_summary(column_name: str, model_type: str = 'additive', period: int = None) -> str:
    """
    Performs seasonal decomposition on a specified column and returns a summary of trend and seasonality.
    Automatically determines period if data frequency can be inferred and period is not provided.
    Common periods: Daily data often has period 7 (weekly) or 30/31 (monthly). Hourly data might have 24 (daily).

    Args:
        column_name (str): The name of the column to decompose.
        model_type (str): Type of decomposition ('additive' or 'multiplicative'). Defaults to 'additive'.
        period (int, optional): The period of the seasonality. If None, attempts to infer.
                                E.g., 7 for daily data with weekly seasonality, 24 for hourly data with daily seasonality.

    Returns:
        str: A summary of the trend and seasonality, and path to plot, or an error message.
    """
    df = _load_data()
    if df is None:
        return "Error: No processed data found. Please load data first."

    if column_name not in df.columns:
        return f"Error: Column '{column_name}' not found. Available columns: {', '.join(df.columns)}"

    if not pd.api.types.is_numeric_dtype(df[column_name]):
        return f"Error: Column '{column_name}' is not numeric."
        
    series = df[column_name].dropna()
    if series.empty:
        return f"Error: Column '{column_name}' has no valid data after dropping NaNs."

    # Attempt to infer period if not provided
    if period is None:
        inferred_freq = pd.infer_freq(series.index)
        if inferred_freq:
            if 'H' in inferred_freq: period = 24  # Hourly data, daily seasonality
            elif 'D' in inferred_freq: period = 7    # Daily data, weekly seasonality
            elif 'B' in inferred_freq: period = 5    # Business daily data
            elif 'W' in inferred_freq: period = 52//pd.tseries.frequencies.get_freq(inferred_freq).n # approx for W-MON etc.
            elif 'M' in inferred_freq or 'MS' in inferred_freq : period = 12 # Monthly data, yearly seasonality
            else:
                return f"Error: Could not automatically infer a common period for frequency '{inferred_freq}'. Please specify a 'period'."
        else: # If frequency cannot be inferred, require period.
             return "Error: Could not infer data frequency. Please specify a 'period' for decomposition (e.g., 7 for daily data with weekly seasonality)."


    if len(series) < 2 * period:
        return (f"Error: Not enough data for seasonal decomposition with column '{column_name}' and period {period}. "
                f"Need at least {2 * period} data points, but found {len(series)}.")

    try:
        decomposition = seasonal_decompose(series, model=model_type, period=period, extrapolate_trend='freq')

        # Plotting
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                            subplot_titles=['Observed', 'Trend', 'Seasonal', 'Residual'])
        fig.add_trace(go.Scatter(x=decomposition.observed.index, y=decomposition.observed, name='Observed'), row=1, col=1)
        fig.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, name='Trend'), row=2, col=1)
        fig.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, name='Seasonal'), row=3, col=1)
        fig.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid, name='Residual', mode='markers'), row=4, col=1)
        
        fig.update_layout(title_text=f"Seasonal Decomposition for '{column_name}' (Period: {period})", height=800)
        plot_file_path = _save_plotly_fig(fig, f"seasonal_decomposition_{column_name}")

        # Strength of Trend and Seasonality
        # F_T = max(0, 1 - Var(Residual)/Var(Trend + Residual))
        # F_S = max(0, 1 - Var(Residual)/Var(Seasonal + Residual))
        trend_strength = max(0, 1 - np.var(decomposition.resid.dropna()) / np.var((decomposition.trend + decomposition.resid).dropna())) if len(decomposition.resid.dropna()) > 1 and len((decomposition.trend + decomposition.resid).dropna()) > 1 else "N/A"
        seasonal_strength = max(0, 1 - np.var(decomposition.resid.dropna()) / np.var((decomposition.seasonal + decomposition.resid).dropna())) if len(decomposition.resid.dropna()) > 1 and len((decomposition.seasonal + decomposition.resid).dropna()) > 1 else "N/A"


        summary = (f"Seasonal decomposition for column '{column_name}' (Model: {model_type}, Period: {period}):\n"
                   f"Trend component strength: {trend_strength if isinstance(trend_strength, str) else f'{trend_strength:.2f}'}\n"
                   f"Seasonal component strength: {seasonal_strength if isinstance(seasonal_strength, str) else f'{seasonal_strength:.2f}'}\n"
                   f"Decomposition plot saved to: {plot_file_path}")
        return summary

    except Exception as e:
        return f"Error during seasonal decomposition for column '{column_name}': {str(e)}"

