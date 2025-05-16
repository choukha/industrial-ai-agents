# itisa/tools/plotting_tool.py
from smolagents import tool
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
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
def plot_aggregated_time_series(aggregation_rule: str = 'D', columns: list = None) -> str:
    """
    Loads the processed time series data, aggregates it, and plots all specified
    columns (or all numeric columns if none specified) using Plotly. Saves the plot as an HTML file.

    Args:
        aggregation_rule (str): The pandas resampling rule (e.g., 'D' for daily, 'W' for weekly, 'H' for hourly). Defaults to 'D'.
        columns (list, optional): Specific columns to plot. If None, all numeric columns are plotted.

    Returns:
        str: A message indicating where the HTML plot is saved, or an error message.
    """
    df = _load_data()
    if df is None:
        return "Error: No processed data found. Please load data first using 'load_and_describe_data'."

    try:
        if columns:
            df_to_plot = df[columns]
        else:
            df_to_plot = df.select_dtypes(include=['number'])

        if df_to_plot.empty:
            return "Error: No numeric columns available or specified columns not found/numeric."

        df_aggregated = df_to_plot.resample(aggregation_rule).mean()

        num_plots = len(df_aggregated.columns)
        fig = make_subplots(rows=num_plots, cols=1, shared_xaxes=True, subplot_titles=df_aggregated.columns.tolist())

        for i, col in enumerate(df_aggregated.columns):
            fig.add_trace(go.Scatter(x=df_aggregated.index, y=df_aggregated[col], name=col, mode='lines+markers'),
                          row=i+1, col=1)

        fig.update_layout(
            title_text=f"Aggregated Time Series ({aggregation_rule})",
            height=300 * num_plots,
            showlegend=True
        )
        
        plot_file_path = _save_plotly_fig(fig, "aggregated_time_series_plot")
        return f"Aggregated time series plot saved to: {plot_file_path}"

    except Exception as e:
        return f"Error plotting aggregated time series: {str(e)}"

@tool
def plot_correlation_matrix(columns: list = None) -> str:
    """
    Loads the processed time series data and plots the correlation matrix for specified
    columns (or all numeric columns if none specified) using Plotly. Saves the plot as an HTML file.

    Args:
        columns (list, optional): Specific columns to include in correlation. If None, all numeric columns are used.

    Returns:
        str: A message indicating where the HTML plot is saved, or an error message.
    """
    df = _load_data()
    if df is None:
        return "Error: No processed data found. Please load data first."

    try:
        if columns:
            df_to_correlate = df[columns].select_dtypes(include=['number'])
        else:
            df_to_correlate = df.select_dtypes(include=['number'])

        if df_to_correlate.empty or df_to_correlate.shape[1] < 2:
            return "Error: Need at least two numeric columns to calculate correlation."

        corr_matrix = df_to_correlate.corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            text=corr_matrix.round(2).astype(str), # Add text annotations
            texttemplate="%{text}", # Show the text
            hoverongaps=False
        ))
        fig.update_layout(title_text="Correlation Matrix")

        plot_file_path = _save_plotly_fig(fig, "correlation_matrix_plot")
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.7:
                    strong_correlations.append(f"{corr_matrix.columns[i]} and {corr_matrix.columns[j]}: {corr_matrix.iloc[i, j]:.2f}")
        
        summary_message = f"Correlation matrix plot saved to: {plot_file_path}\n"
        if strong_correlations:
            summary_message += "Strong correlations (abs(corr) > 0.7):\n" + "\n".join(strong_correlations)
        else:
            summary_message += "No strong correlations (abs(corr) > 0.7) found."
            
        return summary_message

    except Exception as e:
        return f"Error plotting correlation matrix: {str(e)}"
