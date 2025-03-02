import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union

def save_timeseries_to_csv(df: pd.DataFrame, filename: str = None) -> str:
    """
    Save a timeseries DataFrame to CSV in the data directory.
    
    Args:
        df: DataFrame to save
        filename: Optional filename, if None generates timestamp-based name
        
    Returns:
        Path to saved CSV file
    """
    # Create data directory if it doesn't exist
    data_dir = Path(__file__).parent.parent / 'data'
    data_dir.mkdir(exist_ok=True)
    
    # Ensure .csv extension
    if not filename.endswith('.csv'):
        filename += '.csv'
    
    filepath = data_dir / filename
    
    # Save DataFrame
    df.to_csv(filepath)
    return str('data' + '/' + filename)

def create_timeseries_plot(
    df: pd.DataFrame,
    plot_title: str = "Time Series Data",
    height_per_subplot: int = 300,
    resample_rule: str = '15min'  # '15T' for 15 minutes
) -> go.Figure:
    """
    Create an interactive Plotly time series plot with each column in separate subplot.
    
    Args:
        df: DataFrame with datetime index and tag columns
        plot_title: Title for the entire plot
        height_per_subplot: Height in pixels for each subplot
        resample_rule: Pandas resample rule (e.g., '15min' for 15 minutes)
        
    Returns:
        Plotly figure object
    """
    # Resample data to reduce points
    df_resampled = df.resample(resample_rule).mean()
    
    n_subplots = len(df_resampled.columns)
    
    # Create subplots
    fig = make_subplots(
        rows=n_subplots,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=df_resampled.columns.tolist()
    )
    
    # Add traces for each column
    for i, col in enumerate(df_resampled.columns, 1):
        fig.add_trace(
            go.Scatter(
                x=df_resampled.index,
                y=df_resampled[col],
                name=col,
                mode='lines',
                connectgaps=True
            ),
            row=i,
            col=1
        )
    
    # Update layout
    fig.update_layout(
        height=height_per_subplot * n_subplots,
        title=f"{plot_title} (Aggregated {resample_rule})",
        showlegend=True
    )
    
    return fig

def add_events_to_plot(
    fig: go.Figure,
    events_df: pd.DataFrame,
    start_col: str = 'start_time',
    end_col: str = 'end_time',
    color: str = "LightSalmon",
    opacity: float = 0.5
) -> go.Figure:
    """
    Add event overlays to an existing time series plot.
    
    Args:
        fig: Existing Plotly figure
        events_df: DataFrame containing event data with start and end times
        start_col: Column name for event start times
        end_col: Column name for event end times
        color: Color for event rectangles
        opacity: Opacity for event rectangles
        
    Returns:
        Updated Plotly figure
    """
    for idx, event in events_df.iterrows():
        # Format the duration as a string for annotation
        duration = pd.to_datetime(event[end_col]) - pd.to_datetime(event[start_col])
        duration_str = f"Event {idx+1}\n{duration}"
        
        fig.add_vrect(
            x0=event[start_col],
            x1=event[end_col],
            fillcolor=color,
            opacity=opacity,
            layer="below",
            line_width=0,
            annotation_text=duration_str,
            annotation_position="top left",
            annotation_textangle=90,
            annotation_font_size=8
        )
    
    return fig

def save_plot(
    fig: go.Figure,
    filename: str
) -> str:
    """
    Save a Plotly figure as HTML.
    
    Args:
        fig: Plotly figure to save
        filename: Base filename without extension
        
    Returns:
        Path to saved HTML file
    """
    # Create plots directory if it doesn't exist
    plots_path = Path(__file__).parent.parent / 'plots'
    plots_path.mkdir(exist_ok=True)
    
    # Save only HTML
    filepath = plots_path / f"{filename}.html"
    fig.write_html(str(filepath))
    
    return str('plots' + '/' + f"{filename}.html")