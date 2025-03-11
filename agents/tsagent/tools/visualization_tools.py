from smolagents import tool
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from typing import List, Optional, Dict, Union
import base64
import io
import json

@tool
def create_time_series_plot(
    dataset_name: str,
    columns: Optional[List[str]] = None,
    title: str = "Time Series Data",
    resample_rule: Optional[str] = None,
    highlight_anomalies: bool = False
) -> str:
    """
    Create time series plots for selected columns.
    
    Args:
        dataset_name: Name of the dataset to visualize
        columns: List of columns to visualize (default: all numeric columns)
        title: Title for the plots
        resample_rule: Optional rule for resampling data (e.g., '1h', '15min')
        highlight_anomalies: Whether to highlight known anomalies (if available)
        
    Returns:
        String with the path to the saved plot
    """
    try:
        # Load the DataFrame
        file_path = Path("data") / f"{dataset_name}.pkl"
        if not file_path.exists():
            return f"Dataset '{dataset_name}' not found. Please load it first using load_timeseries_data."
        
        df = pd.read_pickle(str(file_path))
        
        # Resample if specified
        if resample_rule:
            df = df.resample(resample_rule).mean()
        
        # Select columns to visualize
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            # Check if columns exist
            for col in columns:
                if col not in df.columns:
                    return f"Column '{col}' not found in dataset '{dataset_name}'"
        
        # Load anomaly information if available and requested
        anomaly_regions = None
        if highlight_anomalies:
            anomaly_path = Path("data") / f"{dataset_name}_anomalies.json"
            if anomaly_path.exists():
                with open(anomaly_path, 'r') as f:
                    anomaly_data = json.load(f)
                    anomaly_regions = anomaly_data.get("anomalies", [])
        
        # Create figure with subplots
        n_plots = len(columns)
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots), sharex=True)
        
        # Handle single axis case
        if n_plots == 1:
            axes = [axes]
        
        # Plot each column
        for i, col in enumerate(columns):
            # Plot the time series
            axes[i].plot(df.index, df[col], label=col)
            
            # Highlight anomalies if available
            if highlight_anomalies and anomaly_regions:
                for anomaly in anomaly_regions:
                    if anomaly["column"] == col:
                        idx = anomaly["index"]
                        if 0 <= idx < len(df):
                            axes[i].scatter(df.index[idx], df.iloc[idx][col], 
                                         color='red', s=50, marker='o', label='Anomaly')
            
            # Set labels and title
            axes[i].set_title(f"{col}")
            axes[i].set_ylabel("Value")
            axes[i].grid(True)
            
            # Only show legend if there are anomalies
            if highlight_anomalies and anomaly_regions:
                handles, labels = axes[i].get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                axes[i].legend(by_label.values(), by_label.keys())
        
        # Set common x-axis label
        axes[-1].set_xlabel("Time")
        
        # Set figure title
        fig.suptitle(title, fontsize=16)
        fig.tight_layout()
        plt.subplots_adjust(top=0.95)
        
        # Save the figure
        os.makedirs("results", exist_ok=True)
        plot_path = f"results/{dataset_name}_timeseries.png"
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Also save interactive HTML using plotly if available
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            fig = make_subplots(rows=n_plots, cols=1, shared_xaxes=True,
                              subplot_titles=columns)
            
            for i, col in enumerate(columns):
                fig.add_trace(
                    go.Scatter(x=df.index, y=df[col], name=col),
                    row=i+1, col=1
                )
                
                # Add anomaly points if available
                if highlight_anomalies and anomaly_regions:
                    anomaly_indices = [a["index"] for a in anomaly_regions if a["column"] == col]
                    if anomaly_indices:
                        fig.add_trace(
                            go.Scatter(
                                x=[df.index[idx] for idx in anomaly_indices if 0 <= idx < len(df)],
                                y=[df.iloc[idx][col] for idx in anomaly_indices if 0 <= idx < len(df)],
                                mode='markers',
                                marker=dict(color='red', size=10),
                                name=f'{col} Anomalies'
                            ),
                            row=i+1, col=1
                        )
            
            fig.update_layout(
                height=300*n_plots,
                title_text=title,
                showlegend=True
            )
            
            html_path = f"results/{dataset_name}_timeseries.html"
            fig.write_html(html_path)
            
            return f"Time series plot created for {len(columns)} columns:\n" + \
                   f"Static plot saved to: {plot_path}\n" + \
                   f"Interactive plot saved to: {html_path}"
                   
        except ImportError:
            return f"Time series plot created for {len(columns)} columns:\n" + \
                   f"Plot saved to: {plot_path}"
    
    except Exception as e:
        return f"Error creating time series plot: {str(e)}"

@tool
def create_correlation_heatmap(
    dataset_name: str,
    columns: Optional[List[str]] = None,
    title: str = "Correlation Heatmap"
) -> str:
    """
    Create a correlation heatmap for the dataset.
    
    Args:
        dataset_name: Name of the dataset to analyze
        columns: List of columns to include (default: all numeric columns)
        title: Title for the heatmap
        
    Returns:
        String with the path to the saved plot
    """
    try:
        # Load the DataFrame
        file_path = Path("data") / f"{dataset_name}.pkl"
        if not file_path.exists():
            return f"Dataset '{dataset_name}' not found. Please load it first using load_timeseries_data."
        
        df = pd.read_pickle(str(file_path))
        
        # Select columns to visualize
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            # Check if columns exist
            for col in columns:
                if col not in df.columns:
                    return f"Column '{col}' not found in dataset '{dataset_name}'"
        
        # Calculate correlation matrix
        corr = df[columns].corr()
        
        # Create figure
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title(title)
        plt.tight_layout()
        
        # Save the figure
        os.makedirs("results", exist_ok=True)
        plot_path = f"results/{dataset_name}_correlation.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate and format strong correlations
        strong_correlations = []
        for i in range(len(columns)):
            for j in range(i+1, len(columns)):
                corr_val = corr.iloc[i, j]
                if abs(corr_val) > 0.5:  # Threshold for "strong" correlation
                    strong_correlations.append(f"{columns[i]} and {columns[j]}: {corr_val:.2f}")
        
        # Return paths and strong correlations
        result = f"Correlation heatmap created for {len(columns)} columns.\n" + \
                 f"Plot saved to: {plot_path}\n"
        
        if strong_correlations:
            result += "\nStrong correlations:\n"
            result += "\n".join(strong_correlations)
        
        return result
    
    except Exception as e:
        return f"Error creating correlation heatmap: {str(e)}"

@tool
def create_distribution_plot(
    dataset_name: str,
    columns: Optional[List[str]] = None,
    plot_type: str = "histogram",
    bins: int = 30
) -> str:
    """
    Create distribution plots for selected columns.
    
    Args:
        dataset_name: Name of the dataset to analyze
        columns: List of columns to include (default: all numeric columns)
        plot_type: Type of plot ('histogram', 'kde', or 'both')
        bins: Number of bins for histogram
        
    Returns:
        String with the path to the saved plot
    """
    try:
        # Load the DataFrame
        file_path = Path("data") / f"{dataset_name}.pkl"
        if not file_path.exists():
            return f"Dataset '{dataset_name}' not found. Please load it first using load_timeseries_data."
        
        df = pd.read_pickle(str(file_path))
        
        # Select columns to visualize
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            # Check if columns exist
            for col in columns:
                if col not in df.columns:
                    return f"Column '{col}' not found in dataset '{dataset_name}'"
        
        # Check plot_type
        if plot_type not in ['histogram', 'kde', 'both']:
            return f"Invalid plot_type: {plot_type}. Must be one of: 'histogram', 'kde', 'both'."
        
        # Create figure with subplots
        n_plots = len(columns)
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots))
        
        # Handle single axis case
        if n_plots == 1:
            axes = [axes]
        
        # Plot each column
        for i, col in enumerate(columns):
            data = df[col].dropna()
            
            if plot_type == 'histogram':
                axes[i].hist(data, bins=bins, alpha=0.7)
                
            elif plot_type == 'kde':
                sns.kdeplot(data, ax=axes[i])
                
            elif plot_type == 'both':
                sns.histplot(data, bins=bins, kde=True, ax=axes[i])
            
            # Add mean and median lines
            mean_val = data.mean()
            median_val = data.median()
            axes[i].axvline(mean_val, color='red', linestyle='-', label=f'Mean: {mean_val:.2f}')
            axes[i].axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}')
            
            # Set labels and add statistics
            axes[i].set_title(f"Distribution of {col}")
            axes[i].set_xlabel("Value")
            axes[i].set_ylabel("Frequency" if plot_type != 'kde' else "Density")
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()
            
            # Add statistics as text
            stats = f"Mean: {mean_val:.2f}\nMedian: {median_val:.2f}\n"
            stats += f"Min: {data.min():.2f}\nMax: {data.max():.2f}\n"
            stats += f"Std Dev: {data.std():.2f}"
            axes[i].text(0.02, 0.95, stats, transform=axes[i].transAxes, 
                      fontsize=9, verticalalignment='top', 
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        # Set figure title
        fig.suptitle(f"Distribution Plots ({plot_type})", fontsize=16)
        fig.tight_layout()
        plt.subplots_adjust(top=0.95)
        
        # Save the figure
        os.makedirs("results", exist_ok=True)
        plot_path = f"results/{dataset_name}_distribution_{plot_type}.png"
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return f"Distribution plots ({plot_type}) created for {len(columns)} columns.\n" + \
               f"Plot saved to: {plot_path}"
    
    except Exception as e:
        return f"Error creating distribution plot: {str(e)}"

@tool
def create_lag_plot(
    dataset_name: str,
    column: str,
    lags: List[int] = [1, 7, 14, 30]
) -> str:
    """
    Create lag plots to analyze time series autocorrelation.
    
    Args:
        dataset_name: Name of the dataset to analyze
        column: Column to analyze
        lags: List of lag values to plot
        
    Returns:
        String with the path to the saved plot
    """
    try:
        # Load the DataFrame
        file_path = Path("data") / f"{dataset_name}.pkl"
        if not file_path.exists():
            return f"Dataset '{dataset_name}' not found. Please load it first using load_timeseries_data."
        
        df = pd.read_pickle(str(file_path))
        
        # Check if column exists
        if column not in df.columns:
            return f"Column '{column}' not found in dataset '{dataset_name}'"
        
        # Extract the series
        series = df[column].dropna()
        
        # Calculate number of subplots
        n_lags = len(lags)
        n_cols = min(2, n_lags)
        n_rows = (n_lags + n_cols - 1) // n_cols
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
        
        # Handle different subplot layouts
        if n_lags == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Create lag plots
        for i, lag in enumerate(lags):
            if i < len(axes):
                # Create lag series
                lag_series = series.shift(lag)
                valid_idx = ~pd.isna(lag_series)
                
                # Calculate correlation
                corr = series[valid_idx].corr(lag_series[valid_idx])
                
                # Plot
                axes[i].scatter(lag_series, series, alpha=0.5)
                axes[i].set_title(f"Lag {lag} (Correlation: {corr:.3f})")
                axes[i].set_xlabel(f"{column} (t-{lag})")
                axes[i].set_ylabel(f"{column} (t)")
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_lags, len(axes)):
            axes[i].set_visible(False)
        
        # Set figure title
        fig.suptitle(f"Lag Plots for {column}", fontsize=16)
        fig.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        # Save the figure
        os.makedirs("results", exist_ok=True)
        plot_path = f"results/{dataset_name}_{column}_lag_plot.png"
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Calculate autocorrelation summary
        autocorr_summary = []
        for lag in lags:
            lag_series = series.shift(lag)
            valid_idx = ~pd.isna(lag_series)
            corr = series[valid_idx].corr(lag_series[valid_idx])
            significance = "Strong" if abs(corr) > 0.7 else "Moderate" if abs(corr) > 0.3 else "Weak"
            autocorr_summary.append(f"Lag {lag}: {corr:.3f} ({significance})")
        
        return f"Lag plots created for column '{column}' with {len(lags)} lags.\n" + \
               f"Plot saved to: {plot_path}\n\n" + \
               "Autocorrelation summary:\n" + \
               "\n".join(autocorr_summary)
    
    except Exception as e:
        return f"Error creating lag plot: {str(e)}"

@tool
def create_seasonality_plot(
    dataset_name: str,
    column: str,
    period: int = 24  # Default: daily seasonality for hourly data
) -> str:
    """
    Create a seasonality decomposition plot for a time series.
    
    Args:
        dataset_name: Name of the dataset to analyze
        column: Column to analyze
        period: Seasonality period
        
    Returns:
        String with the path to the saved plot
    """
    try:
        # Import statsmodels
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        # Load the DataFrame
        file_path = Path("data") / f"{dataset_name}.pkl"
        if not file_path.exists():
            return f"Dataset '{dataset_name}' not found. Please load it first using load_timeseries_data."
        
        df = pd.read_pickle(str(file_path))
        
        # Check if column exists
        if column not in df.columns:
            return f"Column '{column}' not found in dataset '{dataset_name}'"
        
        # Extract the series
        series = df[column].dropna()
        
        # Check if enough data
        if len(series) < 2 * period:
            return f"Not enough data for seasonality analysis with period {period}. Need at least {2 * period} data points."
        
        # Perform seasonal decomposition
        decomposition = seasonal_decompose(
            series, 
            model='additive', 
            period=period,
            extrapolate_trend='freq'
        )
        
        # Create figure with subplots
        fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
        
        # Plot components
        axes[0].plot(decomposition.observed)
        axes[0].set_title('Observed')
        axes[0].grid(True)
        
        axes[1].plot(decomposition.trend)
        axes[1].set_title('Trend')
        axes[1].grid(True)
        
        axes[2].plot(decomposition.seasonal)
        axes[2].set_title(f'Seasonality (Period: {period})')
        axes[2].grid(True)
        
        axes[3].plot(decomposition.resid)
        axes[3].set_title('Residual')
        axes[3].grid(True)
        
        # Calculate seasonality strength
        seasonal_strength = max(0, 1 - (np.nanvar(decomposition.resid) / np.nanvar(decomposition.seasonal + decomposition.resid)))
        trend_strength = max(0, 1 - (np.nanvar(decomposition.resid) / np.nanvar(decomposition.trend + decomposition.resid)))
        
        # Set overall title with strengths
        fig.suptitle(
            f"Seasonality Decomposition of {column}\n" + 
            f"Seasonal Strength: {seasonal_strength:.3f}, Trend Strength: {trend_strength:.3f}",
            fontsize=16
        )
        
        # Adjust layout
        fig.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # Save the figure
        os.makedirs("results", exist_ok=True)
        plot_path = f"results/{dataset_name}_{column}_seasonality.png"
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Generate interpretation
        interpretation = f"Seasonality analysis for '{column}' with period {period}:\n\n"
        
        interpretation += f"Seasonal Strength: {seasonal_strength:.3f} "
        if seasonal_strength > 0.7:
            interpretation += "(Strong seasonality)\n"
        elif seasonal_strength > 0.3:
            interpretation += "(Moderate seasonality)\n"
        else:
            interpretation += "(Weak or no seasonality)\n"
            
        interpretation += f"Trend Strength: {trend_strength:.3f} "
        if trend_strength > 0.7:
            interpretation += "(Strong trend)\n"
        elif trend_strength > 0.3:
            interpretation += "(Moderate trend)\n"
        else:
            interpretation += "(Weak or no trend)\n"
        
        return f"Seasonality decomposition plot created for '{column}' with period {period}.\n" + \
               f"Plot saved to: {plot_path}\n\n" + \
               interpretation
    
    except ImportError:
        return "Error: statsmodels is required for seasonality analysis but is not installed."
    except Exception as e:
        return f"Error creating seasonality plot: {str(e)}"