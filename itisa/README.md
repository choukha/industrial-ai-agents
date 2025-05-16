# ITISA - Industrial Time Series Analysis Agent

ITISA (Industrial Time Series Analysis Agent) enables users to analyze industrial time series data using natural language queries. It leverages a locally deployed Large Language Model (LLM) via Ollama and its function/tool-calling capabilities to interact with specialized Python data analysis tools.

## Overview

This agent is designed to simplify the analysis of sensor readings, process measurements, and other time-stamped data common in industrial settings. Users can ask questions or give commands in plain English, and ITISA will attempt to execute the appropriate analytical tasks (e.g., plotting, anomaly detection, statistical summary).

**Key Features:**
* Analyzes time series data from CSV files.
* Interprets natural language requests to select and execute relevant Python tools.
* Performs tasks like data loading, exploration, statistical analysis, anomaly detection, and visualization.
* Runs entirely on your local machine, ensuring data privacy.
* Uses an agent framework (e.g., `smolagents` or similar) to manage LLM-tool interaction.

For a comprehensive understanding of ITISA's architecture, capabilities, available tools, and advanced usage, please refer to the main **[ITISA Documentation](../../docs/itisa/index.md)**.

## Quick Start

### Prerequisites

1.  **Main Project Setup:** Ensure you have completed all steps in the main repository's **[Getting Started Guide](../../docs/getting_started.md)**. This includes:
    * Cloning the `industrial-ai-agents` repository.
    * Setting up a Python virtual environment.
    * Installing Ollama and pulling necessary LLMs (choose models known for good function/tool calling, e.g., `qwen2:7b`, `llama3:8b-instruct`, or smaller `phi3:mini-instruct`).
    * Installing root dependencies from `requirements.txt`.
2.  **ITISA Dependencies:** Install ITISA-specific dependencies:
    ```bash
    # From the root of the industrial-ai-agents repository
    pip install -r itisa/requirements.txt
    ```
    This will install packages like `pandas`, `scikit-learn`, `statsmodels`, `matplotlib`, `plotly`, etc.

### Running ITISA

1.  **Activate Virtual Environment:**
    ```bash
    # From the root of the industrial-ai-agents repository
    # On Windows:
    # venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```
2.  **Ensure Ollama is Running:** Check that the Ollama service/application is active.
3.  **Navigate to ITISA Directory:**
    ```bash
    # From the root of the industrial-ai-agents repository
    cd itisa
    ```
4.  **Run the Application:**
    ```bash
    python app.py
    ```
5.  Open your web browser and go to the URL shown in the terminal (usually `http://127.0.0.1:7860`).
    *(Note: If IDOCA or another application is using port 7860, ITISA might start on a different port or fail. Ensure the port is free or modify `app.py` to use a different port.)*
6.  Follow the UI instructions:
    * Upload your time series CSV file.
    * Click "Load Data" (or similar, depending on UI).
    * Select the LLM model.
    * Type your natural language query (e.g., "Plot the temperature column", "Detect anomalies in pressure").
    * View the results, which may include text summaries and plots.

## Detailed Documentation

For more detailed information on:
* Architecture and Tool Library
* Advanced Setup and Configuration
* In-depth Usage Guide and Example Queries
* Adding New Tools and Customization
* Troubleshooting

Please visit the main **[ITISA Documentation section](../../docs/itisa/index.md)**.

## Reporting Issues

If you encounter any issues specific to ITISA, please report them on the main repository's [issue tracker](https://github.com/choukha/industrial-ai-agents/issues), providing details about the ITISA agent.
