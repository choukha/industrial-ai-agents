# ITISA - Industrial Time Series Analysis Agent

Welcome to the documentation for ITISA (Industrial Time Series Analysis Agent). ITISA is designed to enable users to analyze industrial time series data through natural language interaction, leveraging a locally deployed Large Language Model (LLM) and its function/tool-calling capabilities.

## 1. Overview

Industrial processes generate vast amounts of time series data from sensors, machinery, and control systems. This data includes:
* Process variables (temperature, pressure, flow rates)
* Equipment status and performance metrics
* Quality control measurements
* Energy consumption data

Analyzing this data is crucial for process optimization, predictive maintenance, quality assurance, and operational efficiency. ITISA aims to make this analysis more accessible by allowing users to make requests in plain English, which the agent then translates into calls to specialized data analysis and visualization tools.

**Key Capabilities:**
* Load and explore time series data from CSV files.
* Perform statistical analysis and generate descriptive summaries.
* Detect anomalies using various algorithms.
* Analyze trends and seasonality.
* Generate plots and visualizations of the data.
* Interpret natural language queries to select and execute appropriate analytical tools.
* Run entirely locally using Ollama and open-source models.

## 2. Architecture

ITISA is built on a **function-calling (or tool-using)** architecture. The LLM acts as the central orchestrator, interpreting user requests and deciding which predefined Python tools to execute. For a detailed explanation, please refer to the main [Architecture Overview](../architecture.md#itisa-industrial-time-series-analysis-agent-architecture).

**Core Workflow:**

1.  **User Query:** A user submits a natural language request related to time series data (e.g., "Plot the temperature for the last 24 hours," "Are there any anomalies in the pressure readings?").
2.  **LLM Interpretation:** The LLM, provided with a list of available tools and their descriptions, parses the query to understand the user's intent and identify necessary parameters.
3.  **Tool Selection & Parameterization:** The LLM selects the most appropriate tool(s) from its library and determines the arguments to pass to them.
4.  **Tool Execution:** The agent framework executes the selected Python function(s) with the specified arguments. These tools interact with the data (e.g., using Pandas, Scikit-learn, Matplotlib).
5.  **Result Aggregation:** The output from the tool(s) (e.g., a statistical summary, a path to a generated plot, a list of anomalies) is returned to the LLM.
6.  **Response Generation:** The LLM formulates a natural language response to the user, incorporating the results from the tool execution.

![ITISA Function Calling Architecture Diagram](../assets/itisa-architecture.png)
*(Reminder: Ensure you have an `itisa-architecture.png` in your `docs/assets/` folder or update path)*

## 3. Setup and Installation

Before running ITISA, ensure you have completed the general setup described in the main **[Getting Started Guide](../getting_started.md)**. This includes:
* System prerequisites met.
* Python virtual environment created and activated.
* Ollama installed and running.
* Root project dependencies installed (`pip install -r requirements.txt`).

**ITISA-Specific Setup:**

1.  **Install ITISA Dependencies:**
    Navigate to the repository's root directory and run:
    ```bash
    pip install -r itisa/requirements.txt
    ```
    This will install packages like `pandas`, `scikit-learn`, `statsmodels`, `matplotlib`, `seaborn`, `plotly`, and `smolagents` (or the chosen agent framework).
2.  **Pull Required Ollama Models:**
    * **LLM for Tool Calling (choose based on your hardware):**
        * Models with strong function/tool calling capabilities are recommended.
        * Examples: `ollama pull qwen2:7b` (often good for tool use), `ollama pull llama3:8b-instruct`
        * Lighter alternatives: `ollama pull phi3:mini-instruct`, `ollama pull qwen2:1.5b-instruct`
        Ensure the model selected is specified in `itisa/app.py` (or its configuration) or selectable in the UI.

## 4. Running ITISA

1.  Ensure Ollama is running.
2.  Activate your Python virtual environment.
3.  Navigate to the `itisa` directory within the project:
    ```bash
    cd itisa
    ```
4.  Run the main application:
    ```bash
    python app.py
    ```
5.  Open your web browser and go to the URL displayed in the terminal (usually `http://127.0.0.1:7860`).
    *(Note: If IDOCA or another app is using port 7860, ITISA might use a different port or fail. Ensure the port is free or configure ITISA to use a different one in `app.py`.)*

## 5. Usage Guide

The ITISA Gradio interface typically includes:

1.  **Upload CSV:**
    * Click to upload your time series data in CSV format. Ensure your CSV has a parsable datetime column and clear headers.
2.  **Load Data:**
    * After uploading, click a "Load Data" button (or similar) to make the agent process and recognize the dataset. The agent might provide a brief description of the loaded data (e.g., columns, number of rows).
3.  **Select Model:**
    * Choose the Ollama LLM you want the agent to use for interpreting your requests.
4.  **Ask your question:**
    * Type your natural language query about the loaded time series data in the input textbox.
    * Examples:
        * "Load and describe the data from the uploaded file."
        * "Plot the 'temperature' and 'pressure' columns."
        * "Show me a trend chart of 'flow_rate_A' over the last week." (Assuming your data has appropriate timestamps)
        * "Detect anomalies in the 'vibration_sensor_X' column using Isolation Forest."
        * "What are the basic statistics for the 'power_consumption' column?"
        * "Plot the correlation matrix for all numeric columns."
        * "Get trend and seasonality summary for the 'temperature' column with period 24." (If seasonality tool is implemented)
5.  **Submit & View Results:**
    * Click "Submit" (or similar).
    * The agent will process your request, potentially calling one or more tools.
    * The response area will display the natural language answer from the LLM, along with any generated text, summaries, or links/images of plots.
    * The terminal where `app.py` is running might show logs of tool calls and LLM interactions, which can be useful for debugging.

## 6. Available Tools

ITISA's capabilities are defined by its library of Python tools, typically located in `itisa/tools/`. These tools are Python functions decorated or registered with the agent framework. Common categories include:

* **Data Management Tools (`data_processing_tool.py`):**
    * `load_timeseries_data(file_path: str)`: Loads data from a CSV.
    * `explore_dataset(dataset_name: str)`: Provides a summary of the dataset (columns, data types, missing values, basic stats).
    * `get_dataset_columns(dataset_name: str)`: Lists available columns in a loaded dataset.
* **Visualization Tools (`plotting_tool.py`):**
    * `plot_timeseries(dataset_name: str, columns: list[str], start_date: Optional[str], end_date: Optional[str])`: Generates time series plots.
    * `plot_anomalies(dataset_name: str, data_column: str, anomaly_column: str)`: Plots data with highlighted anomalies.
    * `create_correlation_heatmap(dataset_name: str, columns: Optional[list[str]])`: Generates a heatmap of correlations.
    * `plot_distribution(dataset_name: str, column: str)`: Plots the distribution of a data column.
* **Analysis Tools (`analysis_tool.py`):**
    * `calculate_statistics(dataset_name: str, columns: list[str])`: Calculates descriptive statistics.
    * `detect_anomalies_zscore(dataset_name: str, column: str, threshold: float)`: Detects anomalies using Z-score.
    * `detect_anomalies_isolation_forest(dataset_name: str, column: str, contamination: float)`: Detects anomalies using Isolation Forest.
    * `analyze_trend_seasonality(dataset_name: str, column: str, period: Optional[int])`: Performs time series decomposition.

*Note: The exact tool names, parameters, and their availability depend on the specific implementation in `itisa/tools/`. Refer to the docstrings within these files for the most accurate details.*

## 7. Configuration

Key configuration options for ITISA can usually be found directly in `itisa/app.py` or a dedicated config file if one exists.
This might include:
* Default Ollama model for ITISA.
* Ollama API endpoint.
* Parameters for the agent framework (e.g., `smolagents` settings like `max_steps`).
* Default parameters for tools (though many are expected to be inferred by the LLM from the query).

## 8. Customization and Extension

* **Adding New Tools:** This is the primary way to extend ITISA's capabilities.
    1.  Create a new Python function in one of the files in `itisa/tools/` (or a new file).
    2.  Ensure the function has a clear name, a detailed docstring explaining its purpose, arguments (with type hints), and what it returns. This docstring is crucial for the LLM to understand and use the tool correctly.
    3.  Decorate or register the function with the agent framework being used (e.g., `@tool` for `smolagents`).
    4.  Import and add the new tool to the list of tools provided to the agent in `itisa/app.py`.
* **Changing LLMs:** Modify the model name in `itisa/app.py` (or UI selector) to any compatible model available in your Ollama instance. Prioritize models known for good tool/function calling.
* **Supporting New Data Sources:** Modify or add new data loading tools to read from databases, APIs, or other file formats beyond CSV.
* **Improving Tool Logic:** Enhance the existing tools with more robust error handling, more sophisticated analytical techniques, or better visualization options.

## 9. Troubleshooting

Refer to the main **[Troubleshooting Guide](../troubleshooting.md)** for general issues (Ollama, Python, Gradio).
ITISA-specific tips:
* **Tool Not Called or Incorrect Tool Called:**
    * **Improve Docstrings:** The LLM heavily relies on tool names and docstrings. Make them very clear, specific, and ensure all parameters are well-described with types.
    * **Query Clarity:** Rephrase your natural language query to be more explicit about what you want.
    * **LLM Choice:** Some LLMs are much better at tool calling than others. Experiment with different models available in Ollama.
    * **Too Many Similar Tools:** If multiple tools seem to fit a query, the LLM might get confused. Try to make tool functionalities distinct.
* **Tool Called with Incorrect Parameters:**
    * Again, check docstrings for clarity on parameter names, types, and expected values.
    * Ensure your query provides enough information for the LLM to infer the parameters.
* **Errors During Tool Execution:**
    * Check the terminal output for Python errors originating from the tool's code. This can indicate a bug in the tool, an issue with the input data, or an unexpected parameter value.
    * Add more print statements or logging within the tool functions to debug.
* **Data Not Loaded Correctly:**
    * Verify the CSV format and that the `load_timeseries_data` tool (or equivalent) is parsing it as expected.
    * Check that the dataset name used in subsequent queries matches how it was loaded/named by the agent.

For further details on the underlying implementation, refer to the source code in the `itisa/` directory, especially `app.py` and the files within `itisa/tools/`.
