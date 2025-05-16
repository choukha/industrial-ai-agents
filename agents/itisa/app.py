# itisa/app.py
import os
import argparse
from pathlib import Path
import dotenv
import gradio as gr
import logging # For logging Ollama model fetching
from typing import Optional
# Attempt to import ollama, will be used for fetching model list
try:
    import ollama
except ImportError:
    ollama = None # Will be checked later

# Load environment variables from .env file
dotenv.load_dotenv()

from smolagents import CodeAgent, LiteLLMModel

# Import your custom tools
from tools.data_processing_tool import load_and_describe_data
from tools.plotting_tool import plot_aggregated_time_series, plot_correlation_matrix
from tools.analysis_tool import detect_anomalies_iforest_and_plot, get_trend_seasonality_summary

# Import streaming utilities
from streaming_utils import stream_agent_responses

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure data and results directories exist
DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_OLLAMA_MODELS_FALLBACK = [
    "llama3:latest",
    "mistral:latest",
    "qwen:latest",
    "gemma:latest",
]

def get_ollama_models_list() -> list:
    """
    Queries the Ollama API to get a list of locally available models.
    Uses the 'ollama' Python package.
    """
    if ollama is None:
        logger.warning("'ollama' package not installed. Falling back to default model list.")
        return DEFAULT_OLLAMA_MODELS_FALLBACK
    try:
        models_response = ollama.list() # This is a dict like {'models': [{'name': 'model1:tag', ...}]}
        
        # Correctly access model names
        model_names = [model_info['name'] for model_info in models_response.get('models', [])]
        
        model_names.sort()
        if not model_names:
            logger.warning("No models returned from ollama.list(). Falling back to default list.")
            return DEFAULT_OLLAMA_MODELS_FALLBACK
        logger.info(f"Successfully fetched Ollama models: {model_names}")
        return model_names
    except Exception as e:
        logger.error(f"Error getting Ollama models: {e}. Falling back to default list.")
        return DEFAULT_OLLAMA_MODELS_FALLBACK

AVAILABLE_OLLAMA_MODELS = get_ollama_models_list()


def create_itisa_agent(
    ollama_model_name: str = "llama3:latest",
    ollama_base_url: str = "http://localhost:11434"
):
    """
    Creates and configures the Industrial Time Series Analysis (ITISA) agent.
    """
    llm_model_id = f"ollama_chat/{ollama_model_name}" # Using ollama_chat for better structured output
    
    llm = LiteLLMModel(
        model_id=llm_model_id,
        api_base=ollama_base_url,
        # Consider adding num_ctx and max_tokens if needed, e.g.:
        # num_ctx=8192,
        # max_tokens=2048,
        temperature=0.1 # Lower temperature for more deterministic tool usage
    )

    itisa_tools = [
        load_and_describe_data,
        plot_aggregated_time_series,
        plot_correlation_matrix,
        detect_anomalies_iforest_and_plot,
        get_trend_seasonality_summary
    ]

    additional_imports = [
        "pandas", "numpy", "plotly.graph_objects", "json", "pathlib", "os", "statsmodels.tsa.seasonal"
    ]

    agent = CodeAgent(
        tools=itisa_tools,
        model=llm,
        additional_authorized_imports=additional_imports,
        name="ITISA_Agent",
        description=(
            "An AI agent specialized in industrial time series analysis. "
            "It can load data, generate plots, detect anomalies, and summarize trends/seasonality. "
            "Always start by loading data using 'load_and_describe_data' tool if a new file is provided or mentioned. "
            "When a plot is generated, clearly state the full path to the HTML file for the user."
        ),
        max_steps=15, # Increased max steps for potentially complex queries
        verbosity_level=1
    )
    return agent

# --- Gradio UI Setup ---
# Global agent instance for UI mode - managed by session state in Gradio
# This helps maintain conversation context if `reset=False` is used in agent.run()

def handle_chat_interaction(
    message: str,
    history: list, # List of gr.ChatMessage objects if chatbot type is "messages"
    uploaded_file_path: Optional[str],
    selected_ollama_model: str,
    # request: gr.Request # Access request for session ID if needed for advanced state
    # For simplicity, we'll manage agent per call for now, or rely on smolagents' internal memory if reset=False
):
    """
    Handles the chat interaction with the ITISA agent, now with streaming.
    History is expected to be a list of gr.ChatMessage objects.
    """
    current_agent = create_itisa_agent(ollama_model_name=selected_ollama_model)

    full_query = message
    if uploaded_file_path:
        full_query = (
            f"An input file has been provided at path: '{uploaded_file_path}'. "
            f"Please use this file for analysis. User's request: {message}"
        )

    # Add user's message to history
    history.append(gr.ChatMessage(role="user", content=message if not uploaded_file_path else full_query))
    yield history, "", None # Update chat, clear textbox, clear file upload path for next turn

    # Stream agent's response
    try:
        for msg_chunk in stream_agent_responses(current_agent, task=full_query, reset_agent_memory=False):
            history.append(msg_chunk)
            yield history, "", None # Update chat, keep textbox and file upload clear
    except Exception as e:
        logger.error(f"Error during agent interaction: {e}", exc_info=True)
        history.append(gr.ChatMessage(role="assistant", content=f"An error occurred: {str(e)}"))
        yield history, "", None


def build_gradio_ui(default_ollama_model: str):
    """Builds and returns the Gradio UI for the ITISA agent."""
    with gr.Blocks(theme=gr.themes.Soft(primary_hue=gr.themes.colors.blue, secondary_hue=gr.themes.colors.sky)) as demo:
        gr.Markdown("# Industrial Time Series Analysis Agent (ITISA)")
        gr.Markdown(
            "Upload a CSV file, select an Ollama model, and ask questions about your time series data. "
            "The agent can load data, plot time series, calculate correlations, detect anomalies, and analyze trends/seasonality."
        )

        with gr.Row():
            with gr.Column(scale=1):
                file_upload = gr.File(label="Upload Time Series CSV File", file_types=[".csv"])
                
                # Use dynamically fetched models, with a fallback if fetching fails
                ollama_model_choices = AVAILABLE_OLLAMA_MODELS
                if not ollama_model_choices: # Should not happen due to fallback in get_ollama_models_list
                    ollama_model_choices = DEFAULT_OLLAMA_MODELS_FALLBACK
                
                # Ensure default_ollama_model is in choices, otherwise pick first
                current_default_model = default_ollama_model
                if default_ollama_model not in ollama_model_choices and ollama_model_choices:
                    current_default_model = ollama_model_choices[0]
                elif not ollama_model_choices: # No models at all
                    current_default_model = "No models available"


                ollama_model_selector = gr.Dropdown(
                    label="Select Ollama Model",
                    choices=ollama_model_choices,
                    value=current_default_model,
                    interactive=True if ollama_model_choices and current_default_model != "No models available" else False
                )
                gr.Markdown(
                    "**Example Queries:**\n"
                    "- Load and describe the data from the uploaded file.\n"
                    "- Plot the aggregated time series with daily ('D') aggregation.\n"
                    "- Plot the correlation matrix for all numeric columns.\n"
                    "- Detect anomalies in the 'temperature' column using Isolation Forest.\n"
                    "- Get trend and seasonality summary for the 'pressure' column (e.g., period 24 for hourly data, or 7 for daily data)."
                )
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="ITISA Chat",
                    height=600,
                    type="messages", # Important for streaming gr.ChatMessage objects
                    avatar_images=(None, "https://raw.githubusercontent.com/huggingface/smolagents/main/docs/assets/mascot_smol.png") # Smolagent icon
                )
                chat_input = gr.Textbox(label="Your Query", placeholder="Type your query here and press Enter or click Send")
                submit_button = gr.Button("Send", variant="primary")

        uploaded_file_state = gr.State()

        file_upload.upload(
            fn=lambda file_obj: file_obj.name if file_obj else None,
            inputs=[file_upload],
            outputs=[uploaded_file_state],
            show_progress="hidden"
        )
        
        # Combine inputs for action: message, history, file_path, model_name
        submit_button.click(
            handle_chat_interaction,
            inputs=[chat_input, chatbot, uploaded_file_state, ollama_model_selector],
            outputs=[chatbot, chat_input, uploaded_file_state] # Clear input and uploaded_file_state
        )
        chat_input.submit(
            handle_chat_interaction,
            inputs=[chat_input, chatbot, uploaded_file_state, ollama_model_selector],
            outputs=[chatbot, chat_input, uploaded_file_state] # Clear input and uploaded_file_state
        )
    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Industrial Time Series Analysis Agent (ITISA) with Gradio UI")
    parser.add_argument(
        "--ollama-model",
        type=str,
        default=AVAILABLE_OLLAMA_MODELS[0] if AVAILABLE_OLLAMA_MODELS else DEFAULT_OLLAMA_MODELS_FALLBACK[0], # Default to first available or fallback
        help="Default Ollama model to use (e.g., 'llama3:latest')."
    )
    parser.add_argument(
        "--no-ui", action="store_true", help="Run in command-line mode (no UI)"
    )
    parser.add_argument(
        "--query", type=str, help="Query to process in command-line mode (requires --no-ui)"
    )
    parser.add_argument(
        "--share", action="store_true", help="Enable Gradio sharing (creates a public link)"
    )
    parser.add_argument(
        "--server_name", type=str, default="0.0.0.0", help="Gradio server name"
    )
    parser.add_argument(
        "--server_port", type=int, default=7860, help="Gradio server port"
    )
    args = parser.parse_args()

    if args.no_ui:
        if args.query:
            logger.info(f"Running in command-line mode with model: {args.ollama_model}")
            agent = create_itisa_agent(ollama_model_name=args.ollama_model)
            # In CLI mode, streaming is usually handled by printing step logs.
            # agent.run already prints logs if verbosity > 0.
            # For a cleaner final output:
            final_result = "Running..." # Placeholder
            try:
                # Non-streamed run for CLI to get final answer, verbosity will print steps
                # Or iterate through stream_agent_responses and print each part.
                print(f"User Query: {args.query}")
                print("Agent thinking...")
                for step in stream_agent_responses(agent, task=args.query, reset_agent_memory=True):
                    # In CLI, we can print the content of the ChatMessage
                    if isinstance(step.content, dict) and 'path' in step.content: # File content
                        print(f"Assistant ({step.metadata.get('message_type', '') if step.metadata else ''}): File at {step.content['path']}")
                    else:
                        print(f"Assistant ({step.metadata.get('message_type', '') if step.metadata else ''}): {step.content}")
                    if step.metadata and step.metadata.get("message_type") == "final_answer":
                        final_result = step.content
                print("\n--- End of Agent Response ---")

            except Exception as e:
                logger.error(f"CLI agent run failed: {e}", exc_info=True)
                print(f"Error: {e}")
        else:
            print("Error: --query argument is required when using --no-ui.")
    else:
        logger.info(f"Starting ITISA Gradio UI with default model: {args.ollama_model}")
        logger.info(f"Available Ollama models for dropdown: {AVAILABLE_OLLAMA_MODELS}")
        logger.info(f"Access the UI at http://{args.server_name}:{args.server_port}")
        if args.share:
            logger.info("Gradio share link will be generated.")

        ui = build_gradio_ui(default_ollama_model=args.ollama_model)
        ui.launch(server_name=args.server_name, server_port=args.server_port, share=args.share)
