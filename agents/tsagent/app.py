import os
import argparse
from pathlib import Path
import dotenv

# Load environment variables
dotenv.load_dotenv()

# Import smolagents components
from smolagents import CodeAgent, GradioUI, TransformersModel, HfApiModel, LiteLLMModel
# from smolagents.loggers import HTMLLogger

# Import tools
from tools import (
    load_timeseries_data,
    explore_dataset,
    generate_synthetic_data,
    create_time_series_plot,
    create_correlation_heatmap,
    create_distribution_plot,
    create_lag_plot,
    create_seasonality_plot,
    extract_statistical_features,
    detect_anomalies_zscore,
    detect_anomalies_isolation_forest,
    analyze_anomalies
)

def create_agent(
    model_name: str = "llama3", 
    use_local_model: bool = False,
    use_ollama: bool = False,
    ollama_base_url: str = "http://localhost:11434",
    executor_type: str = "local"
):
    """
    Create a time series analysis agent.
    
    Args:
        model_name: Name of the model to use
        use_local_model: Whether to use a local model with transformers
        use_ollama: Whether to use an Ollama-hosted model
        ollama_base_url: Base URL for the Ollama API
        executor_type: Execution environment ('local', 'docker', or 'e2b')
        
    Returns:
        Initialized CodeAgent
    """
    # Create data and results directories if they don't exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Initialize LLM
    if use_ollama:
        # Use Ollama-hosted model
        model = LiteLLMModel(
            model_id=f"ollama_chat/{model_name}",
            api_base=ollama_base_url,
            num_ctx=16384  # Ollama default is 2048 which is too small for complex tasks
        )
    elif use_local_model:
        # Use local model with transformers
        model = TransformersModel(model_id=model_name)
    else:
        # Use HF Inference API
        hf_token = os.getenv("HF_TOKEN", None)
        if not hf_token:
            print("Warning: HF_TOKEN not set. Using default free model.")
        
        model = HfApiModel(model_id=model_name, token=hf_token)
    
    # Define all tools
    tools = [
        load_timeseries_data,
        explore_dataset,
        generate_synthetic_data,
        create_time_series_plot,
        create_correlation_heatmap,
        create_distribution_plot,
        create_lag_plot,
        create_seasonality_plot,
        extract_statistical_features,
        detect_anomalies_zscore,
        detect_anomalies_isolation_forest,
        analyze_anomalies
    ]
    
    # Additional imports that may be needed
    additional_imports = [
        "pandas", 
        "numpy", 
        "matplotlib.pyplot", 
        "seaborn", 
        "json",
        "os", 
        "pathlib", 
        "datetime"
    ]
    
    # Initialize the agent
    agent = CodeAgent(
        tools=tools,
        model=model,
        additional_authorized_imports=additional_imports,
        name="TSAgent",
        description="A specialized agent for time series analysis with focus on anomaly detection",
        max_steps=15,
        executor_type=executor_type,
        verbosity_level=1
    )
    
    return agent

def start_ui(agent):
    """Launch the Gradio UI for the agent."""
    ui = GradioUI(
        agent=agent,
        file_upload_folder="data"
    )
    
    # Start the UI
    ui.launch(share=False)

def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="Time Series Analysis Agent")
    
    # Existing arguments
    parser.add_argument("--model", type=str, default="llama3", 
                      help="Name of the model to use")
    
    parser.add_argument("--local", action="store_true", 
                      help="Use a local model with transformers")
    
    # Add new Ollama arguments
    parser.add_argument("--ollama", action="store_true",
                      help="Use a locally-hosted Ollama model")
                      
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434",
                      help="Base URL for the Ollama API")
    
    parser.add_argument("--executor", type=str, default="local", 
                      choices=["local", "docker", "e2b"],
                      help="Code execution environment")
    
    parser.add_argument("--no-ui", action="store_true",
                      help="Run in command-line mode (no UI)")
    
    parser.add_argument("--query", type=str,
                      help="Query to process in command-line mode")
    
    args = parser.parse_args()
    
    # Create the agent
    agent = create_agent(
        model_name=args.model,
        use_local_model=args.local,
        use_ollama=args.ollama,
        ollama_base_url=args.ollama_url,
        executor_type=args.executor
    )
    
    # Run in command-line mode or start UI
    if args.no_ui:
        if args.query:
            # Create HTML logger
            # logger = HTMLLogger(path="results/logs", agent=agent)
            
            # Run query
            result = agent.run(args.query)
            print("\nResult:")
            print(result)
            
            # Save log
            # logger.save()
            # print(f"Log saved to: {logger.path}")
        else:
            print("Error: --query argument required when using --no-ui")
    else:
        # Start the UI
        start_ui(agent)

if __name__ == "__main__":
    main()