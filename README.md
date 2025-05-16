# Industrial AI Agents

A collection of practical, open-source AI agents for industrial applications, designed to run locally on modest hardware, enabling analysis of documents and time series data.

## Overview

This repository hosts implementations of industrial AI agents that leverage Large Language Models (LLMs) to perform useful tasks in industrial settings. The core philosophy is to provide solutions that can be deployed locally, utilize open-source components, and operate effectively on consumer-grade hardware. This approach addresses common industrial concerns regarding data privacy, security, and the cost of specialized AI infrastructure.

The project aims to serve as an educational resource and a practical toolkit for engineers, researchers, and developers looking to build and deploy AI agents for their specific industrial data and use cases.

### Current Agents:

* **IDOCA (Industrial Document Analysis Agent)**: Utilizes Retrieval-Augmented Generation (RAG) to analyze and answer questions about industrial documents, such as technical manuals, Standard Operating Procedures (SOPs), safety guidelines, and maintenance logs.
* **ITISA (Industrial Time Series Analysis Agent)**: Employs function/tool calling capabilities to analyze time series data from industrial processes, enabling natural language queries for tasks like anomaly detection, trend analysis, and data visualization.

## Key Features

* **Local First**: Designed for local deployment, ensuring data privacy and control.
* **Open Source Stack**: Built entirely with open-source LLMs, frameworks (LangChain, smolagents), and tools (Ollama, Chroma).
* **Practical & Industrial Focus**: Addresses real-world challenges in industrial data analysis.
* **Modular Design**: Components are designed to be adaptable and extensible, allowing for future enhancements and integration of new data types (e.g., images, video).
* **Educational & Tutorial-Oriented**: Comprehensive documentation and examples are provided to guide users in developing and customizing agents for their own industrial data.
* **Resource Efficient**: Optimized to run on modest hardware (e.g., modern CPU, 16-32GB RAM, optional consumer-grade GPU with ~4GB VRAM).

## Repository Structure
```
industrial-ai-agents/
├── README.md                   # Main repository documentation (This file)
├── CONTRIBUTING.md             # Contribution guidelines
├── LICENSE                     # License information (e.g., MIT License)
├── requirements.txt            # Root-level Python dependencies
├── setup.py                    # Optional: for packaging the project
├── docs/                       # Comprehensive documentation
│   ├── index.md                # Documentation landing page
│   ├── getting_started.md      # Quick start and setup guide
│   ├── architecture.md         # High-level architecture overview
│   ├── troubleshooting.md      # Common issues and solutions
│   ├── idoca/                  # IDOCA-specific documentation
│   │   └── index.md
│   ├── itisa/                  # ITISA-specific documentation
│   │   └── index.md
│   └── examples/               # Example notebooks and tutorials
│       └── index.md
├── idoca/                      # Industrial Document Analysis Agent (IDOCA)
│   ├── README.md               # IDOCA-specific overview and quick start
│   ├── init.py
│   ├── config.py               # Configuration settings
│   ├── data_processor.py       # Document processing logic
│   ├── rag_system.py           # RAG pipeline implementation
│   ├── agent.py                # Agentic RAG implementation with tools
│   ├── interface.py            # Gradio UI
│   ├── main.py                 # Main application entry point
│   └── requirements.txt        # IDOCA-specific dependencies
├── itisa/                      # Industrial Time Series Analysis Agent (ITISA)
│   ├── README.md               # ITISA-specific overview and quick start
│   ├── init.py
│   ├── app.py                  # Main application (Gradio UI and agent logic)
│   ├── tools/                  # Tool implementations for ITISA
│   │   ├── init.py
│   │   ├── data_processing_tool.py
│   │   ├── plotting_tool.py
│   │   └── analysis_tool.py
│   ├── streaming_utils.py      # Utilities for streaming responses (if used)
│   └── requirements.txt        # ITISA-specific dependencies
└── datasets/                   # Sample datasets for testing and examples
├── documents/              # Sample documents for IDOCA
├── time_series/            # Sample time series data for ITISA
└── generators/             # Scripts to generate synthetic data (optional)
```

## Getting Started

For detailed setup instructions, system prerequisites, and how to run the agents, please refer to the **[Getting Started Guide](docs/getting_started.md)**.

### Quick Installation Overview

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/choukha/industrial-ai-agents.git
    cd industrial-ai-agents
    ```
2.  **Set up Python Environment & Install Dependencies:**
    It's highly recommended to use a virtual environment.
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    pip install -r requirements.txt
    # Install agent-specific requirements if needed
    pip install -r idoca/requirements.txt
    pip install -r itisa/requirements.txt
    ```
3.  **Install Ollama:**
    Ollama is used for running LLMs locally.
    * **Linux/macOS:** `curl https://ollama.ai/install.sh | sh`
    * **Windows:** Download from [ollama.ai/download](https://ollama.ai/download)
4.  **Pull Required LLM Models via Ollama:**
    ```bash
    # For embedding (used by IDOCA)
    ollama pull nomic-embed-text
    # General purpose LLM (example, choose based on your hardware and needs)
    ollama pull qwen3:1.7b # A smaller model, good for testing
    ollama pull llama3:8b # A more capable model, requires more resources
    # For IDOCA, a model with good instruction following and reasoning is recommended.
    # For ITISA, a model with strong function/tool calling capabilities is preferred.
    ```
    Refer to the `docs/getting_started.md` for more model recommendations.

## System Requirements

* **OS:** Windows (with WSL2 for Docker/Ollama if preferred), macOS, or Linux.
* **Python:** 3.10+
* **CPU:** Modern processor (e.g., Intel i5/i7, AMD Ryzen 5/7) with at least 4 cores.
* **RAM:** Minimum 16GB, **32GB recommended** for running larger models or multiple components.
* **GPU (Optional but Recommended):**
    * NVIDIA: 4GB VRAM minimum for smaller models (e.g., Qwen3:1.7B, Phi-3-mini). 8GB+ VRAM recommended for 7B/8B models.
    * AMD: ROCm support via Ollama (experimental, check Ollama docs).
    * Apple Silicon: Metal support via Ollama.
* **Storage:** 10GB+ free space for models and datasets.
* **Software:** Git, Docker (optional, but useful for tools like Docling or Open WebUI).

## Documentation

* **[Getting Started](docs/getting_started.md)**
* **[Architecture Overview](docs/architecture.md)**
* **[IDOCA Documentation](docs/idoca/index.md)**
* **[ITISA Documentation](docs/itisa/index.md)**
* **[Troubleshooting Guide](docs/troubleshooting.md)**
* **[Examples and Tutorials](docs/examples/index.md)**

## Contributing

We welcome contributions! Please see our **[Contributing Guidelines](CONTRIBUTING.md)** for more details on how to submit issues, feature requests, or pull requests.

## License

This project is licensed under the **[MIT License](LICENSE)**.

## Acknowledgements

This project builds upon the foundational work of many open-source communities and researchers. We extend our gratitude to the developers and maintainers of:

* [Ollama](https://ollama.com/)
* [LangChain](https://www.langchain.com/) & [LangGraph](https://www.langchain.com/langgraph)
* [smolagents](https://github.com/huggingface/smolagents)
* [ChromaDB](https://www.trychroma.com/)
* [Docling Project](https://docling-project.github.io/docling/)
* The various open-source Large Language Models and Embedding Models available on platforms like Hugging Face.

This work is also part of the Master's Thesis **Building Practical Industrial AI Agents: Implementation Examples Using Large Language Models** by Choukha Ram, University of South-Eastern Norway, 2025.

## Disclaimer

The AI agents and tools provided in this repository are for educational and research purposes. While they aim for practical industrial application, they should be thoroughly tested and validated before use in production environments, especially in safety-critical systems.
