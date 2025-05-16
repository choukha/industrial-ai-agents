# Getting Started with Industrial AI Agents

Welcome! This guide will help you set up the Industrial AI Agents project on your local system, install necessary dependencies, and run the agents.

## 1. System Prerequisites

Before you begin, ensure your system meets the following requirements:

* **Operating System:**
    * Windows 10/11 (WSL2 recommended for Docker/Ollama, but Ollama also has a native Windows app)
    * macOS (Intel or Apple Silicon)
    * Linux (Most modern distributions)
* **Python:**
    * Version 3.10 or 3.11 is recommended. You can download Python from [python.org](https://www.python.org/downloads/).
* **Hardware:**
    * **CPU:** A modern multi-core processor (e.g., Intel Core i5/i7 4th gen or newer, AMD Ryzen 5/7 series or newer) with at least 4 cores.
    * **RAM:** Minimum 16GB. **32GB is strongly recommended** for a smoother experience, especially when running larger LLMs or multiple components simultaneously.
    * **GPU (Optional but Highly Recommended for LLM performance):**
        * **NVIDIA:** GPUs with at least 4GB VRAM (e.g., GTX 1650 Super, RTX T1200, RTX 2060 or better). 8GB+ VRAM is better for larger models (7B+ parameters). Ensure you have the latest NVIDIA drivers.
        * **AMD:** Recent AMD GPUs may work with Ollama via ROCm on Linux. Check Ollama documentation for compatibility.
        * **Apple Silicon (M1/M2/M3):** Supported by Ollama via Metal.
    * **Storage:** At least 20GB of free disk space for the repository, dependencies, LLM models, and datasets. LLM models can be several gigabytes each.
* **Software:**
    * **Git:** For cloning the repository. Download from [git-scm.com](https://git-scm.com/).
    * **Ollama:** For running LLMs locally. Download from [ollama.com/download](https://ollama.com/download).
    * **Docker Desktop (Optional but Recommended):** Useful for running auxiliary services like Docling or Open WebUI. Download from [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/).
        * If on Windows, Docker Desktop typically requires WSL2.

## 2. Installation Steps

Follow these steps to get the project up and running:

### Step 2.1: Clone the Repository

Open your terminal or command prompt and clone the `industrial-ai-agents` repository from GitHub:

```bash
git clone [https://github.com/choukha/industrial-ai-agents.git](https://github.com/choukha/industrial-ai-agents.git)
cd industrial-ai-agents
```

### Step 2.2: Set Up a Python Virtual Environment
It's highly recommended to use a Python virtual environment to manage project dependencies and avoid conflicts with other Python projects.Create a virtual environment:Navigate to the cloned repository's root directory (industrial-ai-agents) and run:python -m venv venv
This command creates a folder named venv in your project directory.Activate the virtual environment:On Windows (Command Prompt/PowerShell):venv\Scripts\activate
On macOS and Linux (bash/zsh):source venv/bin/activate
Once activated, your terminal prompt should change to indicate that you are in the (venv) environment.Step 2.3: Install OllamaOllama is essential for running the Large Language Models locally.Go to ollama.com/download.Download the installer for your operating system (Windows, macOS, or Linux) and follow the installation instructions.After installation, Ollama typically runs as a background service. You can verify it's running by opening a terminal and typing:ollama list
This command should show an empty list if no models are downloaded yet, or a list of already pulled models.Step 2.4: Install Project DependenciesWith your virtual environment activated, install the Python packages required by the project.Install root-level dependencies:These are common dependencies shared across the project.pip install -r requirements.txt
Install agent-specific dependencies:Each agent (IDOCA and ITISA) has its own specific dependencies.For IDOCA:pip install -r idoca/requirements.txt
For ITISA:pip install -r itisa/requirements.txt
(Optional) Install Docling for IDOCA (Recommended for best document parsing):Docling provides advanced PDF parsing. The easiest way to use it is via its Docker container.docker pull quay.io/docling-project/docling-serve:latest
# To run Docling server (IDOCA will connect to this):
docker run -d -p 5001:5001 --name docling_server quay.io/docling-project/docling-serve:latest
Alternatively, you can try installing the docling Python package if you prefer not to use Docker:pip install docling
Refer to the IDOCA documentation for more details on Docling setup.Step 2.5: Pull Required LLM Models via OllamaYou need to download the LLM models that the agents will use. Open your terminal and use the ollama pull command.Embedding Model (Essential for IDOCA):ollama pull nomic-embed-text
This model is used to create vector embeddings of your documents.General Purpose LLMs (for reasoning and generation):The choice of LLM depends on your hardware capabilities and the specific agent.Smaller, faster models (good for testing, less VRAM needed):ollama pull qwen2:0.5b # Very small, good for initial testing
ollama pull qwen2:1.5b
ollama pull phi3:mini    # ~3.8B parameters, good balance
Medium, more capable models (require more VRAM/RAM):ollama pull llama3:8b    # ~8B parameters, excellent all-rounder
ollama pull qwen2:7b     # ~7B parameters, strong capabilities
Models with Vision (Optional, for future extensions or if IDOCA uses image processing):ollama pull llava:7b # LLaVA based on Vicuna 7B
ollama pull llava-llama3 # LLaVA based on Llama3
You can list your downloaded models with ollama list. For specific model recommendations for each agent, refer to their respective documentation pages:IDOCA Model ConfigurationITISA Model Configuration3. Running the AgentsOnce everything is installed, you can run the agents. Ensure Ollama is running in the background.Running IDOCA (Industrial Document Analysis Agent)Navigate to the IDOCA directory:cd idoca
Ensure your virtual environment is active and Ollama is running.If you are using Docling via Docker, make sure the docling_server container is running.Run the main application:python main.py
This will start the Gradio web interface. Open your web browser and go to the URL displayed in the terminal (usually http://127.0.0.1:7860 or http://localhost:7860).Follow the instructions in the IDOCA UI to upload documents, process them, and ask questions. For more details, see IDOCA Usage Guide.Running ITISA (Industrial Time Series Analysis Agent)Navigate to the ITISA directory:cd itisa
(If you are in the idoca directory, you might need to do cd ../itisa)Ensure your virtual environment is active and Ollama is running.Run the main application:python app.py
This will start the Gradio web interface. Open your web browser and go to the URL displayed in the terminal (usually http://127.0.0.1:7860 or http://localhost:7860).Note: If IDOCA is already running on port 7860, ITISA might try to use a different port or fail. Stop one agent before starting the other if they conflict.Follow the instructions in the ITISA UI to upload time series data (CSV files) and ask analytical questions. For more details, see ITISA Usage Guide.4. (Optional) Using Open WebUIOpen WebUI provides a more feature-rich chat interface for interacting with LLMs running via Ollama. It also has built-in RAG capabilities.Install Open WebUI (using Docker is recommended):docker pull ghcr.io/open-webui/open-webui:latest
docker run -d -p 3000:8080 --add-host=host.docker.internal:host-gateway -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:latest
Open your browser and go to http://localhost:3000.Create an admin account.Connect to your Ollama instance (it should auto-detect if Ollama is running on the default http://host.docker.internal:11434 or http://localhost:11434).You can then chat with your Ollama models or use Open WebUI's "Documents" feature to upload files and perform RAG.5. TroubleshootingIf you encounter any issues during setup or execution, please refer to the Troubleshooting Guide.Common initial issues might include:Ollama not running or accessible: Ensure the Ollama service/application is started. Check firewall settings if accessing from a different machine or container.Model not found: Double-check that you have pulled the required models using ollama pull <model_name>.Python dependencies conflicts: Ensure you are using a clean virtual environment.Insufficient RAM/VRAM: If the application is very slow or crashes, you might be running out of memory. Try using smaller LLMs or closing other resource-intensive applications.6. Next StepsExplore the Architecture Overview to understand how the agents are designed.Dive into the specific documentation for IDOCA and ITISA.Check out the Examples and Tutorials for practical use cases.Consider Contributing to the project.Happy experimenting with Industrial AI Agents!