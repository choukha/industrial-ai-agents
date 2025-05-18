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
git clone https://github.com/choukha/industrial-ai-agents.git
cd industrial-ai-agents
```

### Step 2.2: Set Up a Python Virtual Environment

It's highly recommended to use a Python virtual environment to manage project dependencies and avoid conflicts with other Python projects.

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   ```
   This command creates a folder named `venv` in your project directory.

2. **Activate the virtual environment**:
   - Windows (Command Prompt/PowerShell):
     ```bash
     venv\Scripts\activate
     ```
   - macOS/Linux (bash/zsh):
     ```bash
     source venv/bin/activate
     ```
   Your terminal prompt should change to indicate the `(venv)` environment.

### Step 2.3: Install Ollama

Ollama is essential for running the Large Language Models locally.

1. Download installer from [ollama.com/download](https://ollama.com/download)
2. Install for your operating system
3. Verify installation:
   ```bash
   ollama list
   ```

### Step 2.4: Install Project Dependencies

With your virtual environment activated:

1. Install root dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Agent-specific dependencies:
   - IDOCA:
     ```bash
     pip install -r idoca/requirements.txt
     ```
   - ITISA:
     ```bash
     pip install -r itisa/requirements.txt
     ```
3. Package Installation for Development

   When running individual components or scripts directly, you might encounter `ModuleNotFoundError: No module named 'idoca'` errors. This happens because Python cannot locate the package modules. One of the ways to resolve this is to nstall the Package in Development Mode. This approach makes the package available system-wide while allowing you to modify the code:

   ```bash
   # From the root directory of the repository
   pip install -e .
   ```
   The `-e` flag installs the package in "editable" or "development" mode, creating a link to the source code rather than copying the files. This means any changes you make to the code will be immediately available without reinstallation.

3. **Docling for IDOCA(Optional)**:
   - Docker method:
     ```bash
     docker pull quay.io/docling-project/docling-serve:latest
     docker run -d -p 5001:5001 --name docling_server quay.io/docling-project/docling-serve:latest
     ```
   - Non-Docker:
     ```bash
     pip install docling
     ```
4. **Vector Database Setup (Optional)**
   - IDOCA supports multiple vector databases for the RAG pipeline. By default, it uses ChromaDB which requires no additional setup beyond what's included in `requirements.txt`.
If you plan to use Milvus, it requires a separate setup process as detailed below. FAISS, if chosen, is also included via `requirements.txt` and typically requires no further installation steps beyond that.

   - **For Milvus:**
      1.  Set up Milvus using Docker:
         Follow the official Milvus guide for [Standalone Docker Compose Installation](https://milvus.io/docs/install_standalone-windows.md)
      2.  Optional: Install Attu GUI for Milvus management: Download from [Attu GitHub Releases](https://github.com/zilliztech/attu/releases) and Connect to your Milvus instance (default: `localhost:19530`).

      For detailed instructions on Milvus setup, refer to the official Milvus documentation.



### Step 2.5: Pull Required LLM Models via Ollama

Download models using these commands:

1. Embedding model:
   ```bash
   ollama pull nomic-embed-text
   ```

2. General purpose LLMs:
   ```bash
   ollama pull qwen2:0.5b
   ollama pull qwen2:1.5b
   ollama pull phi3:mini
   ollama pull llama3:8b
   ollama pull qwen2:7b
   ```

3. Vision models (optional):
   ```bash
   ollama pull llava:7b
   ollama pull llava-llama3
   ```

Verify models:
```bash
ollama list
```

### Step 3: Running the Agents

**IDOCA**:
```bash
cd idoca
python main.py
```

**ITISA**:
```bash
cd itisa
python app.py
```

### (Optional) Step 4: Using Open WebUI

```bash
docker pull ghcr.io/open-webui/open-webui:latest
docker run -d -p 3000:8080 --add-host=host.docker.internal:host-gateway -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:latest
```

### Step 5: Troubleshooting

Common issues:
- Ollama service not running
- Missing models
- Dependency conflicts
- Insufficient RAM/VRAM

### Step 6: Next Steps
- Explore architecture documentation
- Check agent-specific guides
- Review example use cases