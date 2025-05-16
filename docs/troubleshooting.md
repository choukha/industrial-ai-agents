# Troubleshooting Guide

This guide provides solutions to common issues you might encounter when setting up or using the Industrial AI Agents.

## General Issues

### 1. Installation Problems

**Issue: `pip install -r requirements.txt` fails for certain packages.**
* **Solution 1: Python Version:** Ensure you are using Python 3.10 or 3.11. Some dependencies might have issues with very new or very old Python versions. You can check your version with `python --version`.
* **Solution 2: System Dependencies:** Some Python packages require system-level libraries.
    * On Linux, you might need to install development headers (e.g., `python3-dev`, `build-essential`).
    * On macOS, ensure you have Xcode Command Line Tools installed (`xcode-select --install`).
    * On Windows, some packages might need Microsoft C++ Build Tools.
* **Solution 3: Conflicting Packages:** If you have an existing complex Python environment, try creating a fresh virtual environment as described in the [Getting Started Guide](getting_started.md).
* **Solution 4: Network Issues:** Ensure you have a stable internet connection. If you are behind a corporate proxy, you might need to configure `pip` to use the proxy:
    ```bash
    pip install --proxy http://your_proxy_server:port -r requirements.txt
    ```
* **Solution 5: Specific Package Errors:** If a specific package fails, search online for the error message related to that package and your OS. It might be a known issue with a specific solution.

**Issue: "ModuleNotFoundError: No module named 'X'" even after installation.**
* **Solution 1: Virtual Environment Not Activated:** Ensure your Python virtual environment (`venv`) is activated. Your terminal prompt should show `(venv)`. If not, activate it using the instructions in the [Getting Started Guide](getting_started.md).
* **Solution 2: Incorrect Python Interpreter:** If you are using an IDE (like VS Code or PyCharm), ensure it's configured to use the Python interpreter from your virtual environment.
* **Solution 3: Installation in Wrong Environment:** You might have accidentally installed packages globally or in a different virtual environment. Deactivate and reactivate your project's `venv` and try reinstalling.

### 2. Ollama and LLM Issues

**Issue: Ollama service not running or agents can't connect.**
* **Solution 1: Start Ollama:**
    * On Windows/macOS: Ensure the Ollama application is running.
    * On Linux: Ollama usually runs as a systemd service. Check its status: `sudo systemctl status ollama`. If not running, start it: `sudo systemctl start ollama`.
    * You can verify by opening a terminal and typing `ollama list`. If it hangs or gives an error, Ollama is not running correctly.
* **Solution 2: Firewall:** Your firewall might be blocking connections to Ollama (default port `11434`). Ensure the port is open for localhost access.
* **Solution 3: Ollama API URL:** If agents are configured with a custom Ollama URL, ensure it's correct (e.g., `http://localhost:11434`).
* **Solution 4: Docker Networking (if Ollama is in Docker or agent is in Docker):**
    * If Ollama is running on the host and the agent is in Docker, the agent might need to connect to `http://host.docker.internal:11434` instead of `http://localhost:11434`.
    * If both are in Docker containers on the same Docker network, they can use container names.

**Issue: `ollama pull <model_name>` fails.**
* **Solution 1: Internet Connection:** Check your internet connection.
* **Solution 2: Disk Space:** Ensure you have enough free disk space. LLM models can be several gigabytes.
* **Solution 3: Model Name:** Double-check the model name for typos. Refer to the [Ollama Library](https://ollama.com/library) for available models.
* **Solution 4: Ollama Server Issues:** Occasionally, the Ollama model repository might have temporary issues. Try again later.
* **Solution 5: Proxy Issues:** If behind a proxy, ensure Ollama is configured to use it. This might involve setting `HTTP_PROXY` and `HTTPS_PROXY` environment variables before running Ollama.

**Issue: LLM responses are very slow or application crashes (Out of Memory).**
* **Solution 1: Insufficient RAM/VRAM:** This is the most common cause.
    * **RAM:** Running LLMs, especially larger ones, consumes significant RAM. Close other memory-intensive applications. If you have 16GB RAM, stick to smaller models (e.g., 1.5B to 4B parameter models).
    * **VRAM (GPU):** If you have a GPU, Ollama will try to offload layers to it. If VRAM is insufficient, it will use more CPU/RAM, slowing things down.
        * Monitor GPU memory usage (e.g., `nvidia-smi` on Linux, Task Manager on Windows).
        * Try smaller models (e.g., `qwen2:1.5b` instead of `llama3:8b`).
        * Some models have different quantization levels available via their Modelfile (though Ollama usually picks a sensible default). Smaller quantizations use less VRAM but might slightly reduce quality.
* **Solution 2: Model Choice:** Larger models are slower. Experiment with smaller models listed in the [Getting Started Guide](getting_started.md) to find a balance between performance and capability for your hardware.
* **Solution 3: CPU Overload:** If no GPU or limited GPU offload, the CPU will do most of the work. Ensure your CPU is not thermal throttling and has adequate cooling.
* **Solution 4: Concurrent Requests:** If multiple users or processes are hitting the Ollama server simultaneously, performance will degrade.

**Issue: Model produces nonsensical or irrelevant answers.**
* **Solution 1: Wrong Model for the Task:** Some models are better at specific tasks (e.g., instruction-following, coding, RAG) than others. Ensure you're using a recommended model for the agent (IDOCA or ITISA).
* **Solution 2: Prompting:** The quality of the input prompt significantly affects the output. Try rephrasing your query or providing more context.
* **Solution 3: Temperature/Context Length:** These are advanced settings usually managed by the agent code or Ollama's defaults. For RAG, ensure the context provided isn't too long for the model's context window.
* **Solution 4: Insufficient Context (RAG):** For IDOCA, if answers are irrelevant, the RAG system might not be retrieving the correct document chunks. See IDOCA-specific troubleshooting.

## IDOCA (Document Analysis Agent) Specific Issues

**Issue: Documents not parsing correctly or errors during processing.**
* **Solution 1: File Format:** Ensure the document format is supported (PDF, DOCX, TXT, CSV are generally supported by underlying libraries). Scanned PDFs (images of text) require OCR and are much harder to process accurately.
* **Solution 2: Corrupted Files:** The document might be corrupted. Try opening it in a native application.
* **Solution 3: Complex Layouts/Tables (PDFs):**
    * **Docling:** For complex PDFs with tables and intricate layouts, using `Docling` (via Docker or Python package) is highly recommended as it's designed to preserve structure. Ensure the Docling server is running if using the Docker method and IDOCA is configured to connect to it.
    * If not using Docling, simpler parsers like `PyMuPDF` might struggle.
* **Solution 4: Large Files:** Very large files might consume too much memory during parsing. Try splitting them if possible.
* **Solution 5: Docling Server Not Reachable:** If using Docling via Docker, ensure the container is running (`docker ps`) and accessible on the configured port (default `5001`).

**Issue: Poor RAG performance (irrelevant answers, not finding information).**
* **Solution 1: Chunking Strategy:** The way documents are split into chunks is critical.
    * If chunks are too small, context is lost. If too large, they might be too noisy or exceed model context limits.
    * Experiment with `chunk_size` and `chunk_overlap` parameters in `idoca/data_processor.py` or the RAG system configuration.
    * Using Docling's layout-aware chunking can improve this for PDFs.
* **Solution 2: Embedding Model:**
    * Ensure `nomic-embed-text` (or your chosen model) is pulled in Ollama and correctly configured in IDOCA.
    * A different embedding model might yield better results for your specific document types.
* **Solution 3: Vector Database Issues:**
    * Ensure ChromaDB (or your chosen DB) is initialized correctly.
    * If you reprocess documents, ensure the old collection is cleared or a new one is used to avoid stale data. IDOCA's UI usually has an option to "Clear All Data & Session State."
* **Solution 4: Number of Retrieved Chunks (Top-k):** The RAG system retrieves the 'k' most relevant chunks. If 'k' is too low, relevant info might be missed. If too high, too much noise might be fed to the LLM. This is often configurable in `idoca/rag_system.py`.
* **Solution 5: LLM for Generation:** The LLM's ability to synthesize answers from context matters. A more capable LLM might provide better answers even with the same retrieved context.

**Issue: "Agent N/A" or errors related to agent initialization in IDOCA.**
* **Solution:** This usually means the RAG system or the LangGraph agent wasn't initialized. In the IDOCA UI, after uploading and processing files, ensure you click the "Initialize/Re-Initialize RAG & Agent" button.

## ITISA (Time Series Analysis Agent) Specific Issues

**Issue: CSV file won't load or data parsing errors.**
* **Solution 1: CSV Format:**
    * Ensure the file is a valid CSV (comma-separated, though some tools might handle other delimiters if specified).
    * Check for consistent headers and data types in columns.
    * Special characters in headers or data can sometimes cause issues.
* **Solution 2: Timestamp Column:**
    * ITISA tools often expect a clearly identifiable timestamp/datetime column. Ensure it's present and in a format that Pandas can parse (e.g., `YYYY-MM-DD HH:MM:SS`).
    * The agent or tools might try to auto-detect it, but issues can arise with ambiguous formats.
* **Solution 3: File Path/Access:** Ensure the agent has the correct path to the CSV and read permissions. If uploading via Gradio, this is usually handled.

**Issue: LLM not selecting the correct tool or providing incorrect parameters.**
* **Solution 1: Tool Descriptions (Docstrings):** This is critical. The LLM relies heavily on the tool's name and its docstring (description, arguments, types) to understand what it does and how to use it.
    * Ensure docstrings in `itisa/tools/` are clear, concise, and accurately describe each parameter.
    * Use type hints for parameters.
* **Solution 2: LLM Capability:** Not all LLMs are equally good at function/tool calling. Models specifically fine-tuned for instruction-following or tool use (e.g., `qwen2` series, some `llama3` variants) tend to perform better. Try a different LLM via Ollama.
* **Solution 3: Query Phrasing:** Be explicit in your natural language query. Sometimes, rephrasing the request can help the LLM understand the intent better.
    * Example: Instead of "Analyze data," try "Plot the temperature column from dataset 'my_data.csv'."
* **Solution 4: Tool Specificity:** If tools are too generic or overlap too much in functionality, the LLM might get confused. Ensure each tool has a distinct purpose.

**Issue: Visualization/Plots not showing up or errors during plotting.**
* **Solution 1: Matplotlib Backend:** Sometimes, plotting libraries like Matplotlib require a specific backend, especially when running in a non-interactive server environment. Gradio usually handles this, but issues can occur.
* **Solution 2: Data Issues for Plotting:** The selected data column might be non-numeric, or the data might have issues (e.g., all NaNs) that prevent plotting.
* **Solution 3: Tool Implementation:** There might be a bug in the specific plotting tool in `itisa/tools/plotting_tool.py`. Check the terminal for Python errors.

## UI (Gradio) Issues

**Issue: Gradio interface not loading or `OSError: [Errno 98] Address already in use`.**
* **Solution 1: Port Conflict:** Another application (or another instance of an agent) is already using the default Gradio port (usually `7860`).
    * Stop the other application.
    * Or, modify the `app.launch()` or `ui.launch()` line in `main.py` (for IDOCA) or `app.py` (for ITISA) to specify a different port: `ui.launch(server_name="0.0.0.0", server_port=7861)`.
* **Solution 2: Browser Issues:** Try clearing your browser cache or using a different browser. Check the browser's developer console (F12) for JavaScript errors.
* **Solution 3: Gradio Version:** Ensure you have a compatible version of Gradio installed as per `requirements.txt`.

**Issue: File upload not working in Gradio.**
* **Solution 1: File Size Limits:** Gradio might have default file size limits. For very large files, this could be an issue (though usually more relevant for public deployments).
* **Solution 2: Permissions:** Ensure the application has write permissions to any temporary directories it might use.

## Advanced Debugging

1.  **Check Terminal Output:** Always keep an eye on the terminal where you launched the Python application. Detailed error messages and stack traces are usually printed there.
2.  **Enable Verbose Logging:**
    * Many libraries (LangChain, Ollama client) have verbosity settings.
    * You can add more `print()` statements or use Python's `logging` module in the agent code to trace execution flow and variable states.
    * Example for LangChain:
        ```python
        import langchain
        langchain.debug = True # or langchain.verbose = True
        ```
3.  **Test Components Individually:**
    * Can Ollama serve responses via its CLI or a simple `curl` command?
    * Can you load and parse a document using just the `Docling` or `Unstructured` library in a separate Python script?
    * Can you perform a similarity search directly with `ChromaDB`?
    * Can you run one of ITISA's tools as a standalone Python function?
4.  **Simplify the Scenario:**
    * Use a very small document or CSV file.
    * Use a very simple query.
    * Use the smallest, fastest LLM to rule out resource issues.

## Getting Help

If you've tried these steps and are still stuck:

1.  **Check Existing GitHub Issues:** Someone else might have faced the same problem. Look at [https://github.com/choukha/industrial-ai-agents/issues](https://github.com/choukha/industrial-ai-agents/issues).
2.  **Create a New GitHub Issue:** If your problem is new, please provide as much detail as possible:
    * Steps to reproduce the issue.
    * Full error messages and stack traces.
    * Your OS, Python version, relevant library versions.
    * Ollama version and the models you are using.
    * What you expected to happen and what actually happened.

We hope this guide helps you resolve any issues you encounter!
