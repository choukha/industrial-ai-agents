# IDOCA - Industrial Document Analysis Agent

IDOCA (Industrial Document Analysis Agent) is a system designed to understand and answer questions based on the content of your industrial documents. It uses a Retrieval-Augmented Generation (RAG) pipeline with locally deployed Large Language Models (LLMs) via Ollama.

## Overview

This agent helps you unlock the knowledge hidden in your technical manuals, Standard Operating Procedures (SOPs), safety guidelines, maintenance logs, and other industrial documents. By asking questions in natural language, you can quickly find relevant information, speeding up troubleshooting, training, and decision-making.

**Key Features:**
* Processes various document formats (PDF, DOCX, TXT, CSV).
* Utilizes advanced PDF parsing (optionally with Docling) to handle complex layouts and tables.
* Builds a local, searchable knowledge base from your documents.
* Answers questions grounded in the provided document content.
* Supports both simple RAG and more advanced agentic RAG for complex queries.
* Runs entirely on your local machine, ensuring data privacy.

For a comprehensive understanding of IDOCA's architecture, capabilities, and advanced usage, please refer to the main **[IDOCA Documentation](../../docs/idoca/index.md)**.

## Quick Start

### Prerequisites

1.  **Main Project Setup:** Ensure you have completed all steps in the main repository's **[Getting Started Guide](../../docs/getting_started.md)**. This includes:
    * Cloning the `industrial-ai-agents` repository.
    * Setting up a Python virtual environment.
    * Installing Ollama and pulling necessary models (especially `nomic-embed-text` and a suitable LLM like `llama3:8b` or `qwen2:1.5b`).
    * Installing root dependencies from `requirements.txt`.
2.  **IDOCA Dependencies:** Install IDOCA-specific dependencies:
    ```bash
    # From the root of the industrial-ai-agents repository
    pip install -r idoca/requirements.txt
    ```
3.  **(Optional but Recommended) Docling Setup:** For best results with PDF documents, especially those with complex layouts or tables, set up and run the Docling server (preferably via Docker):
    ```bash
    docker pull quay.io/docling-project/docling-serve:latest
    docker run -d -p 5001:5001 --name docling_server quay.io/docling-project/docling-serve:latest
    ```
    Ensure IDOCA's configuration (`idoca/config.py`) points to the Docling server if you use it.

### Running IDOCA

1.  **Activate Virtual Environment:**
    ```bash
    # From the root of the industrial-ai-agents repository
    # On Windows:
    # venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```
2.  **Ensure Ollama is Running:** Check that the Ollama service/application is active.
3.  **Ensure Docling Server is Running (if used).**
4.  **Navigate to IDOCA Directory:**
    ```bash
    # From the root of the industrial-ai-agents repository
    cd idoca
    ```
5.  **Run the Application:**
    ```bash
    python main.py
    ```
6.  Open your web browser and go to the URL shown in the terminal (usually `http://127.0.0.1:7860`).
7.  Follow the UI instructions:
    * Upload your documents.
    * Select models.
    * Click "Process New Files".
    * Click "Initialize/Re-Initialize RAG & Agent".
    * Ask questions in the "Simple RAG" or "Agentic RAG" tabs.

## Detailed Documentation

For more detailed information on:
* Architecture
* Advanced Setup and Configuration
* In-depth Usage Guide
* Customization
* Troubleshooting

Please visit the main **[IDOCA Documentation section](../../docs/idoca/index.md)**.

## Reporting Issues

If you encounter any issues specific to IDOCA, please report them on the main repository's [issue tracker](https://github.com/choukha/industrial-ai-agents/issues), providing details about the IDOCA agent.
