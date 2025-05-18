# Vector Database Options in IDOCA

IDOCA supports multiple vector database options for the RAG (Retrieval-Augmented Generation) system. This document provides detailed information about each option, including setup instructions, benefits, and considerations.

## Overview of Supported Vector Databases

Vector databases are specialized storage systems optimized for similarity search operations on high-dimensional vectors (embeddings). In the context of IDOCA, they store and retrieve document chunk embeddings based on their semantic similarity to a query.

## 1. ChromaDB

### Overview

ChromaDB is a lightweight vector database that can operate either entirely in-memory or with persistence to disk. It's included as the default option in IDOCA due to its simplicity and ease of setup.

### Installation

ChromaDB is installed automatically when you install IDOCA dependencies:

```bash
pip install chromadb
```

### Configuration in IDOCA

* **Persistence Directory:** Optionally specify a local directory where ChromaDB will store its data. If left empty, ChromaDB operates in memory only (data is lost when the application stops).

### Benefits

* **Simplicity:** No external services or complex setup required
* **Zero-configuration option:** Works immediately with no additional setup
* **Integrated persistence:** Simple file-based storage option
* **Low resource usage:** Suitable for development environments

### Considerations

* Less optimized for very large document collections
* Fewer advanced features compared to dedicated vector database systems

## 2. FAISS (Facebook AI Similarity Search)

### Overview

FAISS is a library developed by Facebook AI Research for efficient similarity search and clustering of dense vectors. It's known for its performance optimizations and is widely used in production applications.

### Installation

FAISS can be installed via pip:

```bash
pip install faiss-cpu  # For CPU-only version
# or
pip install faiss-gpu  # For GPU support (requires CUDA)
```

### Configuration in IDOCA

* **Save Path:** Directory where FAISS will save the index files
* **Index Name:** Name of the index file

### Benefits

* **High performance:** Optimized algorithms for vector search
* **Scalability:** Can handle larger datasets efficiently
* **Multiple index types:** Supports various indexing algorithms for different size/performance tradeoffs
* **Good balance:** Between setup simplicity and performance

### Considerations

* Requires separate save/load operations for persistence
* Less suited for dynamic data that changes frequently

## 3. Milvus

### Overview

Milvus is a production-grade vector database designed for enterprise-level applications. It offers advanced features like filtering, horizontal scaling, and high availability. Milvus includes Attu, a graphical user interface for database management and visualization.

### Installation

**Step 1: Install the Python Client**

```bash
pip install pymilvus
```

**Step 2: Install Milvus using Docker**

Milvus runs as a standalone service using Docker:

For Windows:
Follow the instructions at [Milvus Windows Installation Guide](https://milvus.io/docs/install_standalone-windows.md)

For Linux/macOS or Windows with WSL:

Follow the instructions at [Milvus Docker/Linux Installation Guide](https://milvus.io/docs/install_standalone-docker-compose.md)

**Step 3: Install Attu GUI (Optional)**

Attu provides a graphical interface for managing your Milvus database:

Download the installer from [Attu GitHub Releases](https://github.com/zilliztech/attu/releases)
Install and connect to your Milvus instance (default: `localhost:19530`)

**Step 3: Install Attu GUI (Optional)**

Attu provides a graphical interface for managing your Milvus database:

Download the installer from [Attu GitHub Releases](https://github.com/zilliztech/attu/releases)
Install and connect to your Milvus instance (default: `localhost:19530`)

### Configuration in IDOCA

* **Host:** The hostname where Milvus is running (default: `localhost`)
* **Port:** The port Milvus is listening on (default: `19530`)
* **Collection Name:** Name of the Milvus collection to store embeddings
* **Drop Existing:** Whether to drop an existing collection with the same name

### Benefits

* **Advanced features:** Filtering, hybrid search, data management
* **Built-in Web UI:** Milvus offers a built-in web-based user interface for system observability and basic management. Accessible at `http://${MILVUS_PROXY_IP}:9091/webui` (as of v2.5.0). More details at [Milvus Web UI Documentation](https://milvus.io/docs/milvus-webui.md).
* **Optional Attu GUI:** For more detailed database management, operational tasks, and visualization, the standalone Attu GUI is available.
* **Production-ready:** Designed for enterprise workloads
* **Scalability:** Can handle very large datasets
* **Analytics:** Built-in monitoring and performance metrics

### Considerations

* More complex setup requiring Docker
* Higher resource usage compared to simpler options
* Overkill for smaller document collections or prototype applications

## Choosing the Right Vector Database

Consider these factors when selecting a vector database for your IDOCA deployment:

* **Dataset Size:**
    * Small collections (<10K documents): ChromaDB is sufficient
    * Medium collections (10K-100K documents): FAISS offers better performance
    * Large collections (>100K documents): Milvus provides necessary scalability
* **Deployment Environment:**
    * Development/Testing: ChromaDB offers simplicity
    * Production with moderate loads: FAISS balances performance and simplicity
    * Enterprise deployment: Milvus provides robustness and management features
* **Technical Requirements:**
    * Minimal setup: ChromaDB requires no additional services
    * Performance priority: FAISS offers highly optimized search
    * Management needs: Milvus provides visualization and monitoring (via WebUI and/or Attu)
* **Resources Available:**
    * Limited resources: ChromaDB has the lowest overhead
    * Balanced systems: FAISS offers good performance with moderate resources
    * Well-provisioned environments: Milvus can leverage additional resources

For most industrial use cases with moderate document collections and straightforward requirements, FAISS offers an excellent balance between performance and simplicity. For larger enterprise deployments or when visualization and management features are important, Milvus provides the most comprehensive solution.
