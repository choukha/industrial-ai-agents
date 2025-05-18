"""Configuration settings and constants for IDOCA."""

# Default models
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_VISION_MODEL = "granite3.2-vision:2b-fp16"
DEFAULT_LLM_MODEL = "qwen3:0.6b"

# Vector database defaults
DEFAULT_VECTOR_DB_TYPE = "chroma"
DEFAULT_COLLECTION_NAME = "industrial_rag_v1"

# Vector database specific defaults
DEFAULT_CHROMA_PERSIST_DIR = "./chroma_db"
DEFAULT_FAISS_SAVE_PATH = "./faiss_indexes"
DEFAULT_FAISS_INDEX_NAME = "industrial_data"
DEFAULT_MILVUS_HOST = "localhost"
DEFAULT_MILVUS_PORT = "19530"
DEFAULT_MILVUS_COLLECTION = "industrial_docs"
DEFAULT_MILVUS_DROP_OLD = False

# UI elements
BOT_AVATAR_URL = "https://img.icons8.com/plasticine/100/bot.png"  # Generic bot icon

# Document processing
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100

# Logging
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"