"""Configuration settings and constants for IDOCA."""

# Default models
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_VISION_MODEL = "granite3.2-vision:2b-fp16"
DEFAULT_LLM_MODEL = "qwen3:0.6b"

# UI elements
BOT_AVATAR_URL = "https://img.icons8.com/plasticine/100/bot.png"  # Generic bot icon

# Document processing
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100

# Logging
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"