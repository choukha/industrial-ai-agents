"""Utility functions for IDOCA."""

import base64
import os
import traceback
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import ollama
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
import numpy as np

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("idoca")

def encode_image(image_path: str) -> str:
    """Encodes an image file to a base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        logger.error(f"Error encoding image {image_path}: {e}")
        raise

def format_chatbot_message(role: str, content: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
    """Formats a message for the Gradio Chatbot with optional metadata handling."""
    message = {"role": role, "content": content}
    if metadata:
        title = metadata.get("title", "Details")
        log = metadata.get("log", "N/A")
        # Ensure log is a string before replacing
        esc_log = str(log).replace('`', '\\`')
        details_md = f"\n<details><summary>{title}</summary>\n\n```text\n{esc_log}\n```\n\n</details>"
        if message["content"]:
            message["content"] += details_md
        else:
            message["content"] = details_md
    return message


def get_ollama_models():
    """Query Ollama API to get available models using the ollama package."""
    try:
        # List models using the ollama package
        models_response = ollama.list() # This returns a ListResponse object

        # Extract model names from the response
        # Access the .models attribute of the response object,
        # then the .model attribute of each Model object in that list.
        model_names = [model_obj.model for model_obj in models_response.models]
        
        # Sort models alphabetically
        model_names.sort()
        
        return model_names
    except Exception as e:
        logger.error(f"Error getting Ollama models: {e}")
        # Return default models if error occurs
        return ["llama3", "qwen3:8b", "mistral"]

def filter_vision_models(models):
    """Filter for likely vision models based on naming patterns."""
    vision_patterns = ["vision", "llava", "vl", "visual", "multimodal"]
    return [model for model in models if any(pattern in model.lower() for pattern in vision_patterns)]

def filter_embedding_models(models):
    """Filter for likely embedding models based on naming patterns."""
    embedding_patterns = ["embed", "nomic", "mxbai"]
    return [model for model in models if any(pattern in model.lower() for pattern in embedding_patterns)]

class MockEmbeddings(Embeddings):
    """Mock embedding class for fallback."""
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        np.random.seed(hash(text) % 2**32)
        return np.random.rand(384).tolist()

class MockChatModel(BaseChatModel):
    """Mock chat model class for fallback."""
    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager=None, **kwargs) -> AIMessage:
        last_message_content = messages[-1].content if messages else "No message"
        return AIMessage(content=f"Mock response to: '{str(last_message_content)[:50]}...'")

    @property
    def _llm_type(self) -> str:
        return "mock_chat_model"