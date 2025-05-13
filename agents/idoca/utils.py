"""Utility functions for IDOCA."""

import base64
import os
import traceback
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

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