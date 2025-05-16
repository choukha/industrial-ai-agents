"""Document processing and analysis module for IDOCA."""

import os
import logging
import traceback
from typing import List, Optional
from datetime import datetime
from PIL import Image

from langchain_community.document_loaders import (
    TextLoader, CSVLoader, UnstructuredFileLoader, UnstructuredPDFLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from idoca.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from idoca.utils import encode_image

logger = logging.getLogger("idoca.data_processor")

class DataProcessor:
    """Handles loading, processing, and describing various document types."""
    
    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        logger.info(f"DataProcessor initialized with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

    def process_document_file(self, file_path: str) -> List[Document]:
        """Loads and splits a document file (PDF, CSV, TXT, MD)."""
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_name)[1].lower()
        logger.info(f"Processing document: {file_name} ({file_ext})")
        
        try:
            # Select appropriate loader based on file extension
            if file_ext == '.pdf':
                loader = UnstructuredPDFLoader(file_path, mode="single", strategy="fast")
            elif file_ext == '.csv':
                loader = CSVLoader(file_path, autodetect_encoding=True)
            elif file_ext in ['.txt', '.md']:
                loader = TextLoader(file_path, encoding='utf-8')
            else:
                loader = UnstructuredFileLoader(file_path, mode="elements")

            # Load and validate documents
            raw_docs = loader.load()
            if not raw_docs:
                if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                    raise ValueError(f"No content extracted from non-empty file {file_name}.")
                else:
                    raise ValueError(f"File empty or inaccessible: {file_name}.")

            # Add metadata to documents
            for doc in raw_docs:
                if not hasattr(doc, 'metadata'):
                    doc.metadata = {}
                doc.metadata.update({
                    "source": file_name, 
                    "type": "document", 
                    "timestamp": datetime.now().isoformat()
                })

            # Split documents and handle edge cases
            split_docs = self.text_splitter.split_documents(raw_docs)
            if not split_docs and raw_docs:
                logger.warning(f"No chunks for {file_name}. Using raw documents.")
                # Ensure metadata is on raw_docs if returned
                for doc_raw in raw_docs:
                    if not hasattr(doc_raw, 'metadata'): 
                        doc_raw.metadata = {}
                    doc_raw.metadata.setdefault("source", file_name)
                    doc_raw.metadata.setdefault("type", "document")
                    doc_raw.metadata.setdefault("timestamp", datetime.now().isoformat())
                return raw_docs
                
            if not split_docs:
                return []

            logger.info(f"Processed {file_name}, generated {len(split_docs)} chunks.")
            return split_docs
            
        except Exception as e:
            logger.error(f"ERR processing doc '{file_name}': {e}\n{traceback.format_exc()}")
            raise

    def generate_image_description(self, image_path: str, vision_model_name: str) -> str:
        """Generates a detailed description for an image using a vision model."""
        img_name = os.path.basename(image_path)
        logger.info(f"Generating description for {img_name} via {vision_model_name}")
        
        try:
            # Encode image for model input
            img_data = encode_image(image_path)
            
            # Prompt for industrial image description
            prompt = ("Describe this industrial image in detail. Focus on: "
                      "1. **Equipment/Machinery:** Types (e.g., furnace, conveyor). "
                      "2. **Visible Parameters:** Text on gauges, screens, labels. "
                      "3. **Safety Aspects:** PPE, hazards, signs. "
                      "4. **Overall Context:** Activity or state. Comprehensive summary.")
            
            # Get image description from vision model
            llm = ChatOllama(model=vision_model_name)
            msg_content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": f"data:image/jpeg;base64,{img_data}"}
            ]
            response = llm.invoke([HumanMessage(content=msg_content)])
            
            if not response or not response.content:
                raise ValueError(f"Empty response from vision model '{vision_model_name}' for {img_name}.")
            
            logger.info(f"Description generated for {img_name}")
            return response.content
            
        except Exception as e:
            logger.error(f"ERR ({type(e).__name__}) generating img desc for {img_name}: {e}\n{traceback.format_exc()}")
            raise

    def process_image_file(self, file_path: str, vision_model_name: str) -> Document:
        """Processes an image file by generating its description and creating a Document."""
        img_name = os.path.basename(file_path)
        logger.info(f"Processing image: {img_name} via {vision_model_name}")
        
        try:
            # Generate description and create document with metadata
            desc = self.generate_image_description(file_path, vision_model_name)
            doc = Document(
                page_content=desc,
                metadata={
                    "source": file_path, 
                    "image_file": img_name, 
                    "type": "image_description",
                    "vision_model": vision_model_name, 
                    "timestamp": datetime.now().isoformat()
                }
            )
            logger.info(f"Doc created for image {img_name}.")
            return doc
            
        except Exception as e:
            raise