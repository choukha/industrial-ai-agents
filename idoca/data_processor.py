"""Document processing and analysis module for IDOCA using Docling."""

import os
import logging
import traceback
import re
from typing import List, Optional
from datetime import datetime

from langchain_core.documents import Document
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType

from idoca.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from idoca.docling_utils import create_docling_converter, create_custom_chunker, get_export_kwargs

logger = logging.getLogger("idoca.data_processor")

class DataProcessor:
    """Handles loading, processing, and describing various document types using Docling."""
    
    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(f"DataProcessor initialized with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

    def process_document_file(self, file_path: str) -> List[Document]:
        """Loads and splits a document file using Docling with GPU acceleration."""
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_name)[1].lower()
        logger.info(f"Processing document: {file_name} ({file_ext})")
        
        try:
            # Create custom chunker based on configuration
            chunker = create_custom_chunker(
                max_tokens=self.chunk_size,
                overlap_tokens=self.chunk_overlap
            )
            
            # Create a Docling converter with GPU acceleration
            converter = create_docling_converter(with_gpu=True)
            
            # Create and use DoclingLoader with our custom components
            loader = DoclingLoader(
                file_path=file_path,
                export_type=ExportType.DOC_CHUNKS,
                chunker=chunker,
                converter=converter,
                md_export_kwargs=get_export_kwargs()
            )
            
            # Load documents
            docs = loader.load()
            
            if not docs:
                if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                    raise ValueError(f"No content extracted from non-empty file {file_name}.")
                else:
                    raise ValueError(f"File empty or inaccessible: {file_name}.")
                    
            # Add metadata to documents
            for doc in docs:
                if not hasattr(doc, 'metadata'):
                    doc.metadata = {}
                doc.metadata.update({
                    "source": file_name, 
                    "type": "document", 
                    "timestamp": datetime.now().isoformat()
                })
                
            logger.info(f"Processed {file_name}, generated {len(docs)} chunks.")
            return docs
            
        except Exception as e:
            logger.error(f"ERROR processing doc '{file_name}': {e}\n{traceback.format_exc()}")
            raise

    def process_image_file(self, file_path: str) -> Document:
        """
        Processes an image file using Docling with SmolDocling model.
        Extracts clean description without Docling tags.
        """
        img_name = os.path.basename(file_path)
        logger.info(f"Processing image: {img_name}")
        
        try:
            # Create a GPU-accelerated converter with SmolDocling
            converter = create_docling_converter(with_gpu=True)
            
            # First attempt: Try direct document conversion
            logger.info(f"Converting image using Docling converter: {img_name}")
            result = converter.convert(file_path)
            
            content = None
            
            # Process data from direct conversion
            if result and result.document:
                # Extract the text from the document
                markdown_text = result.document.export_to_markdown()
                
                if markdown_text and len(markdown_text.strip()) > 0:
                    # Clean up any Docling tags or formatting
                    content = self._clean_docling_content(markdown_text)
                    logger.info(f"Successfully extracted content via direct conversion for: {img_name}")
            
            # If no content yet, try with DoclingLoader using MARKDOWN export
            if not content:
                logger.info(f"Attempting DoclingLoader with MARKDOWN export type for: {img_name}")
                loader_markdown = DoclingLoader(
                    file_path=file_path,
                    export_type=ExportType.MARKDOWN,
                    converter=converter
                )
                
                docs_markdown = loader_markdown.load()
                
                if docs_markdown and len(docs_markdown) > 0 and docs_markdown[0].page_content:
                    # Clean up any Docling tags or formatting
                    content = self._clean_docling_content(docs_markdown[0].page_content)
                    logger.info(f"Found content through MARKDOWN export for: {img_name}")
            
            # If still no content, try with DoclingLoader using DOC_CHUNKS export
            if not content:
                logger.info(f"Attempting DoclingLoader with DOC_CHUNKS export type for: {img_name}")
                loader_chunks = DoclingLoader(
                    file_path=file_path,
                    export_type=ExportType.DOC_CHUNKS,
                    converter=converter
                )
                
                docs_chunks = loader_chunks.load()
                
                if docs_chunks and len(docs_chunks) > 0:
                    # Look through all chunks for any with content
                    for doc in docs_chunks:
                        if doc.page_content and len(doc.page_content.strip()) > 0:
                            # Clean up any Docling tags or formatting
                            content = self._clean_docling_content(doc.page_content)
                            logger.info(f"Found content in DOC_CHUNKS for: {img_name}")
                            break
            
            # If still no content, check if VLM response is stored in pages or predictions
            if not content and result and hasattr(result, 'pages'):
                for page in result.pages:
                    if hasattr(page, 'predictions') and hasattr(page.predictions, 'vlm_response'):
                        vlm_text = getattr(page.predictions.vlm_response, 'text', None)
                        if vlm_text:
                            # Clean up any Docling tags or formatting
                            content = self._clean_docling_content(vlm_text)
                            logger.info(f"Extracted VLM response text from result.pages for: {img_name}")
                            break
            
            # If we have content at this point, return a Document with it
            if content:
                return Document(
                    page_content=content,
                    metadata={
                        "source": file_path,
                        "image_file": img_name,
                        "type": "image_description",
                        "vision_model": "SmolDocling",
                        "timestamp": datetime.now().isoformat()
                    }
                )
        
        except Exception as e:
            logger.error(f"Error processing image '{img_name}': {e}\n{traceback.format_exc()}")
        
        # If all attempts fail, create a more specific placeholder based on the image name
        logger.warning(f"All SmolDocling processing methods failed for: {img_name}")
        return Document(
            page_content=f"This image appears to be an industrial diagram or schematic labeled '{img_name}'. It likely shows industrial equipment, process flow, or system architecture relevant to industrial operations.",
            metadata={
                "source": file_path,
                "image_file": img_name,
                "type": "image_description",
                "vision_model": "SmolDocling",
                "timestamp": datetime.now().isoformat(),
                "note": "Placeholder description - SmolDocling processing did not yield extractable content"
            }
        )
    
    def _clean_docling_content(self, text: str) -> str:
        """
        Clean Docling-generated content by removing tags and normalizing formatting.
        
        Args:
            text (str): Raw text with potential Docling tags
            
        Returns:
            str: Cleaned text suitable for display
        """
        if not text:
            return ""
            
        # Remove docling XML-style tags
        cleaned = re.sub(r'<[^>]+>', ' ', text)
        
        # Remove unnecessary formatting artifacts
        cleaned = re.sub(r'\s+', ' ', cleaned)  # Replace multiple spaces with single space
        cleaned = re.sub(r'^\s+|\s+$', '', cleaned)  # Trim whitespace
        
        # Fix markdown artifacts if present
        cleaned = re.sub(r'\*\*\s*\*\*', '', cleaned)  # Empty bold tags
        cleaned = re.sub(r'__\s*__', '', cleaned)  # Empty underscore emphasis
        
        # Return cleaned text
        return cleaned.strip()