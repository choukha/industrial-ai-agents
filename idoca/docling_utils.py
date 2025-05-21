"""Utility functions for Docling integration in IDOCA."""

import logging
from typing import Dict, Any

from docling.document_converter import DocumentConverter, PdfFormatOption, ImageFormatOption
from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling_core.types.doc import ImageRefMode, DocItemLabel
from docling_core.types.doc.document import DEFAULT_EXPORT_LABELS
from docling.datamodel.pipeline_options import VlmPipelineOptions, AcceleratorOptions, AcceleratorDevice, smoldocling_vlm_conversion_options
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

logger = logging.getLogger("idoca.docling_utils")


def create_docling_converter(with_gpu: bool = True) -> DocumentConverter:
    """
    Creates a properly configured Docling DocumentConverter with GPU acceleration if available.
    
    Args:
        with_gpu (bool): Whether to use GPU acceleration
        
    Returns:
        DocumentConverter: Configured converter instance
    """
    # Setup VLM pipeline options with acceleration
    pipeline_options = VlmPipelineOptions()
    
    # Configure GPU acceleration if requested
    if with_gpu:
        try:
            accelerator_options = AcceleratorOptions(
                num_threads=8,
                device=AcceleratorDevice.CUDA  # Use CUDA for NVIDIA GPUs
            )
            pipeline_options.accelerator_options = accelerator_options
            logger.info("Configured GPU acceleration for Docling")
        except Exception as e:
            logger.warning(f"Failed to configure GPU acceleration: {e}. Falling back to CPU.")
    
    # Use SmolDocling by default
    pipeline_options.vlm_options = smoldocling_vlm_conversion_options
    
    # Configure format options for different file types
    pdf_format_option = PdfFormatOption(
        pipeline_cls=VlmPipeline,
        pipeline_options=pipeline_options
    )
    
    image_format_option = ImageFormatOption(
        pipeline_cls=VlmPipeline,
        pipeline_options=pipeline_options
    )
    
    # Create and return the configured converter
    converter = DocumentConverter(
        allowed_formats=[
            InputFormat.PDF,
            InputFormat.IMAGE,
            InputFormat.DOCX,
            InputFormat.HTML,
            InputFormat.PPTX,
            InputFormat.ASCIIDOC,
            InputFormat.CSV,
            InputFormat.MD,
        ],
        format_options={
            InputFormat.PDF: pdf_format_option,
            InputFormat.IMAGE: image_format_option
        }
    )
    return converter


def create_custom_chunker(
    max_tokens: int = 512,
    overlap_tokens: int = 128,
    tokenizer_model: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> HybridChunker:
    """
    Creates a HybridChunker with appropriate settings.
    
    Args:
        max_tokens (int): Maximum tokens per chunk
        overlap_tokens (int): Overlap between chunks in tokens
        tokenizer_model (str): Model to use for tokenization
        
    Returns:
        HybridChunker: Configured chunker instance
    """
    # Create tokenizer with explicit truncation
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    huggingface_tokenizer = HuggingFaceTokenizer(
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        truncation=True  # Explicitly set truncation to True
    )
    
    return HybridChunker(
        tokenizer=huggingface_tokenizer,
        max_tokens=max_tokens,
        merge_peers=True
    )


def get_export_kwargs() -> Dict[str, Any]:
    """
    Returns standard export kwargs for Docling document export.
    
    Returns:
        Dict[str, Any]: Export configuration
    """
    return {
        "image_mode": ImageRefMode.PLACEHOLDER,
        "labels": [*DEFAULT_EXPORT_LABELS, DocItemLabel.FOOTNOTE]
    }