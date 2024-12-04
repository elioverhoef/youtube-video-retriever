"""Logging configuration for the YouTube Transcript Retriever."""

import logging
from typing import Optional

def setup_logger(verbose: bool = False) -> logging.Logger:
    """Configure and return the root logger."""
    # Create logger
    logger = logging.getLogger('youtube_transcript_retriever')
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Create console handler with formatter
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(handler)
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module."""
    return logging.getLogger(f'youtube_transcript_retriever.{name}')
