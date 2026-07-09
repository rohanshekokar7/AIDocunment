"""
AI Document Classification System
Developed by Rohan Shekokar
"""

import logging
import sys

def setup_logging(logger_name: str = "document_classification") -> logging.Logger:
    """
    Sets up the application logger with a standard format.
    """
    logger = logging.getLogger(logger_name)
    
    # Only configure if it doesn't already have handlers to avoid duplication
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        # Stream Handler (Stdout)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        
    return logger

# Create a default logger instance
logger = setup_logging()
