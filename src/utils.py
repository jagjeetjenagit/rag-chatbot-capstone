"""
Utility functions for the RAG Chatbot.
"""
import os
from pathlib import Path
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ensure_directory(directory: str) -> Path:
    """
    Ensure a directory exists, create if it doesn't.
    
    Args:
        directory: Directory path
        
    Returns:
        Path object
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def extract_keywords(text: str, top_n: int = 5) -> List[str]:
    """
    Extract simple keywords from text (basic implementation).
    
    Args:
        text: Input text
        top_n: Number of keywords to extract
        
    Returns:
        List of keywords
    """
    import re
    from collections import Counter
    
    # Remove special characters and convert to lowercase
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    
    # Common stop words to filter out
    stop_words = {
        'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'her',
        'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how',
        'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did',
        'its', 'let', 'put', 'say', 'she', 'too', 'use', 'with', 'this', 'that',
        'from', 'have', 'what', 'when', 'will', 'been', 'were', 'there', 'their'
    }
    
    # Filter out stop words
    filtered_words = [w for w in words if w not in stop_words]
    
    # Count word frequency
    word_counts = Counter(filtered_words)
    
    # Get top N keywords
    keywords = [word for word, count in word_counts.most_common(top_n)]
    
    return keywords


def validate_api_key(api_key: str, provider: str = "openai") -> bool:
    """
    Validate API key format (basic check).
    
    Args:
        api_key: API key to validate
        provider: Provider name
        
    Returns:
        True if valid format
    """
    if not api_key or not api_key.strip():
        return False
    
    if provider.lower() == "openai":
        # OpenAI keys start with "sk-"
        return api_key.startswith("sk-") and len(api_key) > 20
    elif provider.lower() == "google":
        # Google API keys are typically 39 characters
        return len(api_key) >= 30
    
    return len(api_key) > 10


def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    import re
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:\-\(\)]', '', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def create_summary(text: str, max_sentences: int = 3) -> str:
    """
    Create a simple summary by extracting first N sentences.
    
    Args:
        text: Input text
        max_sentences: Maximum number of sentences
        
    Returns:
        Summary text
    """
    import re
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Take first N sentences
    summary_sentences = sentences[:max_sentences]
    
    # Join and clean
    summary = ' '.join(summary_sentences)
    
    return summary.strip()


def log_performance(func):
    """
    Decorator to log function performance.
    
    Args:
        func: Function to decorate
        
    Returns:
        Wrapped function
    """
    import time
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger.info(
            f"{func.__name__} executed in {end_time - start_time:.2f} seconds"
        )
        
        return result
    
    return wrapper


def calculate_chunk_stats(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate statistics about text chunks.
    
    Args:
        chunks: List of chunk dictionaries
        
    Returns:
        Dictionary with statistics
    """
    if not chunks:
        return {
            "total_chunks": 0,
            "total_chars": 0,
            "avg_chunk_size": 0,
            "min_chunk_size": 0,
            "max_chunk_size": 0
        }
    
    chunk_sizes = [len(chunk["text"]) for chunk in chunks]
    
    stats = {
        "total_chunks": len(chunks),
        "total_chars": sum(chunk_sizes),
        "avg_chunk_size": sum(chunk_sizes) / len(chunk_sizes),
        "min_chunk_size": min(chunk_sizes),
        "max_chunk_size": max(chunk_sizes)
    }
    
    return stats


def get_env_variable(
    var_name: str,
    default: Any = None,
    required: bool = False
) -> Any:
    """
    Get environment variable with error handling.
    
    Args:
        var_name: Environment variable name
        default: Default value if not found
        required: Whether the variable is required
        
    Returns:
        Environment variable value
        
    Raises:
        ValueError: If required variable is not found
    """
    value = os.getenv(var_name, default)
    
    if required and not value:
        raise ValueError(
            f"Required environment variable '{var_name}' not found. "
            f"Please set it in your .env file or environment."
        )
    
    return value
