"""
Shared ChromaDB client singleton to prevent multiple instances with different settings.
"""
import logging
from pathlib import Path
from typing import Optional

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    raise ImportError("chromadb not installed. Install with: pip install chromadb")

logger = logging.getLogger(__name__)

# Global client instance
_chroma_client: Optional[chromadb.PersistentClient] = None
_chroma_db_path: Optional[str] = None


def get_chroma_client(chroma_db_path: str = "./chroma_db") -> chromadb.PersistentClient:
    """
    Get or create a singleton ChromaDB client.
    
    Args:
        chroma_db_path: Path for persistent ChromaDB storage
        
    Returns:
        ChromaDB PersistentClient instance
    """
    global _chroma_client, _chroma_db_path
    
    # Create directory if it doesn't exist
    db_path = Path(chroma_db_path)
    db_path.mkdir(parents=True, exist_ok=True)
    
    # If client exists and path matches, return it
    if _chroma_client is not None and _chroma_db_path == chroma_db_path:
        logger.debug(f"Reusing existing ChromaDB client for: {chroma_db_path}")
        return _chroma_client
    
    # If path changed, we need to create a new client
    # (This shouldn't happen in normal operation, but handle it gracefully)
    if _chroma_client is not None and _chroma_db_path != chroma_db_path:
        logger.warning(f"ChromaDB path changed from {_chroma_db_path} to {chroma_db_path}")
        _chroma_client = None
    
    # Create new client
    logger.info(f"Creating new ChromaDB client for: {chroma_db_path}")
    _chroma_client = chromadb.PersistentClient(
        path=str(db_path),
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )
    )
    _chroma_db_path = chroma_db_path
    
    return _chroma_client


def reset_chroma_client():
    """
    Reset the global ChromaDB client (useful for testing).
    """
    global _chroma_client, _chroma_db_path
    _chroma_client = None
    _chroma_db_path = None
    logger.info("ChromaDB client reset")
