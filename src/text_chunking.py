"""
Text Chunking Module.
Splits documents into chunks of 500-800 characters with 10% overlap.
"""
import re
from typing import List, Dict, Any
import logging

from .config import CHUNK_SIZE_MIN, CHUNK_SIZE_MAX, CHUNK_OVERLAP_PERCENT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextChunker:
    """
    Splits text into overlapping chunks for efficient retrieval.
    Implements smart chunking that respects sentence boundaries.
    """
    
    def __init__(
        self,
        chunk_size_min: int = CHUNK_SIZE_MIN,
        chunk_size_max: int = CHUNK_SIZE_MAX,
        overlap_percent: int = CHUNK_OVERLAP_PERCENT
    ):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size_min: Minimum characters per chunk (default: 500)
            chunk_size_max: Maximum characters per chunk (default: 800)
            overlap_percent: Percentage overlap between chunks (default: 10%)
        """
        self.chunk_size_min = chunk_size_min
        self.chunk_size_max = chunk_size_max
        self.overlap_percent = overlap_percent
        self.overlap_size = int(chunk_size_max * overlap_percent / 100)
        
        logger.info(
            f"TextChunker initialized: size={chunk_size_min}-{chunk_size_max}, "
            f"overlap={overlap_percent}% ({self.overlap_size} chars)"
        )
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using regex patterns.
        
        Args:
            text: Input text to split
            
        Returns:
            List[str]: List of sentences
        """
        # Pattern to split on sentence boundaries
        # Matches periods, exclamation marks, question marks followed by space/newline
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\n+'
        
        sentences = re.split(sentence_pattern, text)
        
        # Clean up sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _create_chunk(
        self,
        sentences: List[str],
        start_idx: int,
        source: str,
        chunk_index: int
    ) -> Dict[str, Any]:
        """
        Create a chunk from sentences starting at start_idx.
        
        Args:
            sentences: List of sentences
            start_idx: Starting sentence index
            source: Source document name
            chunk_index: Index of this chunk
            
        Returns:
            Dict containing chunk text and metadata
        """
        chunk_text = ""
        sentence_idx = start_idx
        
        # Add sentences until we reach the minimum chunk size
        while sentence_idx < len(sentences) and len(chunk_text) < self.chunk_size_min:
            if chunk_text:
                chunk_text += " "
            chunk_text += sentences[sentence_idx]
            sentence_idx += 1
        
        # Continue adding sentences if we haven't exceeded max size
        while sentence_idx < len(sentences) and len(chunk_text) < self.chunk_size_max:
            next_sentence = sentences[sentence_idx]
            if len(chunk_text) + len(next_sentence) + 1 <= self.chunk_size_max:
                chunk_text += " " + next_sentence
                sentence_idx += 1
            else:
                break
        
        # Create metadata
        metadata = {
            "source": source,
            "chunk_index": chunk_index,
            "char_count": len(chunk_text),
            "start_sentence": start_idx,
            "end_sentence": sentence_idx - 1
        }
        
        return {
            "text": chunk_text.strip(),
            "metadata": metadata,
            "next_start_idx": sentence_idx
        }
    
    def _calculate_overlap_start(
        self,
        sentences: List[str],
        current_idx: int
    ) -> int:
        """
        Calculate the starting sentence index for the next chunk with overlap.
        
        Args:
            sentences: List of sentences
            current_idx: Current sentence index
            
        Returns:
            int: Starting index for next chunk
        """
        # Calculate how many characters we need to go back
        overlap_chars = 0
        overlap_start = current_idx
        
        # Move backward until we have enough overlap
        while overlap_start > 0 and overlap_chars < self.overlap_size:
            overlap_start -= 1
            overlap_chars += len(sentences[overlap_start])
        
        # Ensure we don't go back too far (max overlap is overlap_percent)
        max_overlap_sentences = max(1, int(current_idx * self.overlap_percent / 100))
        overlap_start = max(overlap_start, current_idx - max_overlap_sentences)
        
        return overlap_start
    
    def chunk_text(
        self,
        text: str,
        source: str = "unknown",
        metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks with metadata.
        
        Args:
            text: Input text to chunk
            source: Source document identifier
            metadata: Additional metadata to include with each chunk
            
        Returns:
            List[Dict]: List of chunks with text and metadata
        """
        if not text or not text.strip():
            logger.warning(f"Empty text provided for chunking from source: {source}")
            return []
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split into sentences
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            logger.warning(f"No sentences found in text from source: {source}")
            # If no sentences detected, treat entire text as one sentence
            sentences = [text]
        
        chunks = []
        sentence_idx = 0
        chunk_index = 0
        
        while sentence_idx < len(sentences):
            # Create chunk
            chunk_data = self._create_chunk(
                sentences,
                sentence_idx,
                source,
                chunk_index
            )
            
            # Add additional metadata if provided
            if metadata:
                chunk_data["metadata"].update(metadata)
            
            chunks.append({
                "text": chunk_data["text"],
                "metadata": chunk_data["metadata"]
            })
            
            # Move to next chunk with overlap
            next_idx = chunk_data["next_start_idx"]
            
            if next_idx >= len(sentences):
                break
            
            # Calculate overlap for next chunk
            sentence_idx = self._calculate_overlap_start(sentences, next_idx)
            
            # Ensure we're making progress
            if sentence_idx == chunk_data["start_sentence"]:
                sentence_idx = next_idx
            
            chunk_index += 1
        
        logger.info(
            f"Chunked text from '{source}': {len(text)} chars -> {len(chunks)} chunks"
        )
        
        return chunks


def chunk_text(
    text: str,
    source: str = "unknown",
    metadata: Dict[str, Any] = None,
    chunk_size_min: int = CHUNK_SIZE_MIN,
    chunk_size_max: int = CHUNK_SIZE_MAX,
    overlap_percent: int = CHUNK_OVERLAP_PERCENT
) -> List[Dict[str, Any]]:
    """
    Convenience function to chunk text with default or custom parameters.
    
    Args:
        text: Input text to chunk
        source: Source document identifier
        metadata: Additional metadata to include
        chunk_size_min: Minimum chunk size in characters
        chunk_size_max: Maximum chunk size in characters
        overlap_percent: Overlap percentage between chunks
        
    Returns:
        List[Dict]: List of chunks with text and metadata
    """
    chunker = TextChunker(chunk_size_min, chunk_size_max, overlap_percent)
    return chunker.chunk_text(text, source, metadata)


def chunk_documents(
    documents: List[tuple],
    chunk_size_min: int = CHUNK_SIZE_MIN,
    chunk_size_max: int = CHUNK_SIZE_MAX,
    overlap_percent: int = CHUNK_OVERLAP_PERCENT
) -> List[Dict[str, Any]]:
    """
    Chunk multiple documents.
    
    Args:
        documents: List of (text, metadata) tuples
        chunk_size_min: Minimum chunk size in characters
        chunk_size_max: Maximum chunk size in characters
        overlap_percent: Overlap percentage between chunks
        
    Returns:
        List[Dict]: Combined list of all chunks from all documents
    """
    chunker = TextChunker(chunk_size_min, chunk_size_max, overlap_percent)
    all_chunks = []
    
    for text, metadata in documents:
        source = metadata.get("source", "unknown")
        chunks = chunker.chunk_text(text, source, metadata)
        all_chunks.extend(chunks)
    
    logger.info(f"Total chunks created from {len(documents)} documents: {len(all_chunks)}")
    
    return all_chunks
