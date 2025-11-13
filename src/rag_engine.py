"""
RAG Engine Module.
Handles retrieval of relevant chunks and LLM-based answer generation.
"""
import os
from typing import List, Dict, Any, Optional
import logging

from .vector_store import VectorStore
from .config import (
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    GOOGLE_API_KEY,
    GOOGLE_MODEL,
    MAX_TOKENS,
    TEMPERATURE,
    RAG_PROMPT_TEMPLATE,
    TOP_K_RETRIEVAL
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGEngine:
    """
    Retrieval-Augmented Generation engine.
    Retrieves relevant document chunks and generates answers using LLMs.
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        llm_provider: str = LLM_PROVIDER,
        top_k: int = TOP_K_RETRIEVAL
    ):
        """
        Initialize the RAG engine.
        
        Args:
            vector_store: VectorStore instance for retrieval
            llm_provider: LLM provider ('openai' or 'google')
            top_k: Number of chunks to retrieve
        """
        self.vector_store = vector_store
        self.llm_provider = llm_provider.lower()
        self.top_k = top_k
        
        # Initialize LLM client
        self._initialize_llm()
        
        logger.info(f"RAG Engine initialized with {llm_provider} LLM and top_k={top_k}")
    
    def _initialize_llm(self):
        """Initialize the LLM client based on provider."""
        if self.llm_provider == "openai":
            self._initialize_openai()
        elif self.llm_provider == "google":
            self._initialize_google()
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
    
    def _initialize_openai(self):
        """Initialize OpenAI client."""
        try:
            import openai
            
            if not OPENAI_API_KEY:
                raise ValueError(
                    "OPENAI_API_KEY not found. Please set it in your .env file or environment."
                )
            
            self.openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
            self.model_name = OPENAI_MODEL
            
            logger.info(f"OpenAI client initialized with model: {self.model_name}")
            
        except ImportError:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")
    
    def _initialize_google(self):
        """Initialize Google Generative AI client."""
        try:
            import google.generativeai as genai
            
            if not GOOGLE_API_KEY:
                raise ValueError(
                    "GOOGLE_API_KEY not found. Please set it in your .env file or environment."
                )
            
            genai.configure(api_key=GOOGLE_API_KEY)
            self.google_model = genai.GenerativeModel(GOOGLE_MODEL)
            self.model_name = GOOGLE_MODEL
            
            logger.info(f"Google AI client initialized with model: {self.model_name}")
            
        except ImportError:
            raise ImportError(
                "Google Generative AI package not installed. "
                "Install with: pip install google-generativeai"
            )
    
    def retrieve_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_source: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context chunks for a query.
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve (overrides default)
            filter_source: Optional source filter
            
        Returns:
            List of retrieved chunks with metadata
        """
        k = top_k if top_k is not None else self.top_k
        
        filter_metadata = {"source": filter_source} if filter_source else None
        
        results = self.vector_store.search(
            query=query,
            top_k=k,
            filter_metadata=filter_metadata
        )
        
        return results
    
    def _format_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Format retrieved chunks into a context string for the LLM.
        
        Args:
            retrieved_chunks: List of retrieved chunk dictionaries
            
        Returns:
            Formatted context string
        """
        if not retrieved_chunks:
            return "No relevant context found."
        
        context_parts = []
        
        for i, chunk in enumerate(retrieved_chunks, 1):
            source = chunk["metadata"].get("source", "Unknown")
            chunk_idx = chunk["metadata"].get("chunk_index", "?")
            text = chunk["text"]
            
            context_parts.append(
                f"[Source {i}: {source}, Chunk {chunk_idx}]\n{text}\n"
            )
        
        return "\n".join(context_parts)
    
    def _call_openai(self, prompt: str) -> str:
        """
        Call OpenAI API to generate answer.
        
        Args:
            prompt: Full prompt with context and question
            
        Returns:
            Generated answer
        """
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE
            )
            
            answer = response.choices[0].message.content.strip()
            return answer
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {str(e)}")
            raise
    
    def _call_google(self, prompt: str) -> str:
        """
        Call Google Generative AI to generate answer.
        
        Args:
            prompt: Full prompt with context and question
            
        Returns:
            Generated answer
        """
        try:
            response = self.google_model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": MAX_TOKENS,
                    "temperature": TEMPERATURE,
                }
            )
            
            answer = response.text.strip()
            return answer
            
        except Exception as e:
            logger.error(f"Error calling Google AI API: {str(e)}")
            raise
    
    def generate_answer(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_source: Optional[str] = None,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Generate an answer using RAG (Retrieval-Augmented Generation).
        
        Args:
            query: User query/question
            top_k: Number of chunks to retrieve
            filter_source: Optional source document filter
            include_sources: Whether to include source information
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        try:
            # Step 1: Retrieve relevant context
            logger.info(f"Processing query: {query}")
            retrieved_chunks = self.retrieve_context(query, top_k, filter_source)
            
            # Handle case where no context is found
            if not retrieved_chunks:
                logger.warning("No relevant context found for query")
                return {
                    "answer": "I couldn't find any relevant information in the uploaded documents to answer your question. Please make sure you've uploaded documents or try rephrasing your question.",
                    "sources": [],
                    "retrieved_chunks": [],
                    "context_found": False
                }
            
            # Step 2: Format context
            context = self._format_context(retrieved_chunks)
            
            # Step 3: Create prompt
            prompt = RAG_PROMPT_TEMPLATE.format(
                context=context,
                question=query
            )
            
            # Step 4: Generate answer using LLM
            logger.info(f"Generating answer using {self.llm_provider}...")
            
            if self.llm_provider == "openai":
                answer = self._call_openai(prompt)
            elif self.llm_provider == "google":
                answer = self._call_google(prompt)
            else:
                raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
            
            # Step 5: Extract source information
            sources = []
            if include_sources:
                seen_sources = set()
                for chunk in retrieved_chunks:
                    source = chunk["metadata"].get("source", "Unknown")
                    if source not in seen_sources:
                        sources.append({
                            "name": source,
                            "chunk_index": chunk["metadata"].get("chunk_index"),
                            "similarity_score": chunk.get("similarity_score", 0)
                        })
                        seen_sources.add(source)
            
            logger.info(f"Answer generated successfully from {len(sources)} sources")
            
            return {
                "answer": answer,
                "sources": sources,
                "retrieved_chunks": retrieved_chunks,
                "context_found": True,
                "query": query
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return {
                "answer": f"An error occurred while generating the answer: {str(e)}",
                "sources": [],
                "retrieved_chunks": [],
                "context_found": False,
                "error": str(e)
            }
    
    def chat(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Handle a chat interaction with optional conversation history.
        
        Args:
            query: User query
            conversation_history: Optional list of previous messages
            
        Returns:
            Response dictionary with answer and metadata
        """
        # For now, we'll use simple RAG without full conversation context
        # This can be extended to include conversation history in the prompt
        
        result = self.generate_answer(query)
        
        # Add conversation history handling if needed
        if conversation_history:
            # Could implement conversation-aware retrieval here
            pass
        
        return result


def create_rag_engine(
    vector_store: VectorStore,
    llm_provider: str = LLM_PROVIDER
) -> RAGEngine:
    """
    Convenience function to create a RAG engine.
    
    Args:
        vector_store: VectorStore instance
        llm_provider: LLM provider name
        
    Returns:
        RAGEngine instance
    """
    return RAGEngine(vector_store, llm_provider)
