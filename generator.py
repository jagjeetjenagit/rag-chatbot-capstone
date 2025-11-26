"""
Answer Generation Module for RAG Chatbot.
Generates answers using LLM (Google Gemini, OpenAI, Ollama, or rule-based fallback).

Features:
- Multiple LLM backends (Gemini, OpenAI, Ollama)
- Automatic fallback to rule-based generation
- Confidence scoring
- Source attribution
- Empty context handling
"""
import os
import logging
from typing import List, Dict, Any, Optional
import json
import re

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import API configuration
try:
    from config import LLM_PROVIDER, is_api_configured, get_llm_config
    from llm_api import call_llm
    CONFIG_AVAILABLE = True
except ImportError:
    logger.warning("config.py or llm_api.py not found. Using fallback mode only.")
    CONFIG_AVAILABLE = False
    LLM_PROVIDER = "fallback"


# ============================================================================
# SYSTEM PROMPT TEMPLATE (Tunable by students)
# ============================================================================
SYSTEM_PROMPT_TEMPLATE = """You are a helpful AI assistant answering questions based on provided context.

INSTRUCTIONS:
1. Read the context carefully and answer the question based ONLY on the information provided.
2. If the context doesn't contain relevant information, say so clearly.
3. Keep your answer concise (2-3 sentences maximum).
4. List the sources you used by referencing chunk indices in square brackets [1], [2], etc.
5. At the end, provide a confidence score (0.0 to 1.0) indicating how well the context answers the question.
   - 1.0 = Context directly and completely answers the question
   - 0.5-0.9 = Context partially answers the question
   - 0.0-0.4 = Context has little to no relevant information

FORMAT YOUR RESPONSE AS JSON:
{{
  "answer": "Your concise answer here with source citations [1], [2]",
  "sources_used": [1, 2],
  "confidence": 0.85,
  "reasoning": "Brief explanation of confidence score"
}}

CONTEXT:
{context}

QUESTION: {question}

Remember: Respond ONLY with valid JSON in the format shown above."""


# Alternative simpler prompt (students can switch by changing the variable)
SIMPLE_PROMPT_TEMPLATE = """Answer this question based on the context below.

Context:
{context}

Question: {question}

Provide your answer in JSON format:
{{"answer": "...", "sources_used": [1, 2], "confidence": 0.8}}"""


# ============================================================================
# LLM API INTEGRATION FUNCTIONS
# ============================================================================

def call_openai_api(prompt: str, max_tokens: int = 512) -> Optional[str]:
    """
    Call OpenAI API (GPT-3.5/GPT-4) for answer generation.
    
    Students: Replace this with actual OpenAI API integration.
    
    Args:
        prompt: The complete prompt to send to the LLM
        max_tokens: Maximum tokens to generate
        
    Returns:
        str: The LLM response, or None if API call fails
    """
    try:
        # Check if OpenAI API key is available
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            logger.warning("OPENAI_API_KEY not found in environment")
            return None
        
        # TODO: Students add OpenAI API code here
        # Example integration:
        # from openai import OpenAI
        # client = OpenAI(api_key=api_key)
        # response = client.chat.completions.create(
        #     model="gpt-3.5-turbo",
        #     messages=[{"role": "user", "content": prompt}],
        #     max_tokens=max_tokens,
        #     temperature=0.7
        # )
        # return response.choices[0].message.content
        
        logger.info("OpenAI API integration not implemented (placeholder)")
        return None
        
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {str(e)}")
        return None


def call_gemini_api(prompt: str, max_tokens: int = 512) -> Optional[str]:
    """
    Call Google Gemini API for answer generation.
    
    Students: Replace this with actual Gemini API integration.
    
    Args:
        prompt: The complete prompt to send to the LLM
        max_tokens: Maximum tokens to generate
        
    Returns:
        str: The LLM response, or None if API call fails
    """
    try:
        # Check if Gemini API key is available
        api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
        if not api_key:
            logger.warning("GOOGLE_API_KEY or GEMINI_API_KEY not found in environment")
            return None
        
        # TODO: Students add Gemini API code here
        # Example integration:
        # import google.generativeai as genai
        # genai.configure(api_key=api_key)
        # model = genai.GenerativeModel('gemini-pro')
        # response = model.generate_content(
        #     prompt,
        #     generation_config=genai.types.GenerationConfig(
        #         max_output_tokens=max_tokens,
        #         temperature=0.7,
        #     )
        # )
        # return response.text
        
        logger.info("Gemini API integration not implemented (placeholder)")
        return None
        
    except Exception as e:
        logger.error(f"Error calling Gemini API: {str(e)}")
        return None


def call_ollama(prompt: str, model: str = "llama2", max_tokens: int = 512) -> Optional[str]:
    """
    Call local Ollama model for answer generation.
    
    Students: Replace this with actual Ollama integration.
    Requires Ollama to be installed and running locally.
    
    Args:
        prompt: The complete prompt to send to the LLM
        model: Ollama model name (e.g., "llama2", "mistral", "phi")
        max_tokens: Maximum tokens to generate
        
    Returns:
        str: The LLM response, or None if Ollama is not available
    """
    try:
        # TODO: Students add Ollama API code here
        # Example integration:
        # import requests
        # response = requests.post(
        #     'http://localhost:11434/api/generate',
        #     json={
        #         'model': model,
        #         'prompt': prompt,
        #         'stream': False,
        #         'options': {
        #             'num_predict': max_tokens,
        #             'temperature': 0.7,
        #         }
        #     },
        #     timeout=30
        # )
        # if response.status_code == 200:
        #     return response.json()['response']
        # return None
        
        logger.info("Ollama integration not implemented (placeholder)")
        return None
        
    except Exception as e:
        logger.error(f"Error calling Ollama: {str(e)}")
        return None


def call_llm_api(prompt: str, backend: str = "auto", max_tokens: int = 512) -> Optional[str]:
    """
    Universal LLM API caller - tries available backends in order.
    
    This function attempts to call different LLM backends based on:
    1. Explicit backend parameter
    2. Available API keys in environment
    3. Fallback order: OpenAI → Gemini → Ollama
    
    Args:
        prompt: The complete prompt to send to the LLM
        backend: Which backend to use ("openai", "gemini", "ollama", or "auto")
        max_tokens: Maximum tokens to generate
        
    Returns:
        str: The LLM response, or None if all backends fail
    """
    logger.info(f"Calling LLM API (backend: {backend})")
    
    if backend == "openai":
        return call_openai_api(prompt, max_tokens)
    elif backend == "gemini":
        return call_gemini_api(prompt, max_tokens)
    elif backend == "ollama":
        return call_ollama(prompt, max_tokens=max_tokens)
    elif backend == "auto":
        # Try backends in order of preference
        logger.info("Auto-selecting LLM backend...")
        
        # Try OpenAI first
        if os.getenv('OPENAI_API_KEY'):
            result = call_openai_api(prompt, max_tokens)
            if result:
                return result
        
        # Try Gemini second
        if os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY'):
            result = call_gemini_api(prompt, max_tokens)
            if result:
                return result
        
        # Try Ollama last
        result = call_ollama(prompt, max_tokens=max_tokens)
        if result:
            return result
        
        logger.warning("No LLM backends available, will use fallback")
        return None
    else:
        logger.error(f"Unknown backend: {backend}")
        return None


# ============================================================================
# RULE-BASED FALLBACK GENERATOR
# ============================================================================

def rule_based_generate(query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Rule-based fallback answer generation (when no LLM APIs are available).
    
    This is a simple extractive approach:
    1. Finds chunks with highest keyword overlap with query
    2. Extracts relevant sentences
    3. Provides basic confidence based on keyword matches
    
    Args:
        query: The user's question
        chunks: Retrieved document chunks
        
    Returns:
        Dict: Answer dictionary with answer, sources, and confidence
    """
    logger.info("Using rule-based fallback generator")
    
    if not chunks:
        return {
            "answer": "I couldn't find relevant information in the dataset to answer your question.",
            "sources": [],
            "confidence": 0.0,
            "method": "fallback"
        }
    
    # Extract query keywords (simple tokenization)
    query_lower = query.lower()
    query_words = set(re.findall(r'\b\w{3,}\b', query_lower))  # Words 3+ chars
    
    # Score chunks by keyword overlap
    chunk_scores = []
    for idx, chunk in enumerate(chunks[:3]):  # Only use top 3
        chunk_text = chunk.get('text', '').lower()
        chunk_words = set(re.findall(r'\b\w{3,}\b', chunk_text))
        
        # Calculate overlap
        overlap = len(query_words & chunk_words)
        if overlap > 0:
            chunk_scores.append((idx, chunk, overlap))
    
    if not chunk_scores:
        return {
            "answer": "The retrieved documents don't seem to directly address your question. Please try rephrasing your query.",
            "sources": [chunk.get('metadata', {}).get('source', 'Unknown') for chunk in chunks[:2]],
            "confidence": 0.2,
            "method": "fallback"
        }
    
    # Sort by score
    chunk_scores.sort(key=lambda x: x[2], reverse=True)
    
    # Extract best sentences from top chunk
    best_chunk = chunk_scores[0][1]
    chunk_text = best_chunk.get('text', '')
    
    # Clean the text - remove document metadata but keep structure
    # Remove report metadata lines
    chunk_text = re.sub(r'\*\*Report Period\*\*:[^\n]+\n', '', chunk_text)
    chunk_text = re.sub(r'\*\*Prepared by\*\*:[^\n]+\n', '', chunk_text)
    chunk_text = re.sub(r'\*\*Report Date\*\*:[^\n]+\n', '', chunk_text)
    chunk_text = re.sub(r'\*\*Classification\*\*:[^\n]+\n', '', chunk_text)
    # Remove horizontal rules
    chunk_text = re.sub(r'^-{3,}\s*$', '', chunk_text, flags=re.MULTILINE)
    # Remove document title headers (starting with #)
    chunk_text = re.sub(r'^#{1,6}\s+[^\n]+Report[^\n]+\n', '', chunk_text, flags=re.MULTILINE)
    
    # Split into paragraphs/sections
    sections = chunk_text.split('\n\n')
    
    # Find the most relevant section (usually Executive Summary or content after headers)
    best_section = None
    for section in sections:
        section = section.strip()
        if not section:
            continue
        # Skip metadata sections
        if any(skip in section.lower() for skip in ['report period', 'prepared by', 'classification']):
            continue
        # Look for content sections
        if len(section) > 100 and any(word in section.lower() for word in query_words):
            best_section = section
            break
    
    if not best_section:
        # Fallback to first substantial section
        for section in sections:
            if len(section.strip()) > 100:
                best_section = section.strip()
                break
    
    if best_section:
        # Create a structured, readable answer
        # Remove inline formatting clutter
        best_section = re.sub(r'\*\*([^*]+)\*\*', r'**\1**', best_section)  # Keep bold but clean it
        best_section = re.sub(r'\s*—\s*', ' - ', best_section)  # Clean dashes
        best_section = re.sub(r'\s*\|\s*', ', ', best_section)  # Replace pipes with commas
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', best_section)
        
        # Build a clean answer with key points
        answer_parts = []
        seen_content = set()
        
        for sentence in sentences[:20]:
            sentence = sentence.strip()
            if len(sentence) < 30:
                continue
            
            # Skip duplicate-like content
            sent_key = sentence.lower()[:50]
            if sent_key in seen_content:
                continue
            seen_content.add(sent_key)
            
            sentence_lower = sentence.lower()
            
            # Prioritize sentences with query keywords
            if any(word in sentence_lower for word in query_words):
                answer_parts.append(sentence)
            elif len(answer_parts) < 2:  # Include first 2 sentences even without keywords
                answer_parts.append(sentence)
            
            if len(answer_parts) >= 5:
                break
        
        if answer_parts:
            # Format as bullet points if multiple items, otherwise paragraph
            if len(answer_parts) > 2:
                answer = '\n\n'.join(f"• {part}" if not part.startswith('-') else part for part in answer_parts)
            else:
                answer = '\n\n'.join(answer_parts)
        else:
            answer = best_section[:500].strip() + '...'
    else:
        # Last resort: get first meaningful content
        clean_text = re.sub(r'\s*\|\s*', ', ', chunk_text)
        clean_text = re.sub(r'\s*—\s*', ' - ', clean_text)
        answer = clean_text[:400].strip() + '...'
    
    # Final cleanup
    answer = re.sub(r'\n{3,}', '\n\n', answer)
    answer = re.sub(r'\s+', ' ', answer)  # Normalize spaces
    answer = re.sub(r'\s*\n\s*', '\n', answer)  # Clean newlines
    answer = answer.strip()
    
    # Calculate confidence based on keyword overlap
    max_overlap = chunk_scores[0][2]
    confidence = min(0.8, max_overlap / len(query_words)) if query_words else 0.3
    
    # Extract sources
    sources = list(set([
        cs[1].get('metadata', {}).get('source', 'Unknown')
        for cs in chunk_scores[:3]
    ]))
    
    return {
        "answer": answer,
        "sources": sources,
        "confidence": round(confidence, 2),
        "method": "fallback"
    }


# ============================================================================
# MAIN ANSWER GENERATION FUNCTION
# ============================================================================

def generate_answer(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    max_tokens: int = 512,
    backend: str = "auto",
    use_simple_prompt: bool = False
) -> Dict[str, Any]:
    """
    Generate an answer to a query using retrieved chunks and LLM.
    
    This is the main function students will use for RAG answer generation.
    
    Args:
        query: The user's question
        retrieved_chunks: List of retrieved chunks from retriever.get_top_k()
            Format: [{"id": str, "text": str, "metadata": dict, "score": float}]
        max_tokens: Maximum tokens for LLM generation (default: 512)
        backend: LLM backend to use ("auto", "openai", "gemini", "ollama")
        use_simple_prompt: Use simpler prompt template if True
        
    Returns:
        Dict with keys:
            - answer (str): The generated answer
            - sources (List[str]): List of source documents used
            - confidence (float): Confidence score 0.0-1.0
            - method (str): Generation method used ("llm" or "fallback")
    """
    logger.info(f"Generating answer for query: '{query[:50]}...'")
    
    # Handle empty chunks case
    if not retrieved_chunks:
        logger.warning("No retrieved chunks provided - returning empty context message")
        return {
            "answer": "I couldn't find relevant information in the dataset to answer your question. Please try rephrasing or asking about topics covered in the documents.",
            "sources": [],
            "confidence": 0.0,
            "method": "empty_context"
        }
    
    # Format context from chunks (use top 3)
    context_parts = []
    sources_map = {}  # Map chunk index to source
    
    for idx, chunk in enumerate(retrieved_chunks[:3], start=1):
        chunk_text = chunk.get('text', '')
        metadata = chunk.get('metadata', {})
        source = metadata.get('source', 'Unknown')
        chunk_idx = metadata.get('chunk_index', '?')
        score = chunk.get('score', 0.0)
        
        sources_map[idx] = source
        
        # Format context with clear separators
        context_part = f"""--- CHUNK [{idx}] ---
Source: {source} (Chunk {chunk_idx})
Relevance Score: {score:.3f}
Content:
{chunk_text}
"""
        context_parts.append(context_part)
    
    full_context = "\n\n".join(context_parts)
    
    # Select prompt template
    prompt_template = SIMPLE_PROMPT_TEMPLATE if use_simple_prompt else SYSTEM_PROMPT_TEMPLATE
    
    # Create full prompt
    full_prompt = prompt_template.format(context=full_context, question=query)
    
    logger.debug(f"Prompt length: {len(full_prompt)} characters")
    
    # Try to call LLM using new config-based API
    llm_response = None
    if CONFIG_AVAILABLE:
        try:
            # Use the provider from config or the specified backend
            provider = backend if backend != "auto" else LLM_PROVIDER
            llm_response = call_llm(
                prompt=full_prompt,
                provider=provider,
                config={
                    'max_tokens': max_tokens,
                    'temperature': 0.7
                }
            )
            if llm_response:
                logger.info(f"Successfully got response from LLM (provider: {provider})")
        except Exception as e:
            logger.error(f"Error calling LLM via llm_api: {e}")
    else:
        # Fallback to old call_llm_api if config not available
        llm_response = call_llm_api(full_prompt, backend=backend, max_tokens=max_tokens)
    
    if llm_response:
        # Parse LLM response
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                response_data = json.loads(json_match.group())
            else:
                response_data = json.loads(llm_response)
            
            # Extract fields
            answer = response_data.get('answer', llm_response)
            sources_used = response_data.get('sources_used', [1])
            confidence = float(response_data.get('confidence', 0.7))
            
            # Map source indices to actual source names
            sources = [sources_map.get(i, 'Unknown') for i in sources_used if i in sources_map]
            if not sources:
                sources = list(sources_map.values())[:2]
            
            logger.info(f"LLM generation successful (confidence: {confidence:.2f})")
            
            return {
                "answer": answer,
                "sources": sources,
                "confidence": round(confidence, 2),
                "method": "llm"
            }
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Could not parse LLM response as JSON: {e}")
            logger.debug(f"Raw LLM response: {llm_response[:200]}...")
            
            # Use raw response as answer
            return {
                "answer": llm_response,
                "sources": list(sources_map.values()),
                "confidence": 0.6,
                "method": "llm_raw"
            }
    
    # Fallback to rule-based generation
    logger.info("LLM not available, using rule-based fallback")
    return rule_based_generate(query, retrieved_chunks)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def format_answer_for_display(result: Dict[str, Any]) -> str:
    """
    Format answer result for user-friendly display.
    
    Args:
        result: Result from generate_answer()
        
    Returns:
        str: Formatted answer string
    """
    answer = result.get('answer', 'No answer generated.')
    sources = result.get('sources', [])
    confidence = result.get('confidence', 0.0)
    method = result.get('method', 'unknown')
    
    output = [answer]
    
    if sources:
        output.append(f"\n\n📚 Sources: {', '.join(sources)}")
    
    output.append(f"\n🎯 Confidence: {confidence:.0%}")
    
    if method == "fallback":
        output.append("\n⚠️ Note: Generated using fallback method (LLM not available)")
    
    return ''.join(output)


# ============================================================================
# DEMO AND TESTING
# ============================================================================

if __name__ == "__main__":
    """
    Demo: Test answer generation with sample chunks.
    """
    print("=" * 80)
    print("ANSWER GENERATOR DEMO")
    print("=" * 80)
    
    # Sample retrieved chunks (simulating retriever output)
    sample_chunks = [
        {
            "id": "doc1_chunk0",
            "text": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves.",
            "metadata": {"source": "ai_basics.txt", "chunk_index": 0},
            "score": 0.85
        },
        {
            "id": "doc1_chunk1",
            "text": "Deep learning is a subset of machine learning that uses neural networks with multiple layers. These networks can learn complex patterns in large amounts of data and are particularly effective for tasks like image recognition and natural language processing.",
            "metadata": {"source": "ai_basics.txt", "chunk_index": 1},
            "score": 0.72
        },
        {
            "id": "doc2_chunk0",
            "text": "Neural networks are computing systems inspired by biological neural networks in animal brains. They consist of interconnected nodes (neurons) organized in layers that process information using dynamic responses to external inputs.",
            "metadata": {"source": "neural_nets.pdf", "chunk_index": 0},
            "score": 0.68
        }
    ]
    
    # Test queries
    test_cases = [
        {
            "query": "What is machine learning?",
            "chunks": sample_chunks
        },
        {
            "query": "How do neural networks work?",
            "chunks": sample_chunks[2:]  # Only neural network chunk
        },
        {
            "query": "What is quantum computing?",
            "chunks": []  # Empty chunks
        }
    ]
    
    print("\nTesting answer generation with different scenarios...\n")
    print("Note: Since API keys are not configured, will use rule-based fallback.\n")
    print("-" * 80)
    
    for idx, test_case in enumerate(test_cases, 1):
        print(f"\n{'=' * 80}")
        print(f"TEST CASE {idx}")
        print("=" * 80)
        print(f"Query: {test_case['query']}")
        print(f"Chunks provided: {len(test_case['chunks'])}")
        print("-" * 80)
        
        # Generate answer
        result = generate_answer(
            query=test_case['query'],
            retrieved_chunks=test_case['chunks'],
            max_tokens=512,
            backend="auto"
        )
        
        # Display result
        print("\nRESULT:")
        print(format_answer_for_display(result))
        print("\nRaw result data:")
        print(f"  Method: {result.get('method', 'unknown')}")
        print(f"  Sources: {result.get('sources', [])}")
        print(f"  Confidence: {result.get('confidence', 0.0):.2f}")
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print("\nTo use with real LLMs:")
    print("1. Set environment variables:")
    print("   - OPENAI_API_KEY for OpenAI")
    print("   - GOOGLE_API_KEY for Gemini")
    print("2. Uncomment API integration code in call_openai_api() or call_gemini_api()")
    print("3. For Ollama: Install Ollama and uncomment call_ollama() code")
    print("\nExample usage:")
    print("  from generator import generate_answer")
    print("  from retriever import get_top_k")
    print("  ")
    print("  chunks = get_top_k('your question', k=4)")
    print("  result = generate_answer('your question', chunks)")
    print("  print(result['answer'])")


# ============================================================================
# CLASS WRAPPER FOR COMPATIBILITY
# ============================================================================

class ResponseGenerator:
    """
    Wrapper class for generate_answer function to provide compatibility
    with app_github.py which expects a class-based interface.
    """
    
    def __init__(self, temperature: float = 0.7):
        """
        Initialize the ResponseGenerator.
        
        Args:
            temperature: Generation temperature (currently not used in function-based approach)
        """
        self.temperature = temperature
        logger.info("ResponseGenerator initialized")
    
    def generate(self, query: str, retrieved_docs: List[Dict[str, Any]], 
                 temperature: float = None) -> str:
        """
        Generate a response based on query and retrieved documents.
        
        Args:
            query: The user's question
            retrieved_docs: List of retrieved document chunks
            temperature: Optional temperature override
            
        Returns:
            str: The generated answer
        """
        # Call the underlying generate_answer function
        result = generate_answer(query, retrieved_docs)
        return result.get('answer', 'Unable to generate answer.')

