"""
RAG Chatbot Application for GitHub Hosting
Simplified version optimized for deployment on GitHub Pages or cloud platforms
"""

import os
import logging
import gradio as gr
from typing import List, Tuple, Optional
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import RAG components
try:
    from retriever import DocumentRetriever
    from generator import ResponseGenerator
    logger.info("✅ Successfully imported RAG components")
except ImportError as e:
    logger.error(f"❌ Failed to import RAG components: {e}")
    raise


class RAGChatbot:
    """RAG-based chatbot for question answering"""
    
    def __init__(self):
        """Initialize the RAG chatbot with retriever and generator"""
        logger.info("Initializing RAG Chatbot...")
        
        try:
            # Initialize retriever
            logger.info("Loading document retriever...")
            self.retriever = DocumentRetriever()
            logger.info("✅ Document retriever loaded successfully")
            
            # Initialize generator
            logger.info("Loading response generator...")
            self.generator = ResponseGenerator()
            logger.info("✅ Response generator loaded successfully")
            
            logger.info("✅ RAG Chatbot initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize RAG Chatbot: {e}", exc_info=True)
            raise
    
    def answer_question(
        self, 
        question: str, 
        num_sources: int = 3,
        temperature: float = 0.7
    ) -> Tuple[str, str]:
        """
        Answer a question using RAG approach
        
        Args:
            question: User's question
            num_sources: Number of source documents to retrieve
            temperature: Generation temperature (0.0 to 1.0)
            
        Returns:
            Tuple of (answer, sources)
        """
        try:
            if not question or not question.strip():
                return "Please enter a valid question.", ""
            
            logger.info(f"Processing question: {question[:100]}...")
            
            # Retrieve relevant documents
            logger.info(f"Retrieving top {num_sources} relevant documents...")
            retrieved_docs = self.retriever.retrieve(
                query=question,
                n_results=num_sources
            )
            
            if not retrieved_docs:
                return "No relevant documents found. Please try rephrasing your question.", ""
            
            logger.info(f"✅ Retrieved {len(retrieved_docs)} documents")
            
            # Generate response
            logger.info("Generating response...")
            answer = self.generator.generate(
                query=question,
                retrieved_docs=retrieved_docs,
                temperature=temperature
            )
            
            # Format sources
            sources = self._format_sources(retrieved_docs)
            
            logger.info("✅ Response generated successfully")
            return answer, sources
            
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return f"❌ {error_msg}", ""
    
    def _format_sources(self, docs: List[dict]) -> str:
        """Format retrieved documents as source citations"""
        if not docs:
            return "No sources available"
        
        sources = []
        for i, doc in enumerate(docs, 1):
            source_file = doc.get('metadata', {}).get('source', 'Unknown')
            chunk_idx = doc.get('metadata', {}).get('chunk_index', 0)
            content = doc.get('content', '')[:200]
            
            sources.append(
                f"**Source {i}:** {source_file} (Chunk {chunk_idx})\n"
                f"```\n{content}...\n```\n"
            )
        
        return "\n".join(sources)


def create_gradio_interface():
    """Create and configure Gradio interface"""
    
    # Initialize chatbot
    logger.info("Initializing RAG Chatbot for Gradio interface...")
    chatbot = RAGChatbot()
    
    def process_question(question, num_sources, temperature):
        """Process question and return answer with sources"""
        answer, sources = chatbot.answer_question(question, num_sources, temperature)
        return answer, sources
    
    # Create Gradio interface
    with gr.Blocks(
        title="RAG Chatbot - Document Q&A System",
        theme=gr.themes.Soft()
    ) as demo:
        gr.Markdown(
            """
            # 📚 RAG Chatbot - Intelligent Document Q&A
            
            Ask questions about the indexed documents and get accurate answers with source citations.
            
            **How it works:**
            1. Enter your question in natural language
            2. The system retrieves relevant document chunks
            3. An AI generates an answer based on the retrieved context
            4. View source citations for transparency
            """
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="e.g., What is the average salary for engineers?",
                    lines=3
                )
                
                with gr.Row():
                    num_sources = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=3,
                        step=1,
                        label="Number of Sources",
                        info="How many document chunks to retrieve"
                    )
                    
                    temperature = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        label="Temperature",
                        info="Higher = more creative, Lower = more focused"
                    )
                
                submit_btn = gr.Button("Ask Question", variant="primary", size="lg")
                clear_btn = gr.Button("Clear", size="lg")
        
        with gr.Row():
            with gr.Column():
                answer_output = gr.Textbox(
                    label="Answer",
                    lines=10,
                    interactive=False
                )
                
                sources_output = gr.Markdown(
                    label="Sources"
                )
        
        # Event handlers
        submit_btn.click(
            fn=process_question,
            inputs=[question_input, num_sources, temperature],
            outputs=[answer_output, sources_output]
        )
        
        clear_btn.click(
            fn=lambda: ("", "", ""),
            inputs=[],
            outputs=[question_input, answer_output, sources_output]
        )
        
        # Example questions
        gr.Examples(
            examples=[
                ["What is machine learning?", 3, 0.7],
                ["What is the average salary for engineers?", 3, 0.7],
                ["How much profit did the company make in 2025?", 3, 0.7],
                ["What are the company's HR policies?", 3, 0.7],
                ["Which department contributes most to revenue?", 3, 0.7],
            ],
            inputs=[question_input, num_sources, temperature],
            label="Example Questions"
        )
        
        gr.Markdown(
            """
            ---
            **Note:** This chatbot uses Retrieval-Augmented Generation (RAG) to provide accurate, 
            source-backed answers from the indexed document collection.
            """
        )
    
    return demo


def main():
    """Main function to launch the application"""
    try:
        logger.info("="*80)
        logger.info("Starting RAG Chatbot Application")
        logger.info("="*80)
        
        # Create and launch Gradio interface
        demo = create_gradio_interface()
        
        logger.info("Launching Gradio interface...")
        demo.launch(
            server_name="0.0.0.0",  # Allow external access
            server_port=7860,
            share=False,  # Set to True for Gradio sharing link
            show_error=True
        )
        
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
