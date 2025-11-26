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
            # Initialize retriever with explicit parameters
            logger.info("Loading document retriever...")
            self.retriever = DocumentRetriever(
                chroma_db_path="./chroma_db",
                collection_name="capstone_docs",
                similarity_threshold=0.2
            )
            logger.info("✅ Document retriever loaded successfully")
            
            # Initialize generator
            logger.info("Loading response generator...")
            self.generator = ResponseGenerator()
            logger.info("✅ Response generator loaded successfully")
            
            logger.info("✅ RAG Chatbot initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize RAG Chatbot: {e}", exc_info=True)
            raise
    
    def upload_documents(self, files):
        """Upload and index new documents"""
        if not files:
            return "⚠️ No files uploaded."
        
        try:
            from ingestion import DocumentIngestor
            from embeddings_and_chroma_setup import ChromaDBSetup
            
            # Initialize document processor
            ingestor = DocumentIngestor(
                documents_dir="data/documents/company_data",
                chunk_size_min=500,
                chunk_size_max=800,
                overlap_percent=10
            )
            
            all_chunks = []
            processed = []
            failed = []
            
            for file in files:
                try:
                    chunks = ingestor.load_and_process_file(file.name)
                    if chunks:
                        all_chunks.extend(chunks)
                        processed.append(file.name.split('/')[-1])
                    else:
                        failed.append(file.name.split('/')[-1])
                except Exception as e:
                    failed.append(f"{file.name.split('/')[-1]}: {str(e)}")
            
            # Index new documents
            if all_chunks:
                chroma_setup = ChromaDBSetup(
                    chroma_db_path="./chroma_db",
                    collection_name="capstone_docs",
                    replace_duplicates=True
                )
                stats = chroma_setup.upsert_chunks(all_chunks)
                
                # Reinitialize retriever to pick up new docs
                self.retriever = DocumentRetriever(
                    chroma_db_path="./chroma_db",
                    collection_name="capstone_docs",
                    similarity_threshold=0.2
                )
                
                result = f"✅ Processed {len(processed)} file(s)\n"
                result += f"📊 Added {stats['new']} chunks, Updated {stats['updated']}\n"
                result += f"📚 Total documents: {stats['final_count']}"
                
                if failed:
                    result += f"\n\n⚠️ Failed: {', '.join(failed)}"
                
                return result
            else:
                return "❌ No valid documents to process"
                
        except Exception as e:
            logger.error(f"Upload error: {e}", exc_info=True)
            return f"❌ Error: {str(e)}"
    
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
            retrieved_docs = self.retriever.get_top_k(
                query=question,
                k=num_sources
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
            # 🏢 Company Data Q&A System
            
            Ask questions about company documents including financials, HR policies, salaries, performance metrics, and strategic initiatives.
            
            **Pre-indexed Company Data (11 Documents):**
            - Financial & Profitability Reports
            - Salary & Compensation Data
            - Department Performance & Budgets
            - Employee Contributions & Impact
            - Strategic Initiatives & OKRs
            - Training & Development Programs
            - Benefits & Work Time Analysis
            
            You can also **upload your own documents** (PDF, TXT, DOCX) to expand the knowledge base!
            """
        )
        
        # Upload Section
        with gr.Accordion("📤 Upload New Documents", open=False):
            with gr.Row():
                file_upload = gr.File(
                    label="Select Files",
                    file_count="multiple",
                    file_types=[".pdf", ".txt", ".docx"],
                    type="filepath"
                )
            upload_btn = gr.Button("Upload & Index", variant="secondary")
            upload_status = gr.Textbox(label="Upload Status", lines=4, interactive=False)
        
        gr.Markdown("---")
        
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
        
        upload_btn.click(
            fn=chatbot.upload_documents,
            inputs=[file_upload],
            outputs=[upload_status]
        )
        
        # Example questions
        gr.Examples(
            examples=[
                ["What is the average salary for engineers?", 3, 0.7],
                ["How much profit did the company make in 2025?", 3, 0.7],
                ["Which department has the highest budget?", 3, 0.7],
                ["What are the key strategic initiatives for 2025?", 3, 0.7],
                ["What training programs are available for employees?", 3, 0.7],
                ["How is employee performance measured?", 3, 0.7],
            ],
            inputs=[question_input, num_sources, temperature],
            label="Example Questions - Try These!"
        )
        
        gr.Markdown(
            """
            ---
            **Technology:** Uses Retrieval-Augmented Generation (RAG) with ChromaDB vector database 
            and sentence-transformers for accurate, source-backed answers from company documents.
            
            **Data Source:** 11 company documents covering financials, HR, operations, and strategic planning (2025).
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
