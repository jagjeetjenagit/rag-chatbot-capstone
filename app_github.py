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
            import tempfile
            import shutil
            from pathlib import Path
            
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
            
            # Create temp directory for uploaded files
            with tempfile.TemporaryDirectory() as temp_dir:
                for file in files:
                    try:
                        # file.name contains the path to the temporary file
                        file_path = Path(file.name)
                        
                        # Process the file directly from its temp location
                        chunks = ingestor.load_and_process_file(str(file_path))
                        
                        if chunks:
                            all_chunks.extend(chunks)
                            processed.append(file_path.name)
                        else:
                            failed.append(file_path.name)
                    except Exception as e:
                        failed.append(f"{Path(file.name).name if hasattr(file, 'name') else 'Unknown'}: {str(e)}")
                
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
        temperature: float = 0.7,
        history: List = None
    ) -> Tuple[List, str]:
        """
        Answer a question using RAG approach
        
        Args:
            question: User's question
            num_sources: Number of source documents to retrieve
            temperature: Generation temperature (0.0 to 1.0)
            history: Chat history list
            
        Returns:
            Tuple of (updated_history, sources)
        """
        try:
            if not question or not question.strip():
                return history or [], ""
            
            if history is None:
                history = []
            
            logger.info(f"Processing question: {question[:100]}...")
            
            # Retrieve relevant documents
            logger.info(f"Retrieving top {num_sources} relevant documents...")
            retrieved_docs = self.retriever.get_top_k(
                query=question,
                k=num_sources
            )
            
            if not retrieved_docs:
                answer = "No relevant documents found. Please try rephrasing your question."
                history.append((question, answer))
                return history, ""
            
            logger.info(f"✅ Retrieved {len(retrieved_docs)} documents")
            
            # Generate response
            logger.info("Generating response...")
            answer = self.generator.generate(
                query=question,
                retrieved_docs=retrieved_docs,
                temperature=temperature
            )
            
            # Clean up the answer formatting
            answer = self._clean_answer(answer)
            
            # Format sources
            sources = self._format_sources(retrieved_docs)
            
            # Add to history
            history.append((question, answer))
            
            logger.info("✅ Response generated successfully")
            return history, sources
            
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            logger.error(error_msg, exc_info=True)
            history.append((question, f"❌ {error_msg}"))
            return history, ""
    
    def _clean_answer(self, answer: str) -> str:
        """Clean and format the answer for better readability"""
        import re
        
        # Just do minimal cleanup - let the answer keep its structure
        # Remove excessive blank lines
        answer = re.sub(r'\n{3,}', '\n\n', answer)
        
        # Remove any remaining document metadata artifacts
        answer = re.sub(r'\*\*Report Period\*\*:[^\n]+', '', answer)
        answer = re.sub(r'\*\*Prepared by\*\*:[^\n]+', '', answer)
        answer = re.sub(r'\*\*Report Date\*\*:[^\n]+', '', answer)
        answer = re.sub(r'\*\*Classification\*\*:[^\n]+', '', answer)
        
        # Remove horizontal rules
        answer = re.sub(r'^-{3,}\s*$', '', answer, flags=re.MULTILINE)
        
        return answer.strip()
    
    def _format_sources(self, docs: List[dict]) -> str:
        """Format retrieved documents as source citations"""
        if not docs:
            return "No sources available"
        
        sources = []
        for i, doc in enumerate(docs, 1):
            source_file = doc.get('metadata', {}).get('source', 'Unknown')
            chunk_idx = doc.get('metadata', {}).get('chunk_index', 0)
            content = doc.get('content', '')[:300]  # Show more context
            
            # Clean up the content preview
            content = content.replace('\n', ' ').strip()
            
            sources.append(
                f"### 📄 Source {i}: {source_file}\n"
                f"**Chunk:** {chunk_idx}\n\n"
                f"> {content}...\n"
            )
        
        return "\n---\n\n".join(sources)


def create_gradio_interface():
    """Create and configure Gradio interface"""
    
    # Initialize chatbot
    logger.info("Initializing RAG Chatbot for Gradio interface...")
    chatbot = RAGChatbot()
    
    def process_question(question, num_sources, temperature, history):
        """Process question and return answer with sources"""
        history, sources = chatbot.answer_question(question, num_sources, temperature, history)
        return history, sources, ""  # Return empty string to clear input
    
    # Create Gradio interface
    with gr.Blocks(
        title="RAG Chatbot - Document Q&A System",
        theme=gr.themes.Soft(),
        css="""
        .message-bubble-border {
            border-radius: 12px !important;
        }
        """
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
                    type="file"
                )
            upload_btn = gr.Button("Upload & Index", variant="secondary")
            upload_status = gr.Textbox(label="Upload Status", lines=4, interactive=False)
        
        gr.Markdown("---")
        
        # Chat interface
        with gr.Row():
            with gr.Column(scale=3):
                chatbot_display = gr.Chatbot(
                    label="Conversation",
                    height=500,
                    bubble_full_width=False,
                    show_label=True,
                    elem_classes="message-bubble-border"
                )
        
        with gr.Row():
            with gr.Column(scale=2):
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="e.g., What is the average salary for engineers?",
                    lines=2,
                    show_label=False
                )
                
                submit_btn = gr.Button("Ask Question", variant="primary", size="lg")
                
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
                
                clear_btn = gr.Button("Clear Chat", size="lg")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📚 Sources & References")
                sources_output = gr.Markdown(
                    value="",
                    elem_classes="sources-box"
                )
        
        # Event handlers
        submit_btn.click(
            fn=process_question,
            inputs=[question_input, num_sources, temperature, chatbot_display],
            outputs=[chatbot_display, sources_output, question_input]
        )
        
        question_input.submit(
            fn=process_question,
            inputs=[question_input, num_sources, temperature, chatbot_display],
            outputs=[chatbot_display, sources_output, question_input]
        )
        
        clear_btn.click(
            fn=lambda: ([], "", ""),
            inputs=[],
            outputs=[chatbot_display, sources_output, question_input]
        )
        
        upload_btn.click(
            fn=chatbot.upload_documents,
            inputs=[file_upload],
            outputs=[upload_status]
        )
        
        # Example questions
        gr.Examples(
            examples=[
                ["What is the average salary for engineers?"],
                ["How much profit did the company make in 2025?"],
                ["Which department has the highest budget?"],
                ["What are the key strategic initiatives for 2025?"],
                ["What training programs are available for employees?"],
                ["How is employee performance measured?"],
            ],
            inputs=[question_input],
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
