"""
Gradio RAG Chatbot Application.
Provides a web interface for document upload and conversational Q&A using RAG.

Features:
- Multi-file upload (.pdf, .txt, .docx)
- Real-time document processing and indexing
- Chat interface with conversation history
- Source attribution and confidence scores
- Retrieved chunk snippets display
- Database statistics tracking

Usage:
    python app.py

Then open http://localhost:7860 in your browser.
"""
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import logging
from datetime import datetime

import gradio as gr

# Import our RAG modules
from ingestion import DocumentIngestor
from retriever import get_top_k
from generator import generate_answer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "capstone_docs"
DOCUMENTS_DIR = "data/documents"
GRADIO_PORT = 7860


class RAGChatbotApp:
    """
    Main RAG Chatbot Application with Gradio UI.
    """
    
    def __init__(self):
        """Initialize the RAG chatbot components."""
        logger.info("Starting RAG Chatbot application...")
        
        # Create documents directory if it doesn't exist
        os.makedirs(DOCUMENTS_DIR, exist_ok=True)
        
        # Initialize document ingestor
        self.ingestor = DocumentIngestor(
            documents_dir=DOCUMENTS_DIR,
            chunk_size_min=500,
            chunk_size_max=800,
            overlap_percent=10
        )
        
        # Track application state
        self.last_upload_time = None
        self.conversation_count = 0
        self.total_docs = 0
        
        # Try to get initial document count
        try:
            from retriever import Retriever
            retriever = Retriever(
                chroma_db_path=CHROMA_DB_PATH,
                collection_name=COLLECTION_NAME
            )
            stats = retriever.get_collection_stats()
            self.total_docs = stats['total_documents']
            logger.info(f"Initial document count: {self.total_docs}")
        except Exception as e:
            logger.warning(f"Could not get initial stats: {e}")
        
        logger.info("RAG Chatbot initialized successfully")
    
    def upload_documents(
        self,
        files: List[gr.File],
        progress=gr.Progress()
    ) -> Tuple[str, str]:
        """
        Process uploaded documents and add to vector database.
        
        Args:
            files: List of uploaded file objects from Gradio
            progress: Gradio progress tracker
            
        Returns:
            Tuple of (status_message, updated_stats)
        """
        if not files:
            return "⚠️ No files uploaded. Please select at least one document.", self.get_stats()
        
        try:
            from embeddings_and_chroma_setup import ChromaDBSetup
            
            progress(0, desc="Starting document upload...")
            
            processed_files = []
            failed_files = []
            all_chunks = []
            
            # Process each file
            for i, file in enumerate(files):
                try:
                    file_path = Path(file.name)
                    filename = file_path.name
                    
                    progress((i + 1) / len(files), desc=f"Processing {filename}...")
                    logger.info(f"Processing file: {filename}")
                    
                    # Load and process file
                    chunks = self.ingestor.load_and_process_file(file.name)
                    
                    if chunks:
                        all_chunks.extend(chunks)
                        processed_files.append(filename)
                        logger.info(f"Successfully processed {filename}: {len(chunks)} chunks")
                    else:
                        failed_files.append(f"{filename}: No text extracted")
                        logger.warning(f"No chunks extracted from {filename}")
                    
                except Exception as e:
                    logger.error(f"Error processing {file.name}: {str(e)}")
                    failed_files.append(f"{file_path.name if 'file_path' in locals() else 'Unknown'}: {str(e)}")
            
            # Upsert chunks into ChromaDB
            if all_chunks:
                progress(0.9, desc="Indexing documents in vector database...")
                
                chroma_setup = ChromaDBSetup(
                    chroma_db_path=CHROMA_DB_PATH,
                    collection_name=COLLECTION_NAME,
                    replace_duplicates=True
                )
                
                upsert_stats = chroma_setup.upsert_chunks(all_chunks)
                
                # Update last upload time
                self.last_upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.total_docs = upsert_stats['final_count']
            
            # Build status message
            status_parts = []
            
            if processed_files:
                status_parts.append(
                    f"✅ Successfully processed {len(processed_files)} file(s):\n" +
                    "\n".join(f"  • {f}" for f in processed_files)
                )
                if all_chunks:
                    status_parts.append(
                        f"\n📊 Indexing Statistics:\n"
                        f"  • Total chunks: {upsert_stats['total']}\n"
                        f"  • New: {upsert_stats['new']}\n"
                        f"  • Updated: {upsert_stats['updated']}\n"
                        f"  • Final count: {upsert_stats['final_count']}"
                    )
            
            if failed_files:
                status_parts.append(
                    f"\n⚠️ Failed to process {len(failed_files)} file(s):\n" +
                    "\n".join(f"  • {f}" for f in failed_files)
                )
            
            if not status_parts:
                status_message = "⚠️ No documents were successfully processed."
            else:
                status_message = "\n".join(status_parts)
            
            return status_message, self.get_stats()
            
        except Exception as e:
            logger.error(f"Error in upload_documents: {str(e)}", exc_info=True)
            return f"❌ Error uploading documents: {str(e)}", self.get_stats()
    
    def chat(
        self,
        message: str,
        history: List[Tuple[str, str]],
        k_value: int
    ) -> Tuple[List[Tuple[str, str]], str, str]:
        """
        Handle chat interactions with RAG pipeline.
        
        Args:
            message: User's question
            history: Conversation history
            k_value: Number of chunks to retrieve
            
        Returns:
            Tuple of (updated_history, retrieved_chunks_display, stats)
        """
        if not message or not message.strip():
            return history, "", self.get_stats()
        
        # Check if documents are indexed
        if self.total_docs == 0:
            response = (
                "⚠️ No documents have been uploaded yet.\n\n"
                "Please upload documents using the 'Upload Documents' panel first."
            )
            history.append((message, response))
            return history, "", self.get_stats()
        
        try:
            # Step 1: Retrieve relevant chunks
            logger.info(f"Processing query: {message[:50]}...")
            retrieved_chunks = get_top_k(
                message,
                k=k_value,
                chroma_db_path=CHROMA_DB_PATH,
                collection_name=COLLECTION_NAME
            )
            
            if not retrieved_chunks:
                response = (
                    "⚠️ No relevant information found in the documents.\n\n"
                    "Try rephrasing your question or uploading more relevant documents."
                )
                history.append((message, response))
                return history, "No chunks retrieved.", self.get_stats()
            
            # Step 2: Generate answer
            result = generate_answer(message, retrieved_chunks, max_tokens=512)
            
            # Step 3: Format response
            answer = result['answer']
            sources = result['sources']
            confidence = result['confidence']
            method = result.get('method', 'unknown')
            
            # Build response with metadata
            response_parts = [answer]
            
            # Add confidence indicator
            confidence_emoji = "🟢" if confidence >= 0.7 else "🟡" if confidence >= 0.4 else "🔴"
            response_parts.append(f"\n\n{confidence_emoji} **Confidence:** {confidence:.0%}")
            
            # Add sources
            if sources:
                response_parts.append("\n\n📚 **Sources:**")
                for source in sources:
                    response_parts.append(f"  • {source}")
            
            # Add method note if fallback
            if method == "fallback":
                response_parts.append("\n\n⚠️ *Note: Generated using fallback method (LLM not available)*")
            
            response = "\n".join(response_parts)
            
            # Build chunks display
            chunks_display = self._format_chunks_display(retrieved_chunks)
            
            # Update conversation history
            history.append((message, response))
            self.conversation_count += 1
            
            logger.info(f"Query processed successfully. Confidence: {confidence:.2f}")
            
            return history, chunks_display, self.get_stats()
            
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}", exc_info=True)
            response = f"❌ Error generating response: {str(e)}"
            history.append((message, response))
            return history, "", self.get_stats()
    
    def _format_chunks_display(self, chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks for display."""
        if not chunks:
            return "No chunks retrieved."
        
        display_parts = ["### Retrieved Chunks\n"]
        
        for i, chunk in enumerate(chunks, 1):
            source = chunk['metadata'].get('source', 'Unknown')
            chunk_idx = chunk['metadata'].get('chunk_index', '?')
            score = chunk['score']
            text = chunk['text']
            
            text_preview = text[:300] + "..." if len(text) > 300 else text
            
            chunk_display = f"""
**Chunk {i}** | Score: {score:.3f}  
📄 Source: `{source}` (Chunk #{chunk_idx})  
```
{text_preview}
```
---
"""
            display_parts.append(chunk_display)
        
        return "\n".join(display_parts)
    
    def clear_conversation(self) -> Tuple[List, str, str]:
        """Clear conversation history."""
        logger.info("Clearing conversation history")
        self.conversation_count = 0
        return [], "", self.get_stats()
    
    def get_stats(self) -> str:
        """Get current database and application statistics."""
        try:
            last_upload = self.last_upload_time or "Never"
            
            stats_text = f"""### 📊 System Status

**Indexed Documents:** {self.total_docs}  
**Total Conversations:** {self.conversation_count}  
**Last Upload:** {last_upload}  
**Database:** {CHROMA_DB_PATH}  
**Collection:** {COLLECTION_NAME}
"""
            return stats_text
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return "### 📊 System Status\n\n❌ Error loading statistics"
    
    def build_ui(self) -> gr.Blocks:
        """Build the Gradio UI."""
        with gr.Blocks(
            title="RAG Chatbot",
            theme=gr.themes.Soft(),
            css="""
                .retrieved-chunks {max-height: 400px; overflow-y: auto;}
                .stats-panel {background-color: #f0f0f0; padding: 15px; border-radius: 5px;}
            """
        ) as app:
            
            gr.Markdown("""
# 📚 RAG Chatbot
            
Upload documents and ask questions. The chatbot will retrieve relevant information and generate answers with source citations.

**Supported formats:** PDF, TXT, DOCX
""")
            
            with gr.Row():
                # Left column: Chat interface
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(
                        label="Conversation",
                        height=500,
                        show_label=True,
                        type="tuples"
                    )
                    
                    with gr.Row():
                        msg_input = gr.Textbox(
                            label="Your Question",
                            placeholder="Type your question here...",
                            lines=2,
                            scale=4
                        )
                        send_btn = gr.Button("Send 🚀", scale=1, variant="primary")
                    
                    with gr.Row():
                        clear_btn = gr.Button("Clear Conversation 🗑️", size="sm")
                        k_slider = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=4,
                            step=1,
                            label="Chunks to Retrieve (k)",
                            info="Number of document chunks to retrieve"
                        )
                
                # Right column: Upload and stats
                with gr.Column(scale=1):
                    # Stats panel
                    stats_display = gr.Markdown(
                        value=self.get_stats(),
                        label="Statistics",
                        elem_classes="stats-panel"
                    )
                    
                    gr.Markdown("---")
                    
                    # Upload section
                    gr.Markdown("### 📤 Upload Documents")
                    
                    file_upload = gr.File(
                        label="Select Files",
                        file_types=[".pdf", ".txt", ".docx"],
                        file_count="multiple",
                        type="filepath"
                    )
                    
                    upload_btn = gr.Button("Upload & Index 📥", variant="primary")
                    
                    upload_status = gr.Textbox(
                        label="Upload Status",
                        lines=8,
                        max_lines=15,
                        interactive=False,
                        show_copy_button=True
                    )
            
            # Collapsible section for retrieved chunks
            with gr.Accordion("🔍 Retrieved Chunks", open=False):
                chunks_display = gr.Markdown(
                    value="",
                    elem_classes="retrieved-chunks"
                )
            
            # Event handlers
            send_btn.click(
                fn=self.chat,
                inputs=[msg_input, chatbot, k_slider],
                outputs=[chatbot, chunks_display, stats_display]
            ).then(
                fn=lambda: "",
                inputs=None,
                outputs=msg_input
            )
            
            msg_input.submit(
                fn=self.chat,
                inputs=[msg_input, chatbot, k_slider],
                outputs=[chatbot, chunks_display, stats_display]
            ).then(
                fn=lambda: "",
                inputs=None,
                outputs=msg_input
            )
            
            clear_btn.click(
                fn=self.clear_conversation,
                inputs=None,
                outputs=[chatbot, chunks_display, stats_display]
            )
            
            upload_btn.click(
                fn=self.upload_documents,
                inputs=[file_upload],
                outputs=[upload_status, stats_display]
            )
        
        return app
    
    def launch(self, **kwargs):
        """Launch the Gradio application."""
        app = self.build_ui()
        
        launch_params = {
            "server_name": "0.0.0.0",
            "server_port": GRADIO_PORT,
            "share": False,
            "show_error": True,
            "quiet": False
        }
        
        launch_params.update(kwargs)
        
        logger.info(f"Launching Gradio app on port {launch_params['server_port']}...")
        print("\n" + "=" * 80)
        print(f"🚀 RAG Chatbot is starting...")
        print(f"📍 Local URL: http://localhost:{launch_params['server_port']}")
        if launch_params['share']:
            print("🌐 Public URL will be generated...")
        print("=" * 80 + "\n")
        
        app.launch(**launch_params)


def main():
    """
    Main entry point for the application.
    
    Instructions:
    1. Run this app: python app.py
    2. Open browser: http://localhost:7860
    3. Upload documents in the right panel
    4. Ask questions in the chat interface
    """
    try:
        chatbot_app = RAGChatbotApp()
        chatbot_app.launch(
            server_port=GRADIO_PORT,
            share=False  # Set to True to create a public link
        )
    
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}", exc_info=True)
        print(f"\n❌ Error: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Check that all dependencies are installed: pip install -r requirements.txt")
        print("2. Verify port 7860 is not already in use")


if __name__ == "__main__":
    main()
