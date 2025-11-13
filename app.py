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
"""
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import logging
from datetime import datetime
import shutil

import gradio as gr

# Import our RAG modules
from ingestion import DocumentIngestor
from embeddings_and_chroma_setup import ChromaDBSetup
from retriever import Retriever
from generator import generate_answer, format_answer_for_display

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGChatbot:
    """
    RAG Chatbot with Gradio UI.
    Handles document upload, processing, and conversational Q&A.
    """
    
    def __init__(self):
        """Initialize the RAG chatbot components."""
        logger.info("Initializing RAG Chatbot...")
        
        # Initialize components
        self.ingestor = DocumentIngestor()
        self.chunker = TextChunker(
            chunk_size_min=CHUNK_SIZE_MIN,
            chunk_size_max=CHUNK_SIZE_MAX,
            overlap_percent=CHUNK_OVERLAP_PERCENT
        )
        self.vector_store = VectorStore()
        self.rag_engine = RAGEngine(self.vector_store)
        
        # Track uploaded documents
        self.uploaded_docs = []
        
        logger.info("RAG Chatbot initialized successfully")
    
    def process_documents(
        self,
        files: List[gr.File],
        progress=gr.Progress()
    ) -> str:
        """
        Process uploaded documents and add to vector store.
        
        Args:
            files: List of uploaded file objects
            progress: Gradio progress tracker
            
        Returns:
            Status message
        """
        if not files:
            return "⚠️ No files uploaded. Please upload at least one document."
        
        try:
            progress(0, desc="Starting document processing...")
            processed_count = 0
            failed_files = []
            
            for i, file in enumerate(files):
                try:
                    # Update progress
                    progress((i + 1) / len(files), desc=f"Processing {Path(file.name).name}...")
                    
                    # Ingest document
                    logger.info(f"Processing file: {file.name}")
                    text, metadata = self.ingestor.ingest_document(file.name)
                    
                    # Chunk text
                    chunks = self.chunker.chunk_text(text, metadata["source"], metadata)
                    
                    # Add to vector store
                    self.vector_store.add_chunks(chunks)
                    
                    # Track uploaded document
                    self.uploaded_docs.append(metadata["source"])
                    processed_count += 1
                    
                except DocumentIngestionError as e:
                    logger.error(f"Failed to process {file.name}: {str(e)}")
                    failed_files.append(f"{Path(file.name).name}: {str(e)}")
                except Exception as e:
                    logger.error(f"Unexpected error processing {file.name}: {str(e)}")
                    failed_files.append(f"{Path(file.name).name}: Unexpected error")
            
            # Generate status message
            status_parts = []
            
            if processed_count > 0:
                stats = self.vector_store.get_stats()
                status_parts.append(
                    f"✅ Successfully processed {processed_count} document(s)!\n"
                    f"📊 Total chunks in database: {stats['total_chunks']}\n"
                    f"📁 Unique documents: {stats['unique_sources']}"
                )
            
            if failed_files:
                status_parts.append(
                    f"\n⚠️ Failed to process {len(failed_files)} file(s):\n" +
                    "\n".join(f"  • {f}" for f in failed_files)
                )
            
            return "\n".join(status_parts) if status_parts else "⚠️ No documents were processed."
            
        except Exception as e:
            logger.error(f"Error in process_documents: {str(e)}")
            return f"❌ Error processing documents: {str(e)}"
    
    def chat(
        self,
        message: str,
        history: List[Tuple[str, str]]
    ) -> Tuple[str, List[Tuple[str, str]]]:
        """
        Handle chat interactions.
        
        Args:
            message: User message
            history: Chat history
            
        Returns:
            Tuple of (response, updated_history)
        """
        if not message or not message.strip():
            return "", history
        
        # Check if documents are uploaded
        if self.vector_store.collection.count() == 0:
            response = (
                "⚠️ No documents have been uploaded yet. "
                "Please upload documents using the 'Upload Documents' tab first."
            )
            history.append((message, response))
            return "", history
        
        try:
            # Generate answer using RAG
            result = self.rag_engine.generate_answer(message)
            
            # Format response with sources
            answer = result["answer"]
            
            if result.get("context_found") and result.get("sources"):
                sources_text = "\n\n**Sources:**\n" + "\n".join(
                    f"• {src['name']} (relevance: {src['similarity_score']:.2f})"
                    for src in result["sources"]
                )
                response = answer + sources_text
            else:
                response = answer
            
            # Update history
            history.append((message, response))
            
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            response = f"❌ Error generating response: {str(e)}"
            history.append((message, response))
        
        return "", history
    
    def clear_chat(self) -> List:
        """Clear chat history."""
        return []
    
    def get_database_stats(self) -> str:
        """Get current database statistics."""
        try:
            stats = self.vector_store.get_stats()
            
            if stats["total_chunks"] == 0:
                return "📊 Database is empty. Upload documents to get started."
            
            sources_list = "\n".join(f"  • {src}" for src in stats["sources"])
            
            return (
                f"📊 **Database Statistics**\n\n"
                f"Total chunks: {stats['total_chunks']}\n"
                f"Unique documents: {stats['unique_sources']}\n"
                f"Embedding model: {stats['embedding_model']}\n\n"
                f"**Documents:**\n{sources_list}"
            )
        except Exception as e:
            return f"❌ Error retrieving stats: {str(e)}"
    
    def clear_database(self) -> str:
        """Clear all documents from the database."""
        try:
            self.vector_store.clear()
            self.uploaded_docs = []
            return "✅ Database cleared successfully!"
        except Exception as e:
            return f"❌ Error clearing database: {str(e)}"


def create_interface() -> gr.Blocks:
    """
    Create the Gradio interface.
    
    Returns:
        Gradio Blocks interface
    """
    chatbot = RAGChatbot()
    
    with gr.Blocks(
        title="RAG Chatbot",
        theme=gr.themes.Soft()
    ) as interface:
        
        gr.Markdown(
            """
            # 🤖 RAG Chatbot
            
            Upload your documents (PDF, TXT, DOCX) and ask questions about their content!
            
            The chatbot uses **Retrieval-Augmented Generation (RAG)** to provide accurate answers
            with source attribution.
            """
        )
        
        with gr.Tabs():
            # Tab 1: Upload Documents
            with gr.Tab("📤 Upload Documents"):
                gr.Markdown(
                    """
                    ### Upload your documents
                    
                    Supported formats: PDF, TXT, DOCX
                    
                    Documents will be processed, chunked, and stored in a vector database
                    for efficient retrieval.
                    """
                )
                
                upload_files = gr.File(
                    label="Select files to upload",
                    file_count="multiple",
                    file_types=[".pdf", ".txt", ".docx"]
                )
                
                upload_button = gr.Button("🚀 Process Documents", variant="primary")
                upload_status = gr.Textbox(
                    label="Status",
                    lines=5,
                    interactive=False
                )
                
                upload_button.click(
                    fn=chatbot.process_documents,
                    inputs=[upload_files],
                    outputs=[upload_status]
                )
            
            # Tab 2: Chat
            with gr.Tab("💬 Chat"):
                gr.Markdown(
                    """
                    ### Ask questions about your documents
                    
                    The chatbot will search through your uploaded documents and provide
                    answers with source citations.
                    """
                )
                
                chatbot_ui = gr.Chatbot(
                    label="Conversation",
                    height=500,
                    show_copy_button=True
                )
                
                with gr.Row():
                    msg_input = gr.Textbox(
                        label="Your question",
                        placeholder="Ask a question about your documents...",
                        scale=4
                    )
                    send_button = gr.Button("Send", variant="primary", scale=1)
                
                with gr.Row():
                    clear_button = gr.Button("🗑️ Clear Chat")
                
                # Event handlers
                msg_input.submit(
                    fn=chatbot.chat,
                    inputs=[msg_input, chatbot_ui],
                    outputs=[msg_input, chatbot_ui]
                )
                
                send_button.click(
                    fn=chatbot.chat,
                    inputs=[msg_input, chatbot_ui],
                    outputs=[msg_input, chatbot_ui]
                )
                
                clear_button.click(
                    fn=chatbot.clear_chat,
                    outputs=[chatbot_ui]
                )
            
            # Tab 3: Database Management
            with gr.Tab("⚙️ Database"):
                gr.Markdown(
                    """
                    ### Manage your document database
                    
                    View statistics and manage stored documents.
                    """
                )
                
                stats_output = gr.Textbox(
                    label="Database Statistics",
                    lines=10,
                    interactive=False
                )
                
                with gr.Row():
                    refresh_button = gr.Button("🔄 Refresh Stats", variant="secondary")
                    clear_db_button = gr.Button("🗑️ Clear Database", variant="stop")
                
                clear_status = gr.Textbox(
                    label="Status",
                    lines=2,
                    interactive=False
                )
                
                # Event handlers
                refresh_button.click(
                    fn=chatbot.get_database_stats,
                    outputs=[stats_output]
                )
                
                clear_db_button.click(
                    fn=chatbot.clear_database,
                    outputs=[clear_status]
                ).then(
                    fn=chatbot.get_database_stats,
                    outputs=[stats_output]
                )
                
                # Load stats on tab load
                interface.load(
                    fn=chatbot.get_database_stats,
                    outputs=[stats_output]
                )
        
        gr.Markdown(
            """
            ---
            
            **Note:** This chatbot uses AI to generate responses. Always verify important information.
            """
        )
    
    return interface


def main():
    """Main entry point for the application."""
    logger.info("Starting RAG Chatbot application...")
    
    # Create and launch interface
    interface = create_interface()
    
    interface.launch(
        share=GRADIO_SHARE,
        server_port=GRADIO_SERVER_PORT,
        server_name=GRADIO_SERVER_NAME,
        show_error=True
    )


if __name__ == "__main__":
    main()
