"""
Embeddings and ChromaDB Setup Script.
Loads document chunks from ingestion pipeline, computes embeddings, and stores in ChromaDB.

Uses:
- sentence-transformers 'all-MiniLM-L6-v2' for embeddings
- ChromaDB with persistent storage in ./chroma_db
- Safe re-run behavior with duplicate detection
"""
import os
import logging
from pathlib import Path
from typing import List, Dict, Any

# Import ChromaDB
try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    raise ImportError("chromadb not installed. Install with: pip install chromadb")

# Import shared client
from chroma_client import get_chroma_client

# Import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("sentence-transformers not installed. Install with: pip install sentence-transformers")

# Import ingestion function
from ingestion import ingest_documents

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChromaDBSetup:
    """
    Handles embedding computation and ChromaDB vector storage setup.
    """
    
    def __init__(
        self,
        chroma_db_path: str = "./chroma_db",
        collection_name: str = "capstone_docs",
        embedding_model: str = "all-MiniLM-L6-v2",
        replace_duplicates: bool = True
    ):
        """
        Initialize ChromaDB setup handler.
        
        Args:
            chroma_db_path: Path for persistent ChromaDB storage
            collection_name: Name of the Chroma collection
            embedding_model: SentenceTransformer model name
            replace_duplicates: If True, replace existing documents; if False, skip duplicates
        """
        self.chroma_db_path = Path(chroma_db_path)
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.replace_duplicates = replace_duplicates
        
        logger.info(f"Initializing ChromaDB setup: path={chroma_db_path}, collection={collection_name}")
        
        # Create chroma_db directory if it doesn't exist
        self.chroma_db_path.mkdir(parents=True, exist_ok=True)
        
        # Get shared ChromaDB client
        self.client = get_chroma_client(str(self.chroma_db_path))
        logger.info(f"ChromaDB client initialized with persistent storage at: {self.chroma_db_path}")
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        logger.info(f"Embedding model loaded. Dimension: {self.embedding_model.get_sentence_embedding_dimension()}")
        
        # Get or create collection
        self.collection = self._get_or_create_collection()
    
    def _get_or_create_collection(self):
        """
        Get existing collection or create new one.
        
        Returns:
            chromadb.Collection: The collection object
        """
        try:
            # Try to get existing collection
            collection = self.client.get_collection(name=self.collection_name)
            existing_count = collection.count()
            logger.info(f"Found existing collection '{self.collection_name}' with {existing_count} documents")
            return collection
        except Exception:
            # Create new collection
            logger.info(f"Creating new collection: {self.collection_name}")
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Document chunks with embeddings for RAG chatbot"}
            )
            logger.info(f"Collection '{self.collection_name}' created successfully")
            return collection
    
    def compute_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Compute embeddings for a list of texts using SentenceTransformer.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        logger.info(f"Computing embeddings for {len(texts)} texts...")
        
        # Compute embeddings in batch for efficiency
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Convert to list of lists (ChromaDB format)
        embeddings_list = embeddings.tolist()
        
        logger.info(f"Embeddings computed: {len(embeddings_list)} vectors of dimension {len(embeddings_list[0])}")
        return embeddings_list
    
    def get_existing_ids(self) -> set:
        """
        Get all existing document IDs in the collection.
        
        Returns:
            set: Set of existing document IDs
        """
        try:
            # Get all documents (just IDs)
            results = self.collection.get(include=[])
            existing_ids = set(results['ids']) if results['ids'] else set()
            logger.info(f"Found {len(existing_ids)} existing document IDs in collection")
            return existing_ids
        except Exception as e:
            logger.warning(f"Could not retrieve existing IDs: {e}")
            return set()
    
    def upsert_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Upsert document chunks into ChromaDB with embeddings.
        Handles duplicates based on replace_duplicates setting.
        
        Args:
            chunks: List of chunk dictionaries with 'id', 'text', and 'metadata'
            
        Returns:
            Dict[str, int]: Statistics about the upsert operation
        """
        if not chunks:
            logger.warning("No chunks provided for upserting")
            return {"total": 0, "new": 0, "updated": 0, "skipped": 0}
        
        logger.info(f"Starting upsert of {len(chunks)} chunks...")
        
        # Get existing IDs if not replacing duplicates
        existing_ids = set()
        if not self.replace_duplicates:
            existing_ids = self.get_existing_ids()
        
        # Separate chunks into new and duplicate
        chunks_to_process = []
        skipped_count = 0
        
        for chunk in chunks:
            chunk_id = chunk['id']
            if chunk_id in existing_ids and not self.replace_duplicates:
                skipped_count += 1
                logger.debug(f"Skipping duplicate ID: {chunk_id}")
            else:
                chunks_to_process.append(chunk)
        
        if skipped_count > 0:
            logger.info(f"Skipping {skipped_count} duplicate chunks (already in database)")
        
        if not chunks_to_process:
            logger.info("No new chunks to process")
            return {"total": len(chunks), "new": 0, "updated": 0, "skipped": skipped_count}
        
        # Extract data for ChromaDB
        ids = [chunk['id'] for chunk in chunks_to_process]
        texts = [chunk['text'] for chunk in chunks_to_process]
        metadatas = [chunk['metadata'] for chunk in chunks_to_process]
        
        # Compute embeddings
        embeddings = self.compute_embeddings(texts)
        
        # Upsert into ChromaDB
        logger.info(f"Upserting {len(chunks_to_process)} chunks into ChromaDB collection '{self.collection_name}'...")
        
        try:
            # ChromaDB's upsert will replace if ID exists, or add if new
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
            
            # Determine how many were new vs updated
            new_count = len([cid for cid in ids if cid not in existing_ids])
            updated_count = len(chunks_to_process) - new_count
            
            logger.info(f"Upsert complete: {new_count} new, {updated_count} updated, {skipped_count} skipped")
            
            # Verify final count
            final_count = self.collection.count()
            logger.info(f"Collection now contains {final_count} total documents")
            
            return {
                "total": len(chunks),
                "new": new_count,
                "updated": updated_count,
                "skipped": skipped_count,
                "final_count": final_count
            }
            
        except Exception as e:
            logger.error(f"Error during upsert: {str(e)}", exc_info=True)
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current collection.
        
        Returns:
            Dict[str, Any]: Collection statistics
        """
        try:
            count = self.collection.count()
            
            # Get sample document to check
            sample = None
            if count > 0:
                results = self.collection.get(limit=1, include=['metadatas'])
                if results['ids']:
                    sample = {
                        'id': results['ids'][0],
                        'metadata': results['metadatas'][0] if results['metadatas'] else None
                    }
            
            stats = {
                'collection_name': self.collection_name,
                'total_documents': count,
                'storage_path': str(self.chroma_db_path),
                'embedding_model': self.embedding_model_name,
                'embedding_dimension': self.embedding_model.get_sentence_embedding_dimension(),
                'sample_document': sample
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {'error': str(e)}


def load_chunks(documents_dir: str = "data/documents") -> List[Dict[str, Any]]:
    """
    Load and process document chunks from the ingestion pipeline.
    This is the main function to import and use from other scripts.
    
    Args:
        documents_dir: Directory containing documents to process
        
    Returns:
        List[Dict[str, Any]]: List of document chunks
    """
    logger.info(f"Loading chunks from: {documents_dir}")
    chunks = ingest_documents(documents_dir=documents_dir)
    logger.info(f"Loaded {len(chunks)} chunks from ingestion pipeline")
    return chunks


def setup_chromadb(
    documents_dir: str = "data/documents",
    chroma_db_path: str = "./chroma_db",
    collection_name: str = "capstone_docs",
    replace_duplicates: bool = True
) -> Dict[str, Any]:
    """
    Main setup function: Load chunks, compute embeddings, and store in ChromaDB.
    
    Args:
        documents_dir: Directory containing documents to ingest
        chroma_db_path: Path for ChromaDB persistent storage
        collection_name: Name of the ChromaDB collection
        replace_duplicates: If True, replace existing documents; if False, skip duplicates
        
    Returns:
        Dict[str, Any]: Setup statistics and results
    """
    try:
        # Load chunks from ingestion pipeline
        logger.info("=" * 80)
        logger.info("STEP 1: Loading document chunks")
        logger.info("=" * 80)
        chunks = load_chunks(documents_dir=documents_dir)
        
        if not chunks:
            logger.warning("No chunks loaded. Please add documents to the documents directory.")
            return {"status": "no_documents", "chunks_loaded": 0}
        
        # Initialize ChromaDB setup
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: Initializing ChromaDB and embedding model")
        logger.info("=" * 80)
        chroma_setup = ChromaDBSetup(
            chroma_db_path=chroma_db_path,
            collection_name=collection_name,
            replace_duplicates=replace_duplicates
        )
        
        # Upsert chunks into ChromaDB
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: Computing embeddings and upserting into ChromaDB")
        logger.info("=" * 80)
        upsert_stats = chroma_setup.upsert_chunks(chunks)
        
        # Get final statistics
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: Verifying setup")
        logger.info("=" * 80)
        collection_stats = chroma_setup.get_collection_stats()
        
        return {
            "status": "success",
            "chunks_loaded": len(chunks),
            "upsert_stats": upsert_stats,
            "collection_stats": collection_stats
        }
        
    except Exception as e:
        logger.error(f"Setup failed: {str(e)}", exc_info=True)
        return {"status": "error", "error": str(e)}


if __name__ == "__main__":
    """
    Main execution: Setup ChromaDB with document embeddings.
    """
    print("=" * 80)
    print("CHROMADB EMBEDDINGS SETUP")
    print("=" * 80)
    print("\nThis script will:")
    print("1. Load document chunks from the ingestion pipeline")
    print("2. Initialize ChromaDB with persistent storage")
    print("3. Compute embeddings using 'all-MiniLM-L6-v2'")
    print("4. Upsert vectors into ChromaDB collection")
    print("\n" + "=" * 80 + "\n")
    
    # Configuration
    DOCUMENTS_DIR = "data/documents/company_data"
    CHROMA_DB_PATH = "./chroma_db"
    COLLECTION_NAME = "capstone_docs"
    REPLACE_DUPLICATES = True  # Set to False to skip duplicates instead of replacing
    
    # Check if documents directory exists
    if not os.path.exists(DOCUMENTS_DIR):
        print(f"⚠️  Documents directory not found: {DOCUMENTS_DIR}")
        print("   Creating directory and sample document...")
        os.makedirs(DOCUMENTS_DIR, exist_ok=True)
        
        # Create sample document
        sample_file = Path(DOCUMENTS_DIR) / "sample_document.txt"
        sample_content = """
        Artificial Intelligence and Machine Learning Overview
        
        Artificial Intelligence (AI) is the simulation of human intelligence in machines. 
        Machine Learning (ML) is a subset of AI that enables systems to learn and improve 
        from experience without being explicitly programmed.
        
        Deep Learning is a subset of Machine Learning that uses neural networks with multiple 
        layers. These networks can learn complex patterns in large amounts of data.
        
        Natural Language Processing (NLP) is a field of AI that focuses on the interaction 
        between computers and human language. It enables machines to understand, interpret, 
        and generate human language.
        
        Computer Vision is another important AI field that enables machines to interpret and 
        understand visual information from the world, such as images and videos.
        
        Retrieval-Augmented Generation (RAG) combines information retrieval with language 
        generation. It retrieves relevant documents and uses them to generate accurate, 
        contextual responses.
        """
        
        with open(sample_file, 'w', encoding='utf-8') as f:
            f.write(sample_content)
        
        print(f"   ✓ Created sample document: {sample_file}\n")
    
    # Run setup
    print(f"Configuration:")
    print(f"  Documents Directory: {DOCUMENTS_DIR}")
    print(f"  ChromaDB Path: {CHROMA_DB_PATH}")
    print(f"  Collection Name: {COLLECTION_NAME}")
    print(f"  Replace Duplicates: {REPLACE_DUPLICATES}")
    print("\n" + "=" * 80 + "\n")
    
    results = setup_chromadb(
        documents_dir=DOCUMENTS_DIR,
        chroma_db_path=CHROMA_DB_PATH,
        collection_name=COLLECTION_NAME,
        replace_duplicates=REPLACE_DUPLICATES
    )
    
    # Print results
    print("\n" + "=" * 80)
    print("SETUP RESULTS")
    print("=" * 80)
    
    if results['status'] == 'success':
        print(f"✅ Status: SUCCESS")
        print(f"\nChunks Loaded: {results['chunks_loaded']}")
        
        upsert_stats = results['upsert_stats']
        print(f"\nUpsert Statistics:")
        print(f"  Total chunks processed: {upsert_stats['total']}")
        print(f"  New documents added: {upsert_stats['new']}")
        print(f"  Existing documents updated: {upsert_stats['updated']}")
        print(f"  Duplicates skipped: {upsert_stats['skipped']}")
        print(f"  Final collection count: {upsert_stats['final_count']}")
        
        collection_stats = results['collection_stats']
        print(f"\nCollection Statistics:")
        print(f"  Collection Name: {collection_stats['collection_name']}")
        print(f"  Total Documents: {collection_stats['total_documents']}")
        print(f"  Storage Path: {collection_stats['storage_path']}")
        print(f"  Embedding Model: {collection_stats['embedding_model']}")
        print(f"  Embedding Dimension: {collection_stats['embedding_dimension']}")
        
        if collection_stats.get('sample_document'):
            sample = collection_stats['sample_document']
            print(f"\nSample Document:")
            print(f"  ID: {sample['id']}")
            if sample['metadata']:
                print(f"  Metadata: {sample['metadata']}")
        
        print("\n✅ ChromaDB setup complete! Your vector database is ready for RAG queries.")
        
    elif results['status'] == 'no_documents':
        print(f"⚠️  Status: NO DOCUMENTS")
        print(f"   No documents found in {DOCUMENTS_DIR}")
        print(f"   Please add PDF, TXT, or DOCX files to the directory and run again.")
        
    else:
        print(f"❌ Status: ERROR")
        print(f"   Error: {results.get('error', 'Unknown error')}")
    
    print("=" * 80)
