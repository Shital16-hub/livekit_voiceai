"""Generic knowledge loader for any type of documents"""
import asyncio
import logging
import os
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.core.schema import Document
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams

from config import RAG_CONFIG, QDRANT_CONFIG
from document_processor import DocumentProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GenericKnowledgeLoader:
    """Load any type of documents into the knowledge base"""
    
    def __init__(self):
        self.processor = DocumentProcessor()
        
    async def setup_qdrant_collection(self):
        """Setup Qdrant collection with proper error handling"""
        try:
            logger.info("Setting up Qdrant connection...")
            logger.info(f"URL: {QDRANT_CONFIG['url']}")
            logger.info(f"Collection: {QDRANT_CONFIG['collection_name']}")
            
            # Create sync client for collection management
            client = QdrantClient(
                url=QDRANT_CONFIG["url"],
                api_key=QDRANT_CONFIG["api_key"],
                timeout=30.0,
                prefer_grpc=False  # Important for cloud
            )
            
            collection_name = QDRANT_CONFIG["collection_name"]
            
            # Test connection first
            logger.info("Testing Qdrant connection...")
            try:
                collections = client.get_collections()
                logger.info("✅ Successfully connected to Qdrant Cloud")
                logger.info(f"Found {len(collections.collections)} existing collections")
            except Exception as conn_error:
                logger.error(f"❌ Connection test failed: {conn_error}")
                raise
            
            # Check if collection exists
            collection_exists = any(col.name == collection_name for col in collections.collections)
            
            if not collection_exists:
                logger.info(f"Creating collection: {collection_name}")
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=1536,  # text-embedding-3-small dimension
                        distance=Distance.COSINE
                    )
                )
                logger.info("✅ Collection created successfully")
            else:
                logger.info(f"✅ Collection '{collection_name}' already exists")
                
            return client
            
        except Exception as e:
            logger.error(f"❌ Error with collection: {e}")
            raise
    
    async def load_from_directory(self, directory_path: str = "data"):
        """Load all documents from a directory"""
        # Configure LlamaIndex
        Settings.embed_model = OpenAIEmbedding(
            model=RAG_CONFIG["embedding_model"],
            embed_batch_size=RAG_CONFIG["embedding_batch_size"]
        )
        Settings.chunk_size = RAG_CONFIG["chunk_size"]
        Settings.chunk_overlap = RAG_CONFIG["chunk_overlap"]
        
        # Setup Qdrant sync client
        client = await self.setup_qdrant_collection()
        
        # Create async client with same settings
        logger.info("Creating async client...")
        aclient = AsyncQdrantClient(
            url=QDRANT_CONFIG["url"],
            api_key=QDRANT_CONFIG["api_key"],
            timeout=30.0,
            prefer_grpc=False
        )
        logger.info("✅ Async client created successfully")
        
        # Load documents
        data_path = Path(directory_path)
        if not data_path.exists():
            logger.error(f"❌ '{directory_path}' directory not found. Please create it and add your documents.")
            return
        
        try:
            # Process all documents in directory
            documents = self.processor.load_from_directory(directory_path)
            
            if not documents:
                logger.warning(f"⚠️ No supported documents found in '{directory_path}' directory")
                sample_docs = self._create_sample_documents()
                documents = sample_docs
                logger.info("✅ Created sample documents for testing")
            
            logger.info(f"📚 Loading {len(documents)} documents...")
            
            # Create vector store with BOTH clients (this is the key fix!)
            logger.info("Creating vector store with both sync and async clients...")
            vector_store = QdrantVectorStore(
                client=client,
                aclient=aclient,  # Both clients required for async operations!
                collection_name=QDRANT_CONFIG["collection_name"]
            )
            logger.info("✅ Vector store created successfully")
            
            # Upload documents with async support
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # THIS IS THE KEY: use_async=True for async operations
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                use_async=True,  # This enables async support!
                show_progress=True
            )
            
            logger.info("✅ Documents loaded successfully to Qdrant Cloud!")
            logger.info(f"📊 Collection: {QDRANT_CONFIG['collection_name']}")
            logger.info(f"📄 Documents: {len(documents)}")
            
            # Test search with async query
            logger.info("🧪 Testing async search functionality...")
            try:
                query_engine = index.as_query_engine(similarity_top_k=2)
                test_response = await query_engine.aquery("What is this document about?")
                
                logger.info(f"✅ Async search test successful!")
                logger.info(f"📝 Test result: {str(test_response)[:200]}...")
                
            except Exception as search_error:
                logger.warning(f"⚠️ Async search test failed: {search_error}")
                logger.info("📝 But documents are loaded successfully!")
                
            logger.info("🎉 Your Rich Dad Poor Dad knowledge base is ready!")
            logger.info("📚 All document chunks are now searchable in Qdrant Cloud")
            
        except Exception as e:
            logger.error(f"❌ Error loading documents: {e}")
        finally:
            # Close async client
            await aclient.close()
    
    async def load_from_texts(self, texts: List[str], metadata_list: Optional[List[dict]] = None):
        """Load documents from a list of texts"""
        Settings.embed_model = OpenAIEmbedding(
            model=RAG_CONFIG["embedding_model"],
            embed_batch_size=RAG_CONFIG["embedding_batch_size"]
        )
        
        client = await self.setup_qdrant_collection()
        
        # Create async client separately
        aclient = AsyncQdrantClient(
            url=QDRANT_CONFIG["url"],
            api_key=QDRANT_CONFIG["api_key"],
            timeout=30.0,
            prefer_grpc=False
        )
        
        try:
            # Create documents from texts
            documents = self.processor.create_documents_from_text(texts, metadata_list)
            
            if documents:
                vector_store = QdrantVectorStore(
                    client=client,
                    aclient=aclient,
                    collection_name=QDRANT_CONFIG["collection_name"]
                )
                
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                index = VectorStoreIndex.from_documents(
                    documents,
                    storage_context=storage_context,
                    use_async=True,  # Enable async here too
                    show_progress=True
                )
                
                logger.info(f"✅ Loaded {len(documents)} text documents successfully!")
        finally:
            await aclient.close()
    
    def _create_sample_documents(self) -> List[Document]:
        """Create sample documents for testing"""
        return [
            Document(
                text="This is a sample document in the knowledge base. It contains general information that can be searched and retrieved by the RAG system.",
                metadata={"source": "sample", "category": "general", "type": "example"}
            ),
            Document(
                text="Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines that can perform tasks that typically require human intelligence.",
                metadata={"source": "sample", "category": "technology", "topic": "AI"}
            ),
            Document(
                text="Machine Learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed for every task.",
                metadata={"source": "sample", "category": "technology", "topic": "ML"}
            ),
            Document(
                text="Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and humans through natural language.",
                metadata={"source": "sample", "category": "technology", "topic": "NLP"}
            ),
        ]

async def main():
    """Main function to load knowledge base"""
    # Ensure environment variables are loaded
    load_dotenv()
    
    # Debug: Print environment variables (remove API key for security)
    logger.info("🔍 Environment check:")
    logger.info(f"QDRANT_CLOUD_URL: {os.getenv('QDRANT_CLOUD_URL')}")
    logger.info(f"OPENAI_API_KEY: {'***' if os.getenv('OPENAI_API_KEY') else 'NOT SET'}")
    logger.info(f"COLLECTION_NAME: {os.getenv('COLLECTION_NAME')}")
    
    loader = GenericKnowledgeLoader()
    
    print("🚀 Generic Knowledge Base Loader")
    print("Choose an option:")
    print("1. Load from 'data' directory")
    print("2. Load sample documents")
    print("3. Load custom texts")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        directory = input("Enter directory path (default: 'data'): ").strip() or "data"
        await loader.load_from_directory(directory)
    elif choice == "2":
        # Load just sample documents
        await loader.load_from_texts([
            "This is sample document 1 with information about the knowledge base system.",
            "This is sample document 2 containing different information for testing.",
            "Document 3 has additional content that can be searched and retrieved."
        ])
    elif choice == "3":
        print("Enter your custom texts (one per line, empty line to finish):")
        texts = []
        while True:
            text = input()
            if not text.strip():
                break
            texts.append(text)
        
        if texts:
            await loader.load_from_texts(texts)
        else:
            print("No texts provided.")
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    asyncio.run(main())