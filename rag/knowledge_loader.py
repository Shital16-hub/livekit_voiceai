"""Generic knowledge loader for any type of documents"""
import asyncio
import logging
from pathlib import Path
from typing import List, Optional

from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.core.schema import Document
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from qdrant_client import QdrantClient
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
        """Setup Qdrant collection"""
        client = QdrantClient(
            url=QDRANT_CONFIG["url"],
            api_key=QDRANT_CONFIG["api_key"]
        )
        
        collection_name = QDRANT_CONFIG["collection_name"]
        
        try:
            # Check if collection exists
            collections = client.get_collections()
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
        
        # Setup Qdrant
        client = await self.setup_qdrant_collection()
        
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
                
                # Create sample documents for testing
                sample_docs = self._create_sample_documents()
                documents = sample_docs
                logger.info("✅ Created sample documents for testing")
            
            logger.info(f"📚 Loading {len(documents)} documents...")
            
            # Create vector store
            vector_store = QdrantVectorStore(
                client=client,
                collection_name=QDRANT_CONFIG["collection_name"]
            )
            
            # Upload documents
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                show_progress=True
            )
            
            logger.info("✅ Documents loaded successfully to Qdrant Cloud!")
            logger.info(f"📊 Collection: {QDRANT_CONFIG['collection_name']}")
            logger.info(f"📄 Documents: {len(documents)}")
            
            # Test search
            query_engine = index.as_query_engine(similarity_top_k=2)
            test_response = await query_engine.aquery("What information is available?")
            
            logger.info(f"🧪 Test query result: {test_response}")
            
        except Exception as e:
            logger.error(f"❌ Error loading documents: {e}")
    
    async def load_from_texts(self, texts: List[str], metadata_list: Optional[List[dict]] = None):
        """Load documents from a list of texts"""
        Settings.embed_model = OpenAIEmbedding(
            model=RAG_CONFIG["embedding_model"],
            embed_batch_size=RAG_CONFIG["embedding_batch_size"]
        )
        
        client = await self.setup_qdrant_collection()
        
        # Create documents from texts
        documents = self.processor.create_documents_from_text(texts, metadata_list)
        
        if documents:
            vector_store = QdrantVectorStore(
                client=client,
                collection_name=QDRANT_CONFIG["collection_name"]
            )
            
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                show_progress=True
            )
            
            logger.info(f"✅ Loaded {len(documents)} text documents successfully!")
    
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