"""
RAG Manager for LiveKit Agent
Fixed version with proper async functions
"""
import asyncio
import logging
from typing import List, Optional, Dict, Any
from functools import lru_cache
from pathlib import Path

from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    StorageContext, 
    load_index_from_storage,
    Settings
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import MetadataMode, NodeWithScore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from config import config
from utils.performance_monitor import time_function, performance_monitor

logger = logging.getLogger(__name__)

class FastRAGManager:
    """High-performance RAG manager optimized for voice agents"""
    
    def __init__(self):
        self.index: Optional[VectorStoreIndex] = None
        self.retriever: Optional[VectorIndexRetriever] = None
        self.query_engine = None
        self._embedding_cache: Dict[str, List[float]] = {}
        self._result_cache: Dict[str, str] = {}
        
        # Configure LlamaIndex for performance
        self._setup_llama_index()
        
    def _setup_llama_index(self):
        """Configure LlamaIndex settings for optimal performance"""
        
        # Use faster embedding model
        Settings.embed_model = OpenAIEmbedding(
            model=config.embedding_model,
            api_key=config.openai_api_key
        )
        
        # Configure LLM with timeout
        Settings.llm = OpenAI(
            model="gpt-4o-mini",
            api_key=config.openai_api_key,
            temperature=0.1,
            max_tokens=config.max_tokens,
            timeout=30.0
        )
        
        # Optimize chunk settings
        Settings.chunk_size = config.chunk_size
        Settings.chunk_overlap = config.chunk_overlap
        
        logger.info("✅ LlamaIndex configured for performance")
    
    @time_function("index_loading")
    async def load_or_create_index(self) -> bool:
        """Load existing index or create new one from data"""
        try:
            storage_dir = config.storage_dir
            
            if storage_dir.exists() and any(storage_dir.iterdir()):
                # Load existing index
                logger.info("📂 Loading existing vector index...")
                storage_context = StorageContext.from_defaults(persist_dir=str(storage_dir))
                self.index = load_index_from_storage(storage_context)
                logger.info("✅ Vector index loaded successfully")
            else:
                # Create new index from data
                logger.info("🔨 Creating new vector index from data...")
                success = await self._create_index_from_data()
                if not success:
                    return False
            
            # Setup retriever with optimized parameters
            self.retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=config.top_k_results,
            )
            
            # Setup query engine
            self.query_engine = self.index.as_query_engine(
                use_async=True,
                similarity_top_k=config.top_k_results,
                response_mode="compact"  # Faster response mode
            )
            
            logger.info("🚀 RAG system ready")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to setup RAG system: {e}")
            return False
    
    async def _create_index_from_data(self) -> bool:
        """Create vector index from data directory"""
        try:
            data_dir = config.data_dir
            
            if not data_dir.exists() or not any(data_dir.iterdir()):
                logger.error(f"❌ No data found in {data_dir}")
                logger.info("💡 Add your knowledge base files to the data/ directory")
                return False
            
            # Load documents
            documents = SimpleDirectoryReader(str(data_dir)).load_data()
            logger.info(f"📄 Loaded {len(documents)} documents")
            
            if not documents:
                logger.error("❌ No documents loaded")
                return False
            
            # Create index
            self.index = VectorStoreIndex.from_documents(documents)
            
            # Persist to storage
            self.index.storage_context.persist(persist_dir=str(config.storage_dir))
            logger.info("💾 Vector index created and saved")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating index: {e}")
            return False
    
    @time_function("rag_search")
    async def search(self, query: str) -> List[Dict[str, Any]]:
        """Fast vector search with caching"""
        try:
            if not self.retriever:
                logger.error("❌ RAG system not initialized")
                return []
            
            # Check cache first
            if config.enable_caching and query in self._result_cache:
                logger.debug("🎯 Cache hit for query")
                cached_result = self._result_cache[query]
                return [{"content": cached_result, "score": 1.0, "source": "cache"}]
            
            # Perform retrieval with timeout
            nodes = await asyncio.wait_for(
                self.retriever.aretrieve(query),
                timeout=config.rag_timeout_ms / 1000.0
            )
            
            # Process results
            results = []
            for node in nodes:
                if node.score >= config.similarity_threshold:
                    content = node.get_content(metadata_mode=MetadataMode.NONE)
                    results.append({
                        "content": content,
                        "score": node.score,
                        "source": node.metadata.get("file_name", "unknown")
                    })
            
            # Cache result if good
            if results and config.enable_caching:
                best_content = results[0]["content"]
                if len(self._result_cache) < 100:  # Limit cache size
                    self._result_cache[query] = best_content
            
            logger.debug(f"🔍 Found {len(results)} relevant results")
            return results
            
        except asyncio.TimeoutError:
            logger.warning(f"⏰ RAG search timeout for query: {query[:50]}")
            return []
        except Exception as e:
            logger.error(f"❌ RAG search error: {e}")
            return []
    
    @time_function("rag_query")
    async def query(self, question: str) -> str:
        """Query the knowledge base and get a formatted response"""
        try:
            if not self.query_engine:
                logger.error("❌ Query engine not initialized")
                return "I'm sorry, I don't have access to the knowledge base right now."
            
            # Check cache
            if config.enable_caching and question in self._result_cache:
                return self._result_cache[question]
            
            # Perform query with timeout
            response = await asyncio.wait_for(
                self.query_engine.aquery(question),
                timeout=config.rag_timeout_ms / 1000.0
            )
            
            result = str(response).strip()
            
            # Cache result
            if config.enable_caching and len(self._result_cache) < 100:
                self._result_cache[question] = result
            
            return result
            
        except asyncio.TimeoutError:
            logger.warning(f"⏰ RAG query timeout for: {question[:50]}")
            return "I'm sorry, the search is taking too long. Please try a simpler question."
        except Exception as e:
            logger.error(f"❌ RAG query error: {e}")
            return "I'm sorry, I encountered an error searching the knowledge base."
    
    async def get_context_for_query(self, query: str, max_length: int = 300) -> Optional[str]:
        """
        ✅ FIXED: Get context for parallel injection into chat
        This is the function called by on_user_turn_completed
        """
        try:
            if not self.retriever:
                logger.debug("❌ RAG retriever not available")
                return None
                
            results = await self.search(query)
            
            if not results:
                logger.debug(f"🔍 No results found for: {query[:50]}")
                return None
            
            # Use the best result and truncate if needed
            best_result = results[0]
            content = best_result["content"]
            
            if len(content) > max_length:
                content = content[:max_length] + "..."
            
            logger.debug(f"📚 Context found: {len(content)} chars")
            return content
            
        except Exception as e:
            logger.error(f"❌ Error getting context: {e}")
            return None
    
    def clear_cache(self):
        """Clear all caches"""
        self._result_cache.clear()
        self._embedding_cache.clear()
        logger.info("🧹 RAG caches cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics"""
        return {
            "index_loaded": self.index is not None,
            "retriever_ready": self.retriever is not None,
            "query_engine_ready": self.query_engine is not None,
            "cache_size": len(self._result_cache),
            "embedding_cache_size": len(self._embedding_cache)
        }

# Global RAG manager instance
rag_manager = FastRAGManager()

# Convenience functions
async def initialize_rag() -> bool:
    """Initialize the RAG system"""
    return await rag_manager.load_or_create_index()

async def search_knowledge_base(query: str) -> str:
    """Search knowledge base and return formatted answer"""
    return await rag_manager.query(query)

async def get_context(query: str, max_length: int = 300) -> Optional[str]:
    """Get context for chat injection - FIXED FUNCTION"""
    return await rag_manager.get_context_for_query(query, max_length)

if __name__ == "__main__":
    # Test the RAG manager
    import asyncio
    
    async def test_rag():
        print("🧪 Testing RAG Manager...")
        
        # Initialize
        success = await initialize_rag()
        if not success:
            print("❌ Failed to initialize RAG")
            return
        
        # Test search
        result = await search_knowledge_base("What is this about?")
        print(f"📝 Search result: {result}")
        
        # Test context
        context = await get_context("test query")
        print(f"📄 Context: {context}")
        
        # Show stats
        stats = rag_manager.get_stats()
        print(f"📊 Stats: {stats}")
    
    asyncio.run(test_rag())