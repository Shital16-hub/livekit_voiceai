"""
Fixed RAG Manager for LiveKit Agent
Simplified and optimized for voice interactions
"""
import asyncio
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    StorageContext, 
    load_index_from_storage,
    Settings,
    Document
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import MetadataMode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from config import config

logger = logging.getLogger(__name__)

class SimplifiedRAGManager:
    """
    Simplified RAG manager optimized for LiveKit voice agents
    """
    
    def __init__(self):
        self.index: Optional[VectorStoreIndex] = None
        self.retriever: Optional[VectorIndexRetriever] = None
        self.query_engine = None
        self._result_cache: Dict[str, str] = {}
        
        # Configure LlamaIndex
        self._setup_llama_index()
        
    def _setup_llama_index(self):
        """Configure LlamaIndex settings"""
        try:
            # Use OpenAI embeddings
            Settings.embed_model = OpenAIEmbedding(
                model=config.embedding_model,
                api_key=config.openai_api_key
            )
            
            # Configure LLM
            Settings.llm = OpenAI(
                model="gpt-4o-mini",
                api_key=config.openai_api_key,
                temperature=0.1,
                max_tokens=80,  # Even shorter for voice
                timeout=15.0
            )
            
            # Optimize chunk settings
            Settings.chunk_size = config.chunk_size
            Settings.chunk_overlap = config.chunk_overlap
            
            logger.info("✅ LlamaIndex configured successfully")
            
        except Exception as e:
            logger.error(f"❌ LlamaIndex configuration error: {e}")
            raise
    
    async def load_or_create_index(self) -> bool:
        """Load existing index or create new one"""
        try:
            storage_dir = config.storage_dir
            
            # Try to load existing index
            if storage_dir.exists() and any(storage_dir.iterdir()):
                logger.info("📂 Loading existing vector index...")
                storage_context = StorageContext.from_defaults(persist_dir=str(storage_dir))
                self.index = load_index_from_storage(storage_context)
                logger.info("✅ Existing index loaded successfully")
            else:
                # Create new index
                logger.info("🔨 Creating new vector index...")
                success = await self._create_index_from_data()
                if not success:
                    return False
            
            # Setup retriever and query engine
            self.retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=config.top_k_results,
            )
            
            self.query_engine = self.index.as_query_engine(
                use_async=True,
                similarity_top_k=config.top_k_results,
                response_mode="compact"
            )
            
            logger.info("🚀 RAG system ready")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize RAG system: {e}")
            return False
    
    async def _create_index_from_data(self) -> bool:
        """Create vector index from data directory"""
        try:
            data_dir = config.data_dir
            
            if not data_dir.exists():
                logger.error(f"❌ Data directory not found: {data_dir}")
                return False
            
            # Check for files
            files = list(data_dir.glob("*.*"))
            if not files:
                logger.error("❌ No data files found")
                logger.info("💡 Run: python data_ingestion.py --create-sample")
                return False
            
            # Load documents
            logger.info(f"📄 Loading documents from {data_dir}...")
            reader = SimpleDirectoryReader(
                input_dir=str(data_dir),
                recursive=True,
                required_exts=[".txt", ".md", ".pdf"],
            )
            documents = reader.load_data()
            
            if not documents:
                logger.error("❌ No documents loaded")
                return False
            
            logger.info(f"📚 Loaded {len(documents)} documents")
            
            # Create index
            logger.info("⚡ Creating vector index...")
            self.index = VectorStoreIndex.from_documents(documents)
            
            # Save index
            storage_dir = config.storage_dir
            storage_dir.mkdir(exist_ok=True)
            self.index.storage_context.persist(persist_dir=str(storage_dir))
            
            logger.info("💾 Vector index created and saved")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating index: {e}")
            return False
    
    async def search(self, query: str) -> List[Dict[str, Any]]:
        """
        ✅ FIXED: Simple search function with lowered threshold
        """
        try:
            if not self.retriever:
                logger.warning("❌ RAG retriever not available")
                return []
            
            # Check cache first
            if config.enable_caching and query in self._result_cache:
                cached_result = self._result_cache[query]
                return [{"content": cached_result, "score": 1.0, "source": "cache"}]
            
            # Perform retrieval with reasonable timeout
            logger.debug(f"🔍 Searching for: {query[:50]}...")
            
            try:
                nodes = await asyncio.wait_for(
                    self.retriever.aretrieve(query),
                    timeout=3.0  # 3 second timeout
                )
            except asyncio.TimeoutError:
                logger.warning(f"⏰ RAG search timeout for: {query[:50]}")
                return []
            
            # ✅ FIXED: Process results with much lower threshold
            results = []
            for node in nodes:
                if hasattr(node, 'score'):
                    # ✅ FIXED: Use lower threshold (0.2) or no threshold at all
                    if node.score >= 0.2:  # Much lower threshold
                        content = node.get_content(metadata_mode=MetadataMode.NONE)
                        results.append({
                            "content": content,
                            "score": node.score,
                            "source": node.metadata.get("file_name", "unknown")
                        })
                        logger.debug(f"✅ Including result with score: {node.score:.3f}")
            
            # If no results with threshold, take the best result anyway
            if not results and nodes:
                best_node = max(nodes, key=lambda n: getattr(n, 'score', 0))
                content = best_node.get_content(metadata_mode=MetadataMode.NONE)
                results.append({
                    "content": content,
                    "score": getattr(best_node, 'score', 0),
                    "source": best_node.metadata.get("file_name", "unknown")
                })
                logger.info(f"✅ Taking best result with score: {getattr(best_node, 'score', 0):.3f}")
            
            # Cache the best result
            if results and config.enable_caching:
                best_content = results[0]["content"]
                if len(self._result_cache) < 50:  # Limit cache size
                    self._result_cache[query] = best_content
            
            logger.info(f"✅ Found {len(results)} relevant results")
            return results
            
        except Exception as e:
            logger.error(f"❌ RAG search error: {e}")
            return []
    
    async def query(self, question: str) -> str:
        """
        ✅ FIXED: Query the knowledge base and return formatted response
        """
        try:
            if not self.query_engine:
                logger.warning("❌ Query engine not available")
                return "I don't have access to the knowledge base right now."
            
            # Check cache
            if config.enable_caching and question in self._result_cache:
                return self._result_cache[question]
            
            # Perform query with timeout
            logger.debug(f"💭 Querying: {question[:50]}...")
            
            try:
                response = await asyncio.wait_for(
                    self.query_engine.aquery(question),
                    timeout=5.0  # 5 second timeout
                )
            except asyncio.TimeoutError:
                logger.warning(f"⏰ RAG query timeout for: {question[:50]}")
                return "The search is taking too long. Please try a simpler question."
            
            result = str(response).strip()
            
            # Limit response length for voice
            if len(result) > 200:  # Even shorter for voice
                result = result[:200] + "..."
            
            # Cache result
            if config.enable_caching and len(self._result_cache) < 50:
                self._result_cache[question] = result
            
            logger.debug("✅ Query completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"❌ RAG query error: {e}")
            return "I encountered an error searching the knowledge base."
    
    def clear_cache(self):
        """Clear result cache"""
        self._result_cache.clear()
        logger.info("🧹 RAG cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            "index_loaded": self.index is not None,
            "retriever_ready": self.retriever is not None,
            "query_engine_ready": self.query_engine is not None,
            "cache_size": len(self._result_cache),
        }

# Global RAG manager instance
rag_manager = SimplifiedRAGManager()

# Convenience functions for backward compatibility
async def initialize_rag() -> bool:
    """Initialize the RAG system"""
    return await rag_manager.load_or_create_index()

async def search_knowledge_base(query: str) -> str:
    """Search knowledge base and return formatted answer"""
    return await rag_manager.query(query)

async def get_context(query: str, max_length: int = 150) -> Optional[str]:
    """Get context for chat injection"""
    try:
        results = await rag_manager.search(query)
        if results:
            content = results[0]["content"]
            if len(content) > max_length:
                content = content[:max_length] + "..."
            return content
        return None
    except Exception as e:
        logger.error(f"❌ Error getting context: {e}")
        return None