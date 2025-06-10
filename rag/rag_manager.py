"""Generic RAG Manager for any knowledge domain - FIXED VERSION"""
import asyncio
import time
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from llama_index.core import VectorStoreIndex, Settings, StorageContext
from llama_index.core.schema import Document
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams

from config import RAG_CONFIG, QDRANT_CONFIG, CACHE_CONFIG

logger = logging.getLogger(__name__)

class GenericRAGManager:
    """Generic RAG manager that works with any knowledge domain - FIXED"""
    
    def __init__(self):
        self.query_cache = {}
        self.embedding_cache = {}
        self.last_cleanup = datetime.now()
        self.setup_complete = False
        
        # Performance tracking
        self.search_times = []
        self.cache_hits = 0
        self.total_queries = 0
        
        # Print debug info
        logger.info(f"🔧 RAG Manager created with timeout: {RAG_CONFIG.get('query_timeout', 'NOT SET')} seconds")
        
    async def initialize(self):
        """Initialize generic RAG system with FIXED timeout configuration"""
        try:
            start_time = time.time()
            
            # FIXED: LlamaIndex OpenAI expects simple float timeout, NOT httpx.Timeout
            Settings.llm = OpenAI(
                model=RAG_CONFIG["llm_model"],
                temperature=RAG_CONFIG["temperature"],
                max_tokens=RAG_CONFIG["max_tokens"],
                timeout=30.0,  # Simple float - FIXED
            )
            
            Settings.embed_model = OpenAIEmbedding(
                model=RAG_CONFIG["embedding_model"],
                embed_batch_size=RAG_CONFIG["embedding_batch_size"],
                timeout=30.0,  # Simple float - FIXED
            )
            
            Settings.chunk_size = RAG_CONFIG["chunk_size"]
            Settings.chunk_overlap = RAG_CONFIG["chunk_overlap"]
            
            # Setup Qdrant clients with increased timeouts
            logger.info("Connecting to Qdrant...")
            self.qdrant_client = QdrantClient(
                url=QDRANT_CONFIG["url"],
                api_key=QDRANT_CONFIG["api_key"],
                timeout=QDRANT_CONFIG["timeout"],
                prefer_grpc=QDRANT_CONFIG["prefer_grpc"]
            )
            
            self.async_qdrant_client = AsyncQdrantClient(
                url=QDRANT_CONFIG["url"],
                api_key=QDRANT_CONFIG["api_key"],
                timeout=QDRANT_CONFIG["timeout"],
                prefer_grpc=QDRANT_CONFIG["prefer_grpc"]
            )
            
            # Test connection
            logger.info("Testing Qdrant connection...")
            collections = self.qdrant_client.get_collections()
            logger.info(f"✅ Connected to Qdrant: {len(collections.collections)} collections")
            
            # Create vector store
            logger.info("Creating vector store...")
            self.vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                aclient=self.async_qdrant_client,
                collection_name=QDRANT_CONFIG["collection_name"],
                prefer_grpc=QDRANT_CONFIG["prefer_grpc"]
            )
            
            # Create index
            logger.info("Creating vector store index...")
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store
            )
            
            # Create optimized query engine
            logger.info("Creating query engine...")
            self.query_engine = self.index.as_query_engine(
                similarity_top_k=RAG_CONFIG["similarity_top_k"],
                response_mode="compact",
                streaming=False,
                use_async=True
            )
            
            self.setup_complete = True
            
            init_time = (time.time() - start_time) * 1000
            logger.info(f"✅ Generic RAG Manager initialized in {init_time:.1f}ms")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize RAG Manager: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    async def search_knowledge(self, query: str) -> Optional[str]:
        """Generic knowledge search with FIXED timeout handling"""
        if not self.setup_complete:
            logger.warning("RAG Manager not initialized")
            return None
            
        start_time = time.time()
        self.total_queries += 1
        
        # HARDCODED TIMEOUT - 5 seconds
        SEARCH_TIMEOUT = 5.0
        
        try:
            # Check cache first
            cached_result = self._check_cache(query)
            if cached_result:
                self.cache_hits += 1
                elapsed = (time.time() - start_time) * 1000
                logger.info(f"🚀 Cache hit: {elapsed:.1f}ms")
                return cached_result
            
            # Perform search with FIXED timeout
            logger.info(f"🔍 Starting RAG search with {SEARCH_TIMEOUT}s timeout...")
            try:
                response = await asyncio.wait_for(
                    self.query_engine.aquery(query),
                    timeout=SEARCH_TIMEOUT  # HARDCODED 5 seconds
                )
                logger.info("✅ RAG search completed successfully")
                
                # Process response
                if response and response.source_nodes:
                    context_parts = []
                    for node in response.source_nodes:
                        if hasattr(node, 'score') and node.score > RAG_CONFIG["relevance_threshold"]:
                            # Extract content with source information
                            content = node.text
                            if len(content) > 400:
                                content = content[:400] + "..."
                            
                            # Add source metadata if available
                            metadata = getattr(node, 'metadata', {})
                            source_info = ""
                            if metadata.get('filename'):
                                source_info = f" [Source: {metadata['filename']}]"
                            elif metadata.get('source'):
                                source_info = f" [Source: {metadata['source']}]"
                            
                            context_parts.append(content + source_info)
                    
                    if context_parts:
                        result = "\n\n".join(context_parts[:RAG_CONFIG["similarity_top_k"]])
                        
                        # Cache the result
                        self._cache_result(query, result)
                        
                        elapsed = (time.time() - start_time) * 1000
                        self.search_times.append(elapsed)
                        logger.info(f"✅ RAG search completed: {elapsed:.1f}ms, found {len(context_parts)} results")
                        
                        return result
                    else:
                        elapsed = (time.time() - start_time) * 1000
                        logger.info(f"📝 RAG search found nodes but below relevance threshold: {elapsed:.1f}ms")
                        return None
                else:
                    elapsed = (time.time() - start_time) * 1000
                    logger.info(f"📝 RAG search found no relevant nodes: {elapsed:.1f}ms")
                    return None
                
            except asyncio.TimeoutError:
                elapsed = (time.time() - start_time) * 1000
                logger.warning(f"⏰ RAG search timeout after {elapsed:.1f}ms")
                return None
            except Exception as search_error:
                elapsed = (time.time() - start_time) * 1000
                logger.error(f"❌ RAG search error after {elapsed:.1f}ms: {search_error}")
                return None
                
        except Exception as e:
            elapsed = (time.time() - start_time) * 1000
            logger.error(f"❌ RAG search error after {elapsed:.1f}ms: {e}")
            
        return None
    
    def _check_cache(self, query: str) -> Optional[str]:
        """Check cache for similar queries"""
        if not CACHE_CONFIG["enable_semantic_cache"]:
            return None
            
        query_key = query.lower().strip()
        if query_key in self.query_cache:
            cache_entry = self.query_cache[query_key]
            
            if datetime.now() - cache_entry["timestamp"] < timedelta(seconds=CACHE_CONFIG["cache_ttl"]):
                return cache_entry["result"]
            else:
                del self.query_cache[query_key]
        
        return None
    
    def _cache_result(self, query: str, result: str):
        """Cache query result"""
        if not CACHE_CONFIG["enable_semantic_cache"]:
            return
            
        query_key = query.lower().strip()
        self.query_cache[query_key] = {
            "result": result,
            "timestamp": datetime.now()
        }
        
        if len(self.query_cache) > CACHE_CONFIG["max_cache_size"]:
            self._cleanup_cache()
    
    def _cleanup_cache(self):
        """Remove old cache entries"""
        if datetime.now() - self.last_cleanup < timedelta(minutes=5):
            return
            
        cutoff_time = datetime.now() - timedelta(seconds=CACHE_CONFIG["cache_ttl"])
        keys_to_remove = [
            key for key, entry in self.query_cache.items()
            if entry["timestamp"] < cutoff_time
        ]
        
        for key in keys_to_remove:
            del self.query_cache[key]
            
        self.last_cleanup = datetime.now()
        logger.info(f"🧹 Cache cleanup: removed {len(keys_to_remove)} entries")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.search_times:
            return {"status": "No queries yet"}
            
        avg_latency = sum(self.search_times) / len(self.search_times)
        cache_hit_rate = (self.cache_hits / self.total_queries) * 100 if self.total_queries > 0 else 0
        
        return {
            "total_queries": self.total_queries,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": f"{cache_hit_rate:.1f}%",
            "avg_latency_ms": f"{avg_latency:.1f}",
            "min_latency_ms": f"{min(self.search_times):.1f}",
            "max_latency_ms": f"{max(self.search_times):.1f}",
        }
    
    async def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the current collection"""
        try:
            collection_info = self.qdrant_client.get_collection(QDRANT_CONFIG["collection_name"])
            return {
                "collection_name": QDRANT_CONFIG["collection_name"],
                "vectors_count": collection_info.vectors_count,
                "status": collection_info.status.value if hasattr(collection_info.status, 'value') else str(collection_info.status),
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {"error": str(e)}