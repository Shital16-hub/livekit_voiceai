"""FIXED: General-purpose RAG Manager - works with ANY knowledge base"""
import asyncio
import time
import logging
from typing import Optional
from datetime import datetime, timedelta

from llama_index.core import VectorStoreIndex, Settings, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from qdrant_client import QdrantClient, AsyncQdrantClient

from config import RAG_CONFIG, QDRANT_CONFIG

logger = logging.getLogger(__name__)

class GenericRAGManager:
    """FIXED: General-purpose RAG manager for ANY knowledge domain"""
    
    def __init__(self):
        self.query_cache = {}
        self.setup_complete = False
        self.search_times = []
        self.total_queries = 0
        
        logger.info(f"🔧 RAG Manager created with timeout: {RAG_CONFIG['query_timeout']} seconds")
        
    async def initialize(self):
        """FIXED: Initialize with proper timeout handling"""
        try:
            start_time = time.time()
            
            # FIXED: Remove max_retries parameter that was causing errors
            Settings.llm = OpenAI(
                model=RAG_CONFIG["llm_model"],
                temperature=RAG_CONFIG["temperature"],
                max_tokens=RAG_CONFIG["max_tokens"],
                timeout=30.0,  # Simple float timeout
            )
            
            Settings.embed_model = OpenAIEmbedding(
                model=RAG_CONFIG["embedding_model"],
                timeout=30.0,
            )
            
            # Setup Qdrant clients
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
            collections = self.qdrant_client.get_collections()
            logger.info(f"✅ Connected to Qdrant: {len(collections.collections)} collections")
            
            # Create vector store with both clients
            self.vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                aclient=self.async_qdrant_client,
                collection_name=QDRANT_CONFIG["collection_name"]
            )
            
            # Create index and query engine
            self.index = VectorStoreIndex.from_vector_store(self.vector_store)
            self.query_engine = self.index.as_query_engine(
                similarity_top_k=RAG_CONFIG["similarity_top_k"],
                response_mode="compact",
                use_async=True
            )
            
            self.setup_complete = True
            init_time = (time.time() - start_time) * 1000
            logger.info(f"✅ RAG Manager initialized in {init_time:.1f}ms")
            
        except Exception as e:
            logger.error(f"❌ RAG initialization failed: {e}")
            raise
    
    async def search_knowledge(self, query: str) -> Optional[str]:
        """FIXED: General-purpose search with proper timeout handling"""
        if not self.setup_complete:
            logger.warning("RAG Manager not initialized")
            return None
            
        start_time = time.time()
        self.total_queries += 1
        
        try:
            # Check cache first
            if cached_result := self._check_cache(query):
                elapsed = (time.time() - start_time) * 1000
                logger.info(f"🚀 Cache hit: {elapsed:.1f}ms")
                return cached_result
            
            # FIXED: Use the configured timeout (15 seconds)
            logger.info(f"🔍 Starting RAG search with {RAG_CONFIG['query_timeout']}s timeout...")
            
            response = await asyncio.wait_for(
                self.query_engine.aquery(query),
                timeout=RAG_CONFIG['query_timeout']  # Use config value
            )
            
            if response and response.source_nodes:
                context_parts = []
                for node in response.source_nodes:
                    if hasattr(node, 'score') and node.score > RAG_CONFIG["relevance_threshold"]:
                        content = node.text
                        if len(content) > 400:
                            content = content[:400] + "..."
                        
                        # Add source info if available
                        metadata = getattr(node, 'metadata', {})
                        source_info = ""
                        if metadata.get('filename'):
                            source_info = f" [Source: {metadata['filename']}]"
                        elif metadata.get('source'):
                            source_info = f" [Source: {metadata['source']}]"
                        
                        context_parts.append(content + source_info)
                
                if context_parts:
                    result = "\n\n".join(context_parts[:RAG_CONFIG["similarity_top_k"]])
                    self._cache_result(query, result)
                    
                    elapsed = (time.time() - start_time) * 1000
                    self.search_times.append(elapsed)
                    logger.info(f"✅ RAG search completed: {elapsed:.1f}ms, found {len(context_parts)} results")
                    return result
                    
            elapsed = (time.time() - start_time) * 1000
            logger.info(f"📝 No relevant results found: {elapsed:.1f}ms")
            return None
                
        except asyncio.TimeoutError:
            elapsed = (time.time() - start_time) * 1000
            logger.warning(f"⏰ RAG search timeout after {elapsed:.1f}ms")
            return None
        except Exception as e:
            elapsed = (time.time() - start_time) * 1000
            logger.error(f"❌ RAG search error after {elapsed:.1f}ms: {e}")
            return None
    
    def _check_cache(self, query: str) -> Optional[str]:
        """Simple cache check"""
        query_key = query.lower().strip()
        if query_key in self.query_cache:
            cache_entry = self.query_cache[query_key]
            if datetime.now() - cache_entry["timestamp"] < timedelta(seconds=3600):  # 1 hour TTL
                return cache_entry["result"]
        return None
    
    def _cache_result(self, query: str, result: str):
        """Cache successful results"""
        query_key = query.lower().strip()
        self.query_cache[query_key] = {
            "result": result,
            "timestamp": datetime.now()
        }
        
        # Cleanup if cache gets too big
        if len(self.query_cache) > 100:
            oldest_key = min(self.query_cache.keys(), 
                           key=lambda k: self.query_cache[k]["timestamp"])
            del self.query_cache[oldest_key]
    
    def get_performance_stats(self):
        """Get performance statistics"""
        if not self.search_times:
            return {"status": "No queries yet"}
            
        avg_latency = sum(self.search_times) / len(self.search_times)
        return {
            "total_queries": self.total_queries,
            "avg_latency_ms": f"{avg_latency:.1f}",
            "cache_size": len(self.query_cache)
        }
    
    async def get_collection_info(self):
        """Get collection information"""
        try:
            collection_info = self.qdrant_client.get_collection(QDRANT_CONFIG["collection_name"])
            return {
                "collection_name": QDRANT_CONFIG["collection_name"],
                "vectors_count": collection_info.vectors_count,
            }
        except Exception as e:
            return {"error": str(e)}