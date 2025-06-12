"""
Advanced Semantic Cache with Redis and FAISS
"""
import asyncio
import json
import numpy as np
import redis.asyncio as redis
from typing import Optional, List, Dict, Any
from sentence_transformers import SentenceTransformer
import faiss
import logging
from config import config

logger = logging.getLogger(__name__)

class AdvancedSemanticCache:
    """Multi-level semantic cache with Redis and FAISS"""
    
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384  # MiniLM embedding dimension
        self.faiss_index = faiss.IndexFlatIP(self.dimension)  # Cosine similarity
        self.query_cache = {}  # L1 memory cache
        self.response_cache = {}
        self.query_embeddings = []
        self.redis_client = None
        self.cache_keys = []
        
    async def initialize(self):
        """Initialize Redis connection and load existing cache"""
        try:
            self.redis_client = redis.Redis(
                host=config.redis_host,
                port=config.redis_port,
                db=config.redis_db,
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info("✅ Redis semantic cache initialized")
            
            # Load existing cache from Redis
            await self._load_cache_from_redis()
            
        except Exception as e:
            logger.warning(f"⚠️ Redis not available, using memory-only cache: {e}")
            self.redis_client = None
    
    async def _load_cache_from_redis(self):
        """Load existing cache from Redis"""
        if not self.redis_client:
            return
            
        try:
            # Load cache keys
            keys = await self.redis_client.keys("semantic_cache:*")
            
            for key in keys[:100]:  # Limit to 100 entries for startup
                cache_data = await self.redis_client.get(key)
                if cache_data:
                    data = json.loads(cache_data)
                    query = data['query']
                    response = data['response']
                    embedding = np.array(data['embedding'], dtype=np.float32)
                    
                    # Add to FAISS index
                    self.faiss_index.add(embedding.reshape(1, -1))
                    self.query_embeddings.append(embedding)
                    self.cache_keys.append(key)
                    
                    # Add to memory cache
                    cache_id = len(self.query_embeddings) - 1
                    self.query_cache[cache_id] = query
                    self.response_cache[cache_id] = response
            
            logger.info(f"📂 Loaded {len(self.cache_keys)} cached entries from Redis")
            
        except Exception as e:
            logger.error(f"❌ Error loading cache from Redis: {e}")
    
    def _encode_query(self, query: str) -> np.ndarray:
        """Encode query to embedding"""
        return self.model.encode([query])[0].astype(np.float32)
    
    async def get(self, query: str, threshold: float = 0.8) -> Optional[str]:
        """Get cached response for semantically similar query"""
        try:
            # L1: Memory cache exact match
            query_hash = hash(query)
            if query_hash in self.query_cache:
                logger.debug("🎯 L1 cache hit")
                return self.response_cache.get(query_hash)
            
            # L2: Semantic search with FAISS
            if self.faiss_index.ntotal == 0:
                return None
            
            query_embedding = self._encode_query(query)
            
            # Search for similar queries
            similarities, indices = self.faiss_index.search(
                query_embedding.reshape(1, -1), 
                min(5, self.faiss_index.ntotal)
            )
            
            # Check if any result meets threshold
            for sim, idx in zip(similarities[0], indices[0]):
                if sim >= threshold:
                    cached_response = self.response_cache.get(idx)
                    if cached_response:
                        logger.info(f"🎯 Semantic cache hit (similarity: {sim:.3f})")
                        
                        # Promote to L1 cache
                        self.query_cache[query_hash] = query
                        self.response_cache[query_hash] = cached_response
                        
                        return cached_response
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Cache get error: {e}")
            return None
    
    async def set(self, query: str, response: str):
        """Cache query-response pair"""
        try:
            query_embedding = self._encode_query(query)
            
            # Add to FAISS index
            self.faiss_index.add(query_embedding.reshape(1, -1))
            cache_id = self.faiss_index.ntotal - 1
            
            # Store in memory
            self.query_embeddings.append(query_embedding)
            self.query_cache[cache_id] = query
            self.response_cache[cache_id] = response
            
            # Store in Redis if available
            if self.redis_client:
                cache_key = f"semantic_cache:{cache_id}"
                cache_data = {
                    'query': query,
                    'response': response,
                    'embedding': query_embedding.tolist()
                }
                
                await self.redis_client.setex(
                    cache_key,
                    config.cache_ttl_seconds,
                    json.dumps(cache_data)
                )
                self.cache_keys.append(cache_key)
            
            logger.debug(f"💾 Cached query: {query[:50]}...")
            
            # Limit cache size
            if len(self.query_cache) > 1000:
                await self._evict_oldest()
                
        except Exception as e:
            logger.error(f"❌ Cache set error: {e}")
    
    async def _evict_oldest(self):
        """Evict oldest cache entries"""
        try:
            # Remove oldest 100 entries
            if self.cache_keys:
                keys_to_remove = self.cache_keys[:100]
                self.cache_keys = self.cache_keys[100:]
                
                if self.redis_client:
                    await self.redis_client.delete(*keys_to_remove)
                
                # Rebuild FAISS index (simple approach)
                if len(self.query_embeddings) > 100:
                    self.query_embeddings = self.query_embeddings[100:]
                    self.faiss_index = faiss.IndexFlatIP(self.dimension)
                    
                    if self.query_embeddings:
                        embeddings_array = np.array(self.query_embeddings).astype(np.float32)
                        self.faiss_index.add(embeddings_array)
                    
                    # Update cache IDs
                    new_query_cache = {}
                    new_response_cache = {}
                    
                    for old_id, query in self.query_cache.items():
                        if old_id >= 100:
                            new_id = old_id - 100
                            new_query_cache[new_id] = query
                            new_response_cache[new_id] = self.response_cache[old_id]
                    
                    self.query_cache = new_query_cache
                    self.response_cache = new_response_cache
                
                logger.info("🧹 Cache evicted old entries")
                
        except Exception as e:
            logger.error(f"❌ Cache eviction error: {e}")
    
    async def preload_common_queries(self):
        """Preload common queries for better performance"""
        if not config.preload_common_queries:
            return
            
        common_queries = [
            "what services do you offer",
            "what are your business hours", 
            "how can I contact support",
            "what is your pricing",
            "how do I get started",
            "can I speak to a human",
            "transfer me to an agent",
            "what is your refund policy",
            "do you offer technical support",
            "how do I cancel my subscription"
        ]
        
        logger.info("🔄 Preloading common queries...")
        
        # This would typically be filled from your knowledge base
        # For now, we'll just cache the queries without responses
        for query in common_queries:
            if not await self.get(query):
                # You would retrieve actual responses from your RAG system here
                placeholder_response = f"Information about '{query}' - please contact support"
                await self.set(query, placeholder_response)
        
        logger.info(f"✅ Preloaded {len(common_queries)} common queries")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "total_entries": self.faiss_index.ntotal,
            "memory_entries": len(self.query_cache),
            "redis_available": self.redis_client is not None,
            "cache_keys": len(self.cache_keys)
        }

# Global semantic cache instance
semantic_cache = AdvancedSemanticCache()