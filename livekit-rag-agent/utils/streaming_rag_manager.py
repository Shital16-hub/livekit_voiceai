"""
Advanced Streaming RAG Manager with Parallel Processing
"""
import asyncio
import logging
from typing import List, Optional, Dict, Any, AsyncGenerator
from pathlib import Path
import time

from llama_index.core import (
    VectorStoreIndex, 
    StorageContext, 
    load_index_from_storage,
    Settings
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import MetadataMode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from config import config
from utils.semantic_cache import semantic_cache

logger = logging.getLogger(__name__)

class StreamingRAGManager:
    """Advanced streaming RAG with parallel processing and semantic caching"""
    
    def __init__(self):
        self.index: Optional[VectorStoreIndex] = None
        self.retriever: Optional[VectorIndexRetriever] = None
        self.query_engine = None
        self.bypass_keywords = {
            "hello", "hi", "hey", "thanks", "thank you", "bye", "goodbye",
            "yes", "no", "okay", "sure", "what's your name", "how are you",
            "good morning", "good afternoon", "good evening"
        }
        self._setup_llama_index()
        
    def _setup_llama_index(self):
        """Configure LlamaIndex for optimal performance"""
        try:
            Settings.embed_model = OpenAIEmbedding(
                model=config.embedding_model,
                api_key=config.openai_api_key
            )
            
            Settings.llm = OpenAI(
                model="gpt-4o-mini",
                api_key=config.openai_api_key,
                temperature=0.1,
                max_tokens=config.max_tokens,
                timeout=10.0
            )
            
            Settings.chunk_size = config.chunk_size
            Settings.chunk_overlap = config.chunk_overlap
            
            logger.info("✅ Streaming RAG Manager configured")
            
        except Exception as e:
            logger.error(f"❌ RAG Manager configuration error: {e}")
            raise
    
    async def initialize(self) -> bool:
        """Initialize RAG system with caching"""
        try:
            # Initialize semantic cache
            await semantic_cache.initialize()
            
            # Load RAG index
            storage_dir = config.storage_dir
            
            if storage_dir.exists() and any(storage_dir.iterdir()):
                logger.info("📂 Loading existing vector index...")
                storage_context = StorageContext.from_defaults(persist_dir=str(storage_dir))
                self.index = load_index_from_storage(storage_context)
                logger.info("✅ Existing index loaded")
            else:
                logger.error("❌ No RAG index found - run data ingestion first")
                return False
            
            # Setup retriever and query engine
            self.retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=config.top_k_results,
            )
            
            self.query_engine = self.index.as_query_engine(
                use_async=True,
                similarity_top_k=config.top_k_results,
                response_mode="compact",
                streaming=True  # Enable streaming
            )
            
            # Preload common queries
            await semantic_cache.preload_common_queries()
            
            logger.info("🚀 Streaming RAG system ready")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize streaming RAG: {e}")
            return False
    
    def should_bypass_rag(self, query: str) -> bool:
        """Smart bypass logic for simple queries"""
        if not config.enable_smart_bypass:
            return False
            
        query_lower = query.lower().strip()
        
        # Skip very short queries
        if len(query.split()) < config.min_query_length:
            return True
            
        # Skip greeting/social queries
        if any(keyword in query_lower for keyword in self.bypass_keywords):
            return True
            
        # Skip if query is too simple
        if len(query_lower) < 10:
            return True
            
        return False
    
    async def get_streaming_context(self, query: str) -> AsyncGenerator[str, None]:
        """Get streaming context for real-time injection"""
        try:
            if self.should_bypass_rag(query):
                logger.info(f"⚡ Bypassing RAG for simple query: {query[:30]}...")
                return
            
            # Check semantic cache first
            if config.enable_semantic_cache:
                cached_response = await semantic_cache.get(query)
                if cached_response:
                    logger.info("🎯 Using cached response")
                    yield cached_response
                    return
            
            # Parallel retrieval
            start_time = time.time()
            
            if config.enable_parallel_processing:
                # Start multiple retrieval tasks
                tasks = [
                    self._fast_retrieval(query),
                    self._detailed_retrieval(query)
                ]
                
                # Get results as they complete
                for coro in asyncio.as_completed(tasks):
                    try:
                        result = await asyncio.wait_for(coro, timeout=0.5)
                        if result:
                            yield result
                            break  # Use first good result
                    except asyncio.TimeoutError:
                        continue
            else:
                # Standard retrieval
                result = await self._fast_retrieval(query)
                if result:
                    yield result
            
            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(f"⏱️ RAG retrieval: {elapsed_ms:.1f}ms")
            
        except Exception as e:
            logger.error(f"❌ Streaming context error: {e}")
    
    async def _fast_retrieval(self, query: str) -> Optional[str]:
        """Fast retrieval with aggressive timeout"""
        try:
            if not self.retriever:
                return None
            
            nodes = await asyncio.wait_for(
                self.retriever.aretrieve(query),
                timeout=0.2  # 200ms timeout
            )
            
            # Process only the best result
            if nodes:
                best_node = nodes[0]
                if hasattr(best_node, 'score') and best_node.score >= config.similarity_threshold:
                    content = best_node.get_content(metadata_mode=MetadataMode.NONE)
                    # Limit content for voice
                    if len(content) > 100:
                        content = content[:100] + "..."
                    return content
            
            return None
            
        except asyncio.TimeoutError:
            logger.debug("⏰ Fast retrieval timeout")
            return None
        except Exception as e:
            logger.error(f"❌ Fast retrieval error: {e}")
            return None
    
    async def _detailed_retrieval(self, query: str) -> Optional[str]:
        """More detailed retrieval for better context"""
        try:
            if not self.query_engine:
                return None
            
            response = await asyncio.wait_for(
                self.query_engine.aquery(query),
                timeout=0.4  # 400ms timeout
            )
            
            result = str(response).strip()
            if len(result) > 150:
                result = result[:150] + "..."
            
            # Cache the result
            if config.enable_semantic_cache and result:
                await semantic_cache.set(query, result)
            
            return result
            
        except asyncio.TimeoutError:
            logger.debug("⏰ Detailed retrieval timeout")
            return None
        except Exception as e:
            logger.error(f"❌ Detailed retrieval error: {e}")
            return None
    
    async def quick_search(self, query: str) -> List[Dict[str, Any]]:
        """Quick search for immediate context"""
        try:
            if self.should_bypass_rag(query):
                return []
            
            # Check cache first
            if config.enable_semantic_cache:
                cached_response = await semantic_cache.get(query)
                if cached_response:
                    return [{
                        "content": cached_response,
                        "score": 1.0,
                        "source": "cache"
                    }]
            
            # Fast retrieval
            if not self.retriever:
                return []
            
            nodes = await asyncio.wait_for(
                self.retriever.aretrieve(query),
                timeout=config.rag_timeout_ms / 1000
            )
            
            results = []
            for node in nodes[:config.top_k_results]:
                if hasattr(node, 'score') and node.score >= config.similarity_threshold:
                    content = node.get_content(metadata_mode=MetadataMode.NONE)
                    results.append({
                        "content": content[:150],  # Limit for voice
                        "score": node.score,
                        "source": node.metadata.get("file_name", "kb")
                    })
            
            return results
            
        except asyncio.TimeoutError:
            logger.warning(f"⏰ Quick search timeout for: {query[:30]}")
            return []
        except Exception as e:
            logger.error(f"❌ Quick search error: {e}")
            return []
    
    async def enhanced_query(self, question: str) -> str:
        """Enhanced query with streaming and caching"""
        try:
            if self.should_bypass_rag(question):
                return "I'm ready to help! What would you like to know?"
            
            # Check cache first
            if config.enable_semantic_cache:
                cached_response = await semantic_cache.get(question)
                if cached_response:
                    logger.info("🎯 Returning cached response")
                    return cached_response
            
            # Stream context and generate response
            context_parts = []
            async for context in self.get_streaming_context(question):
                context_parts.append(context)
                if len(context_parts) >= 2:  # Limit context parts
                    break
            
            if context_parts:
                combined_context = " ".join(context_parts)
                # Cache the result
                if config.enable_semantic_cache:
                    await semantic_cache.set(question, combined_context)
                return combined_context
            else:
                return "I don't have specific information about that. Is there something else I can help with?"
                
        except Exception as e:
            logger.error(f"❌ Enhanced query error: {e}")
            return "I encountered an issue searching for that information."
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        base_stats = {
            "index_loaded": self.index is not None,
            "retriever_ready": self.retriever is not None,
            "query_engine_ready": self.query_engine is not None,
            "bypass_enabled": config.enable_smart_bypass,
            "parallel_processing": config.enable_parallel_processing,
            "streaming_enabled": config.enable_streaming_rag,
        }
        
        # Add cache stats
        if config.enable_semantic_cache:
            base_stats.update(semantic_cache.get_stats())
        
        return base_stats

# Global streaming RAG manager instance
streaming_rag_manager = StreamingRAGManager()