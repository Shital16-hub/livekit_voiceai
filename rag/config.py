"""General-purpose RAG configuration - FIXED for any knowledge base"""
import os
from dotenv import load_dotenv

load_dotenv()

# FIXED: Increased timeout from 5 to 15 seconds
RAG_CONFIG = {
    "similarity_top_k": int(os.getenv("TOP_K", 3)),
    "relevance_threshold": float(os.getenv("RELEVANCE_THRESHOLD", 0.7)),
    "max_context_length": int(os.getenv("MAX_CONTEXT_LENGTH", 2000)),
    
    # CRITICAL FIX: Increased timeout
    "query_timeout": 15.0,  # CHANGED: Was 5.0, now 15.0
    "cache_size": int(os.getenv("CACHE_SIZE", 1000)),
    
    # Document processing
    "chunk_size": int(os.getenv("CHUNK_SIZE", 512)),
    "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", 50)),
    
    # FIXED: Faster models for lower latency
    "embedding_model": "text-embedding-3-small",  # Faster than large
    "llm_model": "gpt-4o-mini",                   # Faster than gpt-4
    "max_tokens": 200,                            # Limited for speed
    "temperature": 0.1,                           # Lower for speed
}

# FIXED: Optimized Qdrant settings
QDRANT_CONFIG = {
    "url": os.getenv("QDRANT_CLOUD_URL"),
    "api_key": os.getenv("QDRANT_API_KEY"),
    "collection_name": os.getenv("COLLECTION_NAME", "general_knowledge"),
    "prefer_grpc": False,
    "timeout": 30.0,  # Optimized timeout
}

# Debug output
print("=== FIXED RAG CONFIGURATION ===")
print(f"🔧 RAG query_timeout: {RAG_CONFIG['query_timeout']} seconds (FIXED)")
print(f"🔧 QDRANT_URL: {QDRANT_CONFIG['url']}")
print(f"🔧 COLLECTION_NAME: {QDRANT_CONFIG['collection_name']}")
print("================================")