"""Generic RAG system configuration"""
import os

# RAG Performance Settings
RAG_CONFIG = {
    # Search settings
    "similarity_top_k": int(os.getenv("TOP_K", 3)),
    "relevance_threshold": float(os.getenv("RELEVANCE_THRESHOLD", 0.7)),
    "max_context_length": int(os.getenv("MAX_CONTEXT_LENGTH", 2000)),
    
    # Performance settings
    "embedding_batch_size": int(os.getenv("EMBEDDING_BATCH_SIZE", 20)),
    "query_timeout": float(os.getenv("RAG_TIMEOUT", 0.3)),
    "cache_size": int(os.getenv("CACHE_SIZE", 1000)),
    
    # Document processing
    "chunk_size": int(os.getenv("CHUNK_SIZE", 512)),
    "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", 50)),
    
    # Model settings
    "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
    "llm_model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
    "max_tokens": int(os.getenv("MAX_TOKENS", 200)),
    "temperature": float(os.getenv("TEMPERATURE", 0.1)),
}

# Qdrant Settings
QDRANT_CONFIG = {
    "url": os.getenv("QDRANT_CLOUD_URL"),
    "api_key": os.getenv("QDRANT_API_KEY"),
    "collection_name": os.getenv("COLLECTION_NAME", "general_knowledge"),
    "prefer_grpc": True,
    "timeout": 5.0,
}

# Cache settings
CACHE_CONFIG = {
    "enable_semantic_cache": True,
    "cache_similarity_threshold": 0.95,
    "max_cache_size": int(os.getenv("CACHE_SIZE", 500)),
    "cache_ttl": int(os.getenv("CACHE_TTL", 3600)),
}

# Document processing settings
DOCUMENT_CONFIG = {
    "supported_formats": [".txt", ".pdf", ".docx", ".md", ".json", ".csv"],
    "max_file_size_mb": 10,
    "auto_detect_language": True,
    "extract_metadata": True,
}