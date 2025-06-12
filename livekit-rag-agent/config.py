"""
Enhanced Configuration with Advanced RAG Settings
"""
import os
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class EnhancedAgentConfig(BaseSettings):
    """Enhanced agent configuration with advanced RAG optimizations"""
    
    # LiveKit Settings
    livekit_url: str = Field(default="", env="LIVEKIT_URL")
    livekit_api_key: str = Field(default="", env="LIVEKIT_API_KEY")
    livekit_api_secret: str = Field(default="", env="LIVEKIT_API_SECRET")
    
    # AI Service API Keys
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    deepgram_api_key: str = Field(default="", env="DEEPGRAM_API_KEY")
    cartesia_api_key: Optional[str] = Field(default=None, env="CARTESIA_API_KEY")
    
    # Enhanced RAG Configuration
    embedding_model: str = Field(default="text-embedding-3-small", env="EMBEDDING_MODEL")
    chunk_size: int = Field(default=150, env="CHUNK_SIZE")  # Smaller for voice
    chunk_overlap: int = Field(default=20, env="CHUNK_OVERLAP")
    max_tokens: int = Field(default=50, env="MAX_TOKENS")  # Much shorter for voice
    
    # Advanced RAG Settings
    top_k_results: int = Field(default=2, env="TOP_K_RESULTS")
    similarity_threshold: float = Field(default=0.6, env="SIMILARITY_THRESHOLD")  # Higher threshold
    enable_semantic_cache: bool = Field(default=True, env="ENABLE_SEMANTIC_CACHE")
    enable_streaming_rag: bool = Field(default=True, env="ENABLE_STREAMING_RAG")
    enable_parallel_processing: bool = Field(default=True, env="ENABLE_PARALLEL_PROCESSING")
    
    # Performance Settings
    target_latency_ms: int = Field(default=1500, env="TARGET_LATENCY_MS")  # More aggressive
    rag_timeout_ms: int = Field(default=300, env="RAG_TIMEOUT_MS")  # Very fast
    cache_ttl_seconds: int = Field(default=3600, env="CACHE_TTL_SECONDS")
    preload_common_queries: bool = Field(default=True, env="PRELOAD_COMMON_QUERIES")
    
    # Redis Settings for Semantic Cache
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_db: int = Field(default=0, env="REDIS_DB")
    
    # Smart Bypass Settings
    enable_smart_bypass: bool = Field(default=True, env="ENABLE_SMART_BYPASS")
    bypass_simple_queries: bool = Field(default=True, env="BYPASS_SIMPLE_QUERIES")
    min_query_length: int = Field(default=4, env="MIN_QUERY_LENGTH")
    
    # SIP Transfer
    transfer_sip_address: str = Field(
        default="sip:voiceai@sip.linphone.org", 
        env="TRANSFER_SIP_ADDRESS"
    )
    
    # File Paths
    @property
    def project_root(self) -> Path:
        return Path(__file__).parent
    
    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"
    
    @property
    def storage_dir(self) -> Path:
        return self.project_root / "storage"
    
    @property
    def cache_dir(self) -> Path:
        return self.project_root / "cache"
    
    def ensure_directories(self):
        """Create necessary directories"""
        for directory in [self.data_dir, self.storage_dir, self.cache_dir]:
            directory.mkdir(exist_ok=True)
    
    class ConfigDict:
        env_file = ".env"
        case_sensitive = False

# Global configuration instance
config = EnhancedAgentConfig()
config.ensure_directories()

def validate_config():
    """Validate required configuration"""
    required_fields = [
        ("OPENAI_API_KEY", config.openai_api_key),
        ("DEEPGRAM_API_KEY", config.deepgram_api_key),
    ]
    
    missing_fields = [field for field, value in required_fields if not value]
    
    if missing_fields:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_fields)}")
    
    print("✅ Enhanced configuration validated successfully")