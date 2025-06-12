"""
Simplified Configuration for LiveKit RAG Agent
Focuses on core functionality first
"""
import os
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class SimplifiedConfig(BaseSettings):
    """Simplified configuration focusing on core telephony functionality"""
    
    # ✅ REQUIRED: LiveKit Settings
    livekit_url: str = Field(default="", env="LIVEKIT_URL")
    livekit_api_key: str = Field(default="", env="LIVEKIT_API_KEY")
    livekit_api_secret: str = Field(default="", env="LIVEKIT_API_SECRET")
    
    # ✅ REQUIRED: AI Service API Keys
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    deepgram_api_key: str = Field(default="", env="DEEPGRAM_API_KEY")
    
    # ✅ OPTIONAL: Enhanced TTS
    cartesia_api_key: Optional[str] = Field(default=None, env="CARTESIA_API_KEY")
    
    # ✅ TELEPHONY: SIP Transfer
    transfer_sip_address: str = Field(
        default="sip:voiceai@sip.linphone.org", 
        env="TRANSFER_SIP_ADDRESS"
    )
    
    # ✅ BASIC RAG SETTINGS (simplified)
    embedding_model: str = Field(default="text-embedding-3-small")
    chunk_size: int = Field(default=200)
    chunk_overlap: int = Field(default=50)
    max_tokens: int = Field(default=60)  # Short for voice
    
    # ✅ PERFORMANCE SETTINGS (more realistic for telephony)
    target_latency_ms: int = Field(default=3000)  # 3 seconds (more realistic)
    rag_timeout_ms: int = Field(default=1000)     # 1 second RAG timeout
    
    # ✅ FEATURE FLAGS (optimized for stability)
    enable_semantic_cache: bool = Field(default=True)   # Keep enabled
    enable_streaming_rag: bool = Field(default=True)    # Keep enabled
    enable_parallel_processing: bool = Field(default=True)  # Keep enabled
    enable_smart_bypass: bool = Field(default=True)     # Keep simple bypass
    
    # ✅ SIMPLE SETTINGS
    top_k_results: int = Field(default=2)
    similarity_threshold: float = Field(default=0.5)  # Lower threshold
    min_query_length: int = Field(default=3)
    
    # ✅ REDIS (optional for now)
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_db: int = Field(default=0)
    
    # ✅ DEBUGGING
    log_performance: bool = Field(default=True)
    debug_mode: bool = Field(default=True)
    
    # ✅ PATHS
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
config = SimplifiedConfig()
config.ensure_directories()

def validate_config():
    """Validate only essential configuration"""
    required_fields = [
        ("OPENAI_API_KEY", config.openai_api_key),
        ("DEEPGRAM_API_KEY", config.deepgram_api_key),
    ]
    
    missing_fields = [field for field, value in required_fields if not value]
    
    if missing_fields:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_fields)}")
    
    # ✅ OPTIONAL: Check LiveKit credentials
    if not config.livekit_url or not config.livekit_api_key:
        print("⚠️ LiveKit credentials not set - agent will use dev mode")
    
    print("✅ Essential configuration validated")
    print(f"📞 Transfer destination: {config.transfer_sip_address}")
    print(f"⚡ Target latency: {config.target_latency_ms}ms")
    print(f"🎯 RAG timeout: {config.rag_timeout_ms}ms")
    print(f"🔧 Debug mode: {config.debug_mode}")

if __name__ == "__main__":
    validate_config()