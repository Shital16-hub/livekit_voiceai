"""
Configuration management for LiveKit RAG Agent
"""
import os
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AgentConfig(BaseSettings):
    """Agent configuration settings"""
    
    # LiveKit Settings
    livekit_url: str = Field(default="", env="LIVEKIT_URL")
    livekit_api_key: str = Field(default="", env="LIVEKIT_API_KEY")
    livekit_api_secret: str = Field(default="", env="LIVEKIT_API_SECRET")
    
    # AI Service API Keys
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    deepgram_api_key: str = Field(default="", env="DEEPGRAM_API_KEY")
    elevenlabs_api_key: Optional[str] = Field(default=None, env="ELEVENLABS_API_KEY")
    cartesia_api_key: Optional[str] = Field(default=None, env="CARTESIA_API_KEY")
    
    # RAG Configuration
    embedding_model: str = Field(default="text-embedding-3-small", env="EMBEDDING_MODEL")
    chunk_size: int = Field(default=200, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=30, env="CHUNK_OVERLAP")
    max_tokens: int = Field(default=150, env="MAX_TOKENS")
    top_k_results: int = Field(default=2, env="TOP_K_RESULTS")
    similarity_threshold: float = Field(default=0.3, env="SIMILARITY_THRESHOLD")
    
    # Performance Settings
    target_latency_ms: int = Field(default=2000, env="TARGET_LATENCY_MS")
    rag_timeout_ms: int = Field(default=5000, env="RAG_TIMEOUT_MS")  # 5 seconds
    enable_caching: bool = Field(default=True, env="ENABLE_CACHING")
    log_performance: bool = Field(default=True, env="LOG_PERFORMANCE")
    
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
    def logs_dir(self) -> Path:
        return self.project_root / "logs"
    
    def ensure_directories(self):
        """Create necessary directories if they don't exist"""
        for directory in [self.data_dir, self.storage_dir, self.logs_dir]:
            directory.mkdir(exist_ok=True)
    
    class ConfigDict:
        env_file = ".env"
        case_sensitive = False

# Global configuration instance
config = AgentConfig()

# Ensure directories exist
config.ensure_directories()

# Validation
def validate_config():
    """Validate required configuration settings"""
    required_fields = [
        ("OPENAI_API_KEY", config.openai_api_key),
        ("DEEPGRAM_API_KEY", config.deepgram_api_key),
    ]
    
    missing_fields = [field for field, value in required_fields if not value]
    
    if missing_fields:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_fields)}")
    
    print("✅ Configuration validated successfully")

if __name__ == "__main__":
    validate_config()
    print(f"📁 Data directory: {config.data_dir}")
    print(f"💾 Storage directory: {config.storage_dir}")
    print(f"📋 Logs directory: {config.logs_dir}")