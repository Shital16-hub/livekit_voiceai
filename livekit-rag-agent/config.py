# config.py
"""
Optimized Configuration for LiveKit RAG Agent with Qdrant
"""
import os
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class QdrantConfig(BaseSettings):
    """Qdrant-specific configuration for ultra-low latency"""
    
    # ✅ REQUIRED: LiveKit Settings
    livekit_url: str = Field(default="", env="LIVEKIT_URL")
    livekit_api_key: str = Field(default="", env="LIVEKIT_API_KEY")
    livekit_api_secret: str = Field(default="", env="LIVEKIT_API_SECRET")
    
    # ✅ REQUIRED: AI Service API Keys
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    deepgram_api_key: str = Field(default="", env="DEEPGRAM_API_KEY")
    
    # ✅ OPTIONAL: Enhanced TTS
    cartesia_api_key: Optional[str] = Field(default=None, env="CARTESIA_API_KEY")
    elevenlabs_api_key: Optional[str] = Field(default=None, env="ELEVENLABS_API_KEY")
    eleven_api_key: Optional[str] = Field(default=None, env="ELEVEN_API_KEY")  # Added this
    
    # ✅ TWILIO/SIP SETTINGS (Added all your extra fields)
    twilio_account_sid: Optional[str] = Field(default=None, env="TWILIO_ACCOUNT_SID")
    twilio_phone_number: Optional[str] = Field(default=None, env="TWILIO_PHONE_NUMBER")
    twilio_auth_token: Optional[str] = Field(default=None, env="TWILIO_AUTH_TOKEN")
    twilio_trunk_sid: Optional[str] = Field(default=None, env="TWILIO_TRUNK_SID")
    sip_username: Optional[str] = Field(default=None, env="SIP_USERNAME")
    sip_password: Optional[str] = Field(default=None, env="SIP_PASSWORD")
    sip_domain: Optional[str] = Field(default=None, env="SIP_DOMAIN")
    sip_trunk_id: Optional[str] = Field(default=None, env="SIP_TRUNK_ID")
    livekit_sip_uri: Optional[str] = Field(default=None, env="LIVEKIT_SIP_URI")
    
    # ✅ QDRANT SETTINGS (Local or Cloud)
    qdrant_url: str = Field(default="http://localhost:6333", env="QDRANT_URL")
    qdrant_api_key: Optional[str] = Field(default=None, env="QDRANT_API_KEY")
    qdrant_collection_name: str = Field(default="telephony_knowledge", env="QDRANT_COLLECTION")
    qdrant_collection: str = Field(default="telephony_knowledge", env="QDRANT_COLLECTION")  # Added alias
    qdrant_prefer_grpc: bool = Field(default=True, env="QDRANT_PREFER_GRPC")
    qdrant_timeout: int = Field(default=5, env="QDRANT_TIMEOUT")
    
    # ✅ EMBEDDING SETTINGS (Optimized for telephony)
    embedding_model: str = Field(default="text-embedding-3-small", env="EMBEDDING_MODEL")
    embedding_dimensions: int = Field(default=1536, env="EMBEDDING_DIMENSIONS")
    
    # ✅ RAG SETTINGS (Telephony optimized)
    chunk_size: int = Field(default=300, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, env="CHUNK_OVERLAP")
    max_tokens: int = Field(default=50, env="MAX_TOKENS")  # Short for voice
    
    # ✅ PERFORMANCE SETTINGS (Sub-200ms target)
    rag_timeout_ms: int = Field(default=150, env="RAG_TIMEOUT_MS")
    search_limit: int = Field(default=3, env="SEARCH_LIMIT")
    similarity_threshold: float = Field(default=0.3, env="SIMILARITY_THRESHOLD")
    
    # ✅ TELEPHONY
    transfer_sip_address: str = Field(
        default="sip:voiceai@sip.linphone.org", 
        env="TRANSFER_SIP_ADDRESS"
    )
    
    # ✅ PATHS
    @property
    def project_root(self) -> Path:
        return Path(__file__).parent
    
    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"
    
    def ensure_directories(self):
        """Create necessary directories"""
        self.data_dir.mkdir(exist_ok=True)
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # This allows extra fields to be ignored

# Global configuration instance
config = QdrantConfig()
config.ensure_directories()

def validate_config():
    """Validate essential configuration"""
    required_fields = [
        ("OPENAI_API_KEY", config.openai_api_key),
        ("DEEPGRAM_API_KEY", config.deepgram_api_key),
    ]
    
    missing_fields = [field for field, value in required_fields if not value]
    
    if missing_fields:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_fields)}")
    
    print("✅ Configuration validated")
    print(f"📞 Transfer destination: {config.transfer_sip_address}")
    print(f"🔍 Qdrant URL: {config.qdrant_url}")
    print(f"⚡ RAG timeout: {config.rag_timeout_ms}ms")

if __name__ == "__main__":
    validate_config()