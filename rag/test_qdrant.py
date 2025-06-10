from dotenv import load_dotenv
import os
from qdrant_client import QdrantClient

load_dotenv()

def test_qdrant_credentials():
    print("🔍 Testing Qdrant Cloud Connection...")
    
    url = os.getenv("QDRANT_CLOUD_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    
    print(f"URL: {url}")
    print(f"API Key: {api_key[:20]}...{api_key[-10:] if api_key else 'NOT SET'}")
    
    try:
        client = QdrantClient(
            url=url,
            api_key=api_key,
            timeout=30.0,
            prefer_grpc=False
        )
        
        print("📡 Attempting connection...")
        collections = client.get_collections()
        print("✅ SUCCESS: Connected to Qdrant Cloud!")
        print(f"Collections found: {len(collections.collections)}")
        
        for col in collections.collections:
            print(f"  - {col.name}")
            
    except Exception as e:
        print(f"❌ FAILED: {e}")
        
        if "403" in str(e) or "Forbidden" in str(e):
            print("\n💡 TROUBLESHOOTING 403 FORBIDDEN:")
            print("1. Check your Qdrant Cloud dashboard")
            print("2. Verify your API key is correct")
            print("3. Make sure your cluster is running")
            print("4. API key might be expired")
            print("5. Check if you have the right permissions")

if __name__ == "__main__":
    test_qdrant_credentials()