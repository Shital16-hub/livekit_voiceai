# qdrant_rag_system.py
"""
Ultra-Fast Qdrant RAG System for LiveKit Telephony
Optimized for sub-200ms query latency
"""
import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import uuid

from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct, Filter, 
    FieldCondition, MatchValue, SearchParams, OptimizersConfigDiff
)
import openai
from sentence_transformers import SentenceTransformer

from config import config

logger = logging.getLogger(__name__)

class QdrantRAGSystem:
    """
    Ultra-fast Qdrant RAG system optimized for telephony applications
    """
    
    def __init__(self):
        self.client: Optional[QdrantClient] = None
        self.aclient: Optional[AsyncQdrantClient] = None
        self.openai_client: Optional[openai.AsyncOpenAI] = None
        self.embedding_model: Optional[SentenceTransformer] = None
        self.ready = False
        self.cache = {}
        self.max_cache_size = 100
        
    async def initialize(self) -> bool:
        """Initialize the Qdrant RAG system"""
        try:
            start_time = time.time()
            
            # Initialize clients
            await self._init_clients()
            
            # Initialize embedding model
            await self._init_embedding_model()
            
            # Setup collection
            await self._setup_collection()
            
            # Load existing data if available
            await self._load_existing_data()
            
            self.ready = True
            elapsed = (time.time() - start_time) * 1000
            logger.info(f"✅ Qdrant RAG initialized in {elapsed:.1f}ms")
            return True
            
        except Exception as e:
            logger.error(f"❌ Qdrant RAG initialization failed: {e}")
            return False
    
    async def _init_clients(self):
        """Initialize Qdrant clients"""
        try:
            # Sync client for setup operations
            self.client = QdrantClient(
                url=config.qdrant_url,
                api_key=config.qdrant_api_key,
                timeout=config.qdrant_timeout,
                prefer_grpc=config.qdrant_prefer_grpc
            )
            
            # Async client for queries
            self.aclient = AsyncQdrantClient(
                url=config.qdrant_url,
                api_key=config.qdrant_api_key,
                timeout=config.qdrant_timeout,
                prefer_grpc=config.qdrant_prefer_grpc
            )
            
            # OpenAI async client
            self.openai_client = openai.AsyncOpenAI(
                api_key=config.openai_api_key
            )
            
            logger.info("✅ Qdrant clients initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize clients: {e}")
            raise
    
    async def _init_embedding_model(self):
        """Initialize embedding model"""
        try:
            if config.embedding_model.startswith("text-embedding"):
                # Use OpenAI embeddings
                logger.info("✅ Using OpenAI embeddings")
            else:
                # Use local SentenceTransformer
                self.embedding_model = await asyncio.to_thread(
                    SentenceTransformer,
                    config.embedding_model
                )
                logger.info(f"✅ Using local embedding model: {config.embedding_model}")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize embedding model: {e}")
            raise
    
    async def _setup_collection(self):
        """Setup optimized Qdrant collection"""
        try:
            collection_name = config.qdrant_collection_name
            
            # Check if collection exists
            collections = await asyncio.to_thread(
                self.client.get_collections
            )
            
            collection_exists = any(
                col.name == collection_name 
                for col in collections.collections
            )
            
            if not collection_exists:
                # Create optimized collection
                await asyncio.to_thread(
                    self.client.create_collection,
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=config.embedding_dimensions,
                        distance=Distance.COSINE,
                        # Optimize for speed
                        hnsw_config={
                            "m": 8,  # Lower for faster search
                            "ef_construct": 64,  # Lower for faster indexing
                            "full_scan_threshold": 10000,
                            "max_indexing_threads": 0,  # Use all available
                        }
                    ),
                    # Optimize storage for telephony
                    optimizers_config=OptimizersConfigDiff(
                        default_segment_number=2,
                        max_segment_size=None,
                        memmap_threshold=None,
                        indexing_threshold=20000,
                        flush_interval_sec=5,
                        max_optimization_threads=2
                    )
                )
                logger.info(f"✅ Created optimized collection: {collection_name}")
            else:
                logger.info(f"✅ Using existing collection: {collection_name}")
                
        except Exception as e:
            logger.error(f"❌ Failed to setup collection: {e}")
            raise
    
    async def _load_existing_data(self):
        """Load existing data from data directory"""
        try:
            data_dir = config.data_dir
            if not data_dir.exists():
                logger.info("📁 No data directory found, skipping data load")
                return
            
            # Check if collection has data
            collection_info = await asyncio.to_thread(
                self.client.get_collection,
                config.qdrant_collection_name
            )
            
            if collection_info.points_count > 0:
                logger.info(f"✅ Collection has {collection_info.points_count} existing points")
                return
            
            # Load and index data
            documents = await self._load_documents_from_directory(data_dir)
            if documents:
                await self.add_documents(documents)
                logger.info(f"✅ Loaded {len(documents)} documents into collection")
            
        except Exception as e:
            logger.error(f"❌ Failed to load existing data: {e}")
    
    async def _load_documents_from_directory(self, data_dir: Path) -> List[Dict[str, Any]]:
        """Load documents from data directory"""
        documents = []
        
        # Load JSON files
        for json_file in data_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                if isinstance(data, dict):
                    for key, value in data.items():
                        documents.append({
                            "id": f"{json_file.stem}_{key}",
                            "text": str(value),
                            "metadata": {
                                "source": str(json_file),
                                "category": key,
                                "type": "json_entry"
                            }
                        })
                elif isinstance(data, list):
                    for i, item in enumerate(data):
                        content = str(item.get("content", item) if isinstance(item, dict) else item)
                        documents.append({
                            "id": f"{json_file.stem}_{i}",
                            "text": content,
                            "metadata": {
                                "source": str(json_file),
                                "category": "list_item",
                                "type": "json_list"
                            }
                        })
                        
            except Exception as e:
                logger.warning(f"⚠️ Failed to load {json_file}: {e}")
        
        # Load text files
        for txt_file in data_dir.glob("*.txt"):
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        # Split large files into chunks
                        chunks = self._chunk_text(content)
                        for i, chunk in enumerate(chunks):
                            documents.append({
                                "id": f"{txt_file.stem}_chunk_{i}",
                                "text": chunk,
                                "metadata": {
                                    "source": str(txt_file),
                                    "category": "document",
                                    "type": "text_chunk"
                                }
                            })
                            
            except Exception as e:
                logger.warning(f"⚠️ Failed to load {txt_file}: {e}")
        
        return documents
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks optimized for telephony"""
        if len(text) <= config.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + config.chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                if break_point > start + config.chunk_size // 2:
                    chunk = text[start:break_point + 1]
                    end = break_point + 1
            
            chunks.append(chunk.strip())
            start = end - config.chunk_overlap
        
        return [c for c in chunks if c.strip()]
    
    async def _create_embedding(self, text: str) -> List[float]:
        """Create embedding for text"""
        try:
            if config.embedding_model.startswith("text-embedding"):
                # Use OpenAI
                response = await self.openai_client.embeddings.create(
                    model=config.embedding_model,
                    input=text[:8000]  # Limit length
                )
                return response.data[0].embedding
            else:
                # Use local model
                embedding = await asyncio.to_thread(
                    self.embedding_model.encode,
                    text,
                    show_progress_bar=False
                )
                return embedding.tolist()
                
        except Exception as e:
            logger.error(f"❌ Failed to create embedding: {e}")
            raise
    
    async def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add documents to Qdrant collection"""
        try:
            points = []
            
            for doc in documents:
                # Create embedding
                embedding = await self._create_embedding(doc["text"])
                
                # Create point
                point = PointStruct(
                    id=doc["id"],
                    vector=embedding,
                    payload={
                        "text": doc["text"],
                        **doc.get("metadata", {})
                    }
                )
                points.append(point)
            
            # Batch upsert
            await asyncio.to_thread(
                self.client.upsert,
                collection_name=config.qdrant_collection_name,
                points=points
            )
            
            logger.info(f"✅ Added {len(points)} documents to Qdrant")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to add documents: {e}")
            return False
    
    async def search(self, query: str, limit: int = None) -> List[Dict[str, Any]]:
        """Ultra-fast search with caching"""
        if not self.ready:
            return []
        
        try:
            start_time = time.time()
            
            # Check cache
            cache_key = f"{query.lower().strip()}_{limit}"
            if cache_key in self.cache:
                elapsed = (time.time() - start_time) * 1000
                logger.info(f"⚡ Cache hit in {elapsed:.1f}ms")
                return self.cache[cache_key]
            
            # Create query embedding
            query_embedding = await self._create_embedding(query)
            
            # Search with timeout
            search_result = await asyncio.wait_for(
                self.aclient.search(
                    collection_name=config.qdrant_collection_name,
                    query_vector=query_embedding,
                    limit=limit or config.search_limit,
                    score_threshold=config.similarity_threshold,
                    search_params=SearchParams(
                        hnsw_ef=32,  # Lower for faster search
                        exact=False
                    )
                ),
                timeout=config.rag_timeout_ms / 1000.0
            )
            
            # Format results
            results = []
            for point in search_result:
                results.append({
                    "id": str(point.id),
                    "text": point.payload.get("text", ""),
                    "score": float(point.score),
                    "metadata": {
                        k: v for k, v in point.payload.items() 
                        if k != "text"
                    }
                })
            
            # Cache results
            if len(self.cache) < self.max_cache_size:
                self.cache[cache_key] = results
            
            elapsed = (time.time() - start_time) * 1000
            logger.info(f"⚡ Search completed in {elapsed:.1f}ms, found {len(results)} results")
            return results
            
        except asyncio.TimeoutError:
            logger.warning(f"⚠️ Search timeout after {config.rag_timeout_ms}ms")
            return []
        except Exception as e:
            logger.error(f"❌ Search error: {e}")
            return []
    
    async def close(self):
        """Clean up resources"""
        try:
            if self.client:
                self.client.close()
            if self.aclient:
                await self.aclient.close()
            logger.info("✅ Qdrant RAG system closed")
        except Exception as e:
            logger.error(f"❌ Error closing Qdrant RAG system: {e}")

    async def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add documents to Qdrant collection"""
        try:
            points = []
            
            for doc in documents:
                # Create embedding
                embedding = await self._create_embedding(doc["text"])
                
                # Generate UUID for point ID
                point_id = str(uuid.uuid4())
                
                # Create point
                point = PointStruct(
                    id=point_id,  # Use UUID instead of string ID
                    vector=embedding,
                    payload={
                        "text": doc["text"],
                        "original_id": doc["id"],  # Store original ID in payload
                        **doc.get("metadata", {})
                    }
                )
                points.append(point)
            
            # Batch upsert
            await asyncio.to_thread(
                self.client.upsert,
                collection_name=config.qdrant_collection_name,
                points=points
            )
            
            logger.info(f"✅ Added {len(points)} documents to Qdrant")
            return True
        
        except Exception as e:
            logger.error(f"❌ Failed to add documents: {e}")
            return False

# Global instance
qdrant_rag = QdrantRAGSystem()