"""
SCALABLE FAST RAG: Real vector search that can index any data
Optimized for sub-2-second telephony responses
"""
import asyncio
import logging
import os
import json
import pickle
import time
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import numpy as np

# Fast vector search with FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available - install with: pip install faiss-cpu")

# Lightweight embedding
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("SentenceTransformers not available - install with: pip install sentence-transformers")

# OpenAI fallback
import openai

logger = logging.getLogger(__name__)

class ScalableFastRAG:
    """
    ✅ SCALABLE: Real RAG system that can index any data with fast retrieval
    """
    
    def __init__(self):
        self.texts = []
        self.embeddings = None
        self.faiss_index = None
        self.embedding_model = None
        self.ready = False
        self.cache = {}
        
        # Configuration
        self.embedding_dim = 384  # MiniLM dimension
        self.max_cache_size = 100
        self.similarity_threshold = 0.3  # ✅ LOWERED: More permissive threshold
        
    async def initialize(self) -> bool:
        """Initialize the RAG system with real vector search"""
        try:
            start_time = time.time()
            
            # ✅ STEP 1: Initialize embedding model
            if not await self._init_embedding_model():
                return False
            
            # ✅ STEP 2: Load or create vector index
            if not await self._load_or_create_index():
                return False
            
            self.ready = True
            elapsed = (time.time() - start_time) * 1000
            logger.info(f"✅ Scalable RAG initialized in {elapsed:.1f}ms")
            return True
            
        except Exception as e:
            logger.error(f"❌ RAG initialization failed: {e}")
            return False
    
    async def _init_embedding_model(self) -> bool:
        """Initialize lightweight embedding model"""
        try:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                # ✅ FAST: Use lightweight model
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.embedding_dim = 384
                logger.info("✅ Using SentenceTransformers (MiniLM)")
                return True
            else:
                # ✅ FALLBACK: Use OpenAI (slower but works)
                if os.getenv("OPENAI_API_KEY"):
                    self.embedding_model = "openai"
                    self.embedding_dim = 1536  # OpenAI embedding size
                    logger.info("✅ Using OpenAI embeddings")
                    return True
                else:
                    logger.error("❌ No embedding model available")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Embedding model init failed: {e}")
            return False
    
    async def _load_or_create_index(self) -> bool:
        """Load existing index or create new one"""
        try:
            storage_dir = Path("rag_storage")
            storage_dir.mkdir(exist_ok=True)
            
            index_file = storage_dir / "faiss_index.bin"
            texts_file = storage_dir / "texts.pkl"
            
            # Try to load existing index
            if index_file.exists() and texts_file.exists():
                return await self._load_existing_index(index_file, texts_file)
            else:
                return await self._create_new_index()
                
        except Exception as e:
            logger.error(f"❌ Index loading failed: {e}")
            return False
    
    async def _load_existing_index(self, index_file: Path, texts_file: Path) -> bool:
        """Load existing FAISS index"""
        try:
            if not FAISS_AVAILABLE:
                logger.warning("⚠️ FAISS not available, using fallback")
                return await self._create_simple_fallback()
            
            # Load FAISS index
            self.faiss_index = faiss.read_index(str(index_file))
            
            # Load texts
            with open(texts_file, 'rb') as f:
                self.texts = pickle.load(f)
            
            logger.info(f"✅ Loaded existing index with {len(self.texts)} documents")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load existing index: {e}")
            return await self._create_new_index()
    
    async def _create_new_index(self) -> bool:
        """Create new index from data files"""
        try:
            # ✅ STEP 1: Load documents from data directory
            documents = await self._load_documents()
            if not documents:
                logger.warning("⚠️ No documents found, creating minimal index")
                documents = self._get_default_documents()
            
            # ✅ STEP 2: Create embeddings
            embeddings = await self._create_embeddings(documents)
            if embeddings is None:
                return False
            
            # ✅ STEP 3: Create FAISS index
            if FAISS_AVAILABLE:
                return await self._create_faiss_index(documents, embeddings)
            else:
                return await self._create_simple_fallback()
                
        except Exception as e:
            logger.error(f"❌ Failed to create index: {e}")
            return False
    
    async def _load_documents(self) -> List[Dict[str, str]]:
        """Load documents from various sources"""
        documents = []
        data_dir = Path("data")
        
        if not data_dir.exists():
            return documents
        
        # ✅ Load from JSON files
        for json_file in data_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        for key, value in data.items():
                            documents.append({
                                "id": f"{json_file.stem}_{key}",
                                "content": str(value),
                                "source": str(json_file),
                                "category": key
                            })
                    elif isinstance(data, list):
                        for i, item in enumerate(data):
                            documents.append({
                                "id": f"{json_file.stem}_{i}",
                                "content": str(item),
                                "source": str(json_file),
                                "category": "general"
                            })
            except Exception as e:
                logger.warning(f"⚠️ Failed to load {json_file}: {e}")
        
        # ✅ Load from text files
        for txt_file in data_dir.glob("*.txt"):
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        # Split into chunks if large
                        chunks = self._chunk_text(content)
                        for i, chunk in enumerate(chunks):
                            documents.append({
                                "id": f"{txt_file.stem}_{i}",
                                "content": chunk,
                                "source": str(txt_file),
                                "category": "document"
                            })
            except Exception as e:
                logger.warning(f"⚠️ Failed to load {txt_file}: {e}")
        
        logger.info(f"✅ Loaded {len(documents)} documents from {data_dir}")
        return documents
    
    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                if break_point > start + chunk_size // 2:
                    chunk = text[start:break_point + 1]
                    end = break_point + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
            
        return chunks
    
    def _get_default_documents(self) -> List[Dict[str, str]]:
        """Default documents if no data found"""
        return [
            {
                "id": "services_main",
                "content": "We offer comprehensive AI voice assistant services including 24/7 customer support automation, voice-enabled information systems, call routing and transfer services, multi-language support, and integration with existing systems.",
                "source": "default",
                "category": "services"
            },
            {
                "id": "hours_main", 
                "content": "We're available 24/7 to assist you with our AI voice assistant services.",
                "source": "default",
                "category": "hours"
            },
            {
                "id": "support_main",
                "content": "Our AI voice assistant provides 24/7 support. For complex issues, we can transfer you to a human agent.",
                "source": "default", 
                "category": "support"
            }
        ]
    
    async def _create_embeddings(self, documents: List[Dict[str, str]]) -> Optional[np.ndarray]:
        """Create embeddings for documents"""
        try:
            texts = [doc["content"] for doc in documents]
            self.texts = documents  # Store documents
            
            if self.embedding_model == "openai":
                # ✅ OpenAI embeddings
                embeddings = []
                client = openai.OpenAI()
                
                for text in texts:
                    response = await asyncio.to_thread(
                        client.embeddings.create,
                        model="text-embedding-3-small",
                        input=text[:8000]  # Limit text length
                    )
                    embeddings.append(response.data[0].embedding)
                
                return np.array(embeddings, dtype=np.float32)
                
            else:
                # ✅ SentenceTransformers
                embeddings = await asyncio.to_thread(
                    self.embedding_model.encode,
                    texts,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                return embeddings.astype(np.float32)
                
        except Exception as e:
            logger.error(f"❌ Embedding creation failed: {e}")
            return None
    
    async def _create_faiss_index(self, documents: List[Dict[str, str]], embeddings: np.ndarray) -> bool:
        """Create FAISS index"""
        try:
            # ✅ Create FAISS index
            self.faiss_index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner product for speed
            self.faiss_index.add(embeddings)
            
            # ✅ Save index and texts
            storage_dir = Path("rag_storage")
            storage_dir.mkdir(exist_ok=True)
            
            faiss.write_index(self.faiss_index, str(storage_dir / "faiss_index.bin"))
            
            with open(storage_dir / "texts.pkl", 'wb') as f:
                pickle.dump(documents, f)
            
            logger.info(f"✅ Created FAISS index with {len(documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"❌ FAISS index creation failed: {e}")
            return False
    
    async def _create_simple_fallback(self) -> bool:
        """Simple fallback without FAISS"""
        try:
            documents = await self._load_documents()
            if not documents:
                documents = self._get_default_documents()
            
            self.texts = documents
            logger.info(f"✅ Created simple fallback with {len(documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"❌ Fallback creation failed: {e}")
            return False
    
    async def quick_search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Fast search with sub-200ms target"""
        if not self.ready:
            return []
        
        try:
            start_time = time.time()
            
            # ✅ Check cache first
            cache_key = query.lower().strip()
            if cache_key in self.cache:
                elapsed = (time.time() - start_time) * 1000
                logger.info(f"⚡ Cache hit in {elapsed:.1f}ms")
                return self.cache[cache_key]
            
            # ✅ Fast search
            if self.faiss_index is not None:
                results = await self._faiss_search(query, top_k)
            else:
                results = await self._simple_search(query, top_k)
            
            # ✅ Cache results
            if len(self.cache) < self.max_cache_size:
                self.cache[cache_key] = results
            
            elapsed = (time.time() - start_time) * 1000
            logger.info(f"⚡ Search completed in {elapsed:.1f}ms")
            return results
            
        except Exception as e:
            logger.error(f"❌ Search error: {e}")
            return []
    
    async def _faiss_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """FAISS-based vector search"""
        try:
            # Create query embedding
            if self.embedding_model == "openai":
                client = openai.OpenAI()
                response = await asyncio.to_thread(
                    client.embeddings.create,
                    model="text-embedding-3-small",
                    input=query[:8000]
                )
                query_embedding = np.array([response.data[0].embedding], dtype=np.float32)
            else:
                query_embedding = await asyncio.to_thread(
                    self.embedding_model.encode,
                    [query],
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                query_embedding = query_embedding.astype(np.float32)
            
            # Search FAISS index
            scores, indices = self.faiss_index.search(query_embedding, top_k)
            
            # Format results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.texts) and score > self.similarity_threshold:
                    doc = self.texts[idx]
                    results.append({
                        "content": doc["content"],
                        "score": float(score),
                        "source": doc.get("source", "unknown"),
                        "id": doc.get("id", f"doc_{idx}")
                    })
            
            # ✅ FALLBACK: If no results above threshold, take best result anyway
            if not results and len(scores[0]) > 0 and scores[0][0] > 0.1:
                best_idx = indices[0][0]
                if best_idx < len(self.texts):
                    doc = self.texts[best_idx]
                    results.append({
                        "content": doc["content"],
                        "score": float(scores[0][0]),
                        "source": doc.get("source", "unknown"),
                        "id": doc.get("id", f"doc_{best_idx}")
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"❌ FAISS search error: {e}")
            return []
    
    async def _simple_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Simple keyword-based search fallback"""
        try:
            query_words = set(query.lower().split())
            results = []
            
            for doc in self.texts:
                content_words = set(doc["content"].lower().split())
                overlap = len(query_words.intersection(content_words))
                
                if overlap > 0:
                    score = overlap / max(len(query_words), len(content_words))
                    if score > 0.1:  # Lower threshold for keyword search
                        results.append({
                            "content": doc["content"],
                            "score": score,
                            "source": doc.get("source", "unknown"),
                            "id": doc.get("id", "unknown")
                        })
            
            # Sort by score and return top_k
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"❌ Simple search error: {e}")
            return []
    
    async def add_documents(self, new_documents: List[Dict[str, str]]) -> bool:
        """Add new documents to the index"""
        try:
            logger.info(f"📝 Adding {len(new_documents)} new documents...")
            
            # Add to existing texts
            self.texts.extend(new_documents)
            
            # Create embeddings for new documents
            embeddings = await self._create_embeddings(new_documents)
            if embeddings is None:
                return False
            
            # Add to FAISS index
            if self.faiss_index is not None:
                self.faiss_index.add(embeddings)
            
            # Save updated index
            storage_dir = Path("rag_storage")
            if self.faiss_index is not None:
                faiss.write_index(self.faiss_index, str(storage_dir / "faiss_index.bin"))
            
            with open(storage_dir / "texts.pkl", 'wb') as f:
                pickle.dump(self.texts, f)
            
            # Clear cache since index changed
            self.cache.clear()
            
            logger.info(f"✅ Added {len(new_documents)} documents successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to add documents: {e}")
            return False

# Global scalable RAG instance
scalable_rag = ScalableFastRAG()