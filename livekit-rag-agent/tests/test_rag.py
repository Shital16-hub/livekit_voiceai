"""
Tests for RAG functionality
"""
import pytest
import asyncio
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.rag_manager import rag_manager, search_knowledge_base, get_context
from config import config

@pytest.fixture
async def initialized_rag():
    """Initialize RAG system for testing"""
    success = await rag_manager.load_or_create_index()
    if not success:
        pytest.skip("RAG system not available - run data_ingestion.py first")
    return rag_manager

@pytest.mark.asyncio
async def test_rag_initialization():
    """Test RAG system initialization"""
    success = await rag_manager.load_or_create_index()
    assert success, "RAG system should initialize successfully"
    assert rag_manager.index is not None, "Index should be loaded"
    assert rag_manager.retriever is not None, "Retriever should be ready"

@pytest.mark.asyncio
async def test_search_functionality(initialized_rag):
    """Test basic search functionality"""
    results = await initialized_rag.search("What services do you offer?")
    assert isinstance(results, list), "Search should return a list"
    
    if results:  # If we have results
        assert len(results) > 0, "Should return at least one result"
        assert "content" in results[0], "Results should have content"
        assert "score" in results[0], "Results should have similarity score"

@pytest.mark.asyncio
async def test_query_functionality(initialized_rag):
    """Test query engine functionality"""
    result = await search_knowledge_base("What are the business hours?")
    assert isinstance(result, str), "Query should return a string"
    assert len(result) > 0, "Query should return non-empty result"

@pytest.mark.asyncio
async def test_context_generation(initialized_rag):
    """Test context generation for chat injection"""
    context = await get_context("test query", max_length=100)
    
    if context:  # If context is found
        assert isinstance(context, str), "Context should be a string"
        assert len(context) <= 100, "Context should respect max_length"

@pytest.mark.asyncio
async def test_performance():
    """Test that RAG operations meet performance targets"""
    import time
    
    # Test search performance
    start_time = time.perf_counter()
    results = await rag_manager.search("test query")
    end_time = time.perf_counter()
    
    search_time_ms = (end_time - start_time) * 1000
    assert search_time_ms < config.rag_timeout_ms, f"Search took {search_time_ms:.1f}ms, should be < {config.rag_timeout_ms}ms"

@pytest.mark.asyncio
async def test_caching():
    """Test caching functionality"""
    if not config.enable_caching:
        pytest.skip("Caching disabled")
    
    query = "test caching query"
    
    # First query (cache miss)
    result1 = await search_knowledge_base(query)
    
    # Second query (should hit cache)
    result2 = await search_knowledge_base(query)
    
    # Results should be identical
    assert result1 == result2, "Cached results should match original"

def test_config_validation():
    """Test configuration validation"""
    assert config.chunk_size > 0, "Chunk size should be positive"
    assert config.top_k_results > 0, "Top K should be positive"
    assert config.rag_timeout_ms > 0, "Timeout should be positive"
    assert config.similarity_threshold >= 0.0, "Similarity threshold should be non-negative"

if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])