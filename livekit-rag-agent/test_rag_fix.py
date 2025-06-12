#!/usr/bin/env python3
"""
Quick test to verify RAG system is working
"""
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_rag():
    """Test the RAG system with detailed output"""
    try:
        from scalable_fast_rag import scalable_rag
        
        logger.info("🔧 Testing RAG system with detailed output...")
        
        # Initialize
        success = await scalable_rag.initialize()
        if not success:
            logger.error("❌ RAG initialization failed")
            return
        
        logger.info(f"📊 Loaded {len(scalable_rag.texts)} documents")
        logger.info(f"🎯 Similarity threshold: {scalable_rag.similarity_threshold}")
        
        # Test queries with detailed results
        test_queries = [
            "What services do you offer?",
            "What are your business hours?", 
            "How much does it cost?",
            "What features do you have?",
            "Can you integrate with our phone system?"
        ]
        
        for query in test_queries:
            logger.info(f"\n🔍 Testing: '{query}'")
            results = await scalable_rag.quick_search(query, top_k=3)
            
            if results:
                logger.info(f"✅ Found {len(results)} results:")
                for i, result in enumerate(results):
                    logger.info(f"  {i+1}. Score: {result['score']:.3f}")
                    logger.info(f"     Content: {result['content'][:100]}...")
                    logger.info(f"     Source: {result['source']}")
            else:
                logger.warning(f"⚠️ No results found")
        
        logger.info("\n✅ RAG system test completed")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_rag())