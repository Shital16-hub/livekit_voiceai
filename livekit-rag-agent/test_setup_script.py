#!/usr/bin/env python3
"""
Quick setup and test script for the RAG Voice Agent
"""
import asyncio
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def setup_and_test():
    """Setup data and test the RAG system"""
    
    # Step 1: Create sample data if needed
    logger.info("🔧 Setting up sample data...")
    try:
        from setup_rag import create_sample_data
        create_sample_data()
        logger.info("✅ Sample data created")
    except Exception as e:
        logger.error(f"❌ Failed to create sample data: {e}")
    
    # Step 2: Test the RAG system
    logger.info("🧪 Testing RAG system...")
    try:
        from scalable_fast_rag import scalable_rag
        
        # Initialize
        success = await scalable_rag.initialize()
        if not success:
            logger.error("❌ RAG initialization failed")
            return False
        
        # Test queries
        test_queries = [
            "What services do you offer?",
            "What are your business hours?",
            "How much does it cost?"
        ]
        
        logger.info("🎯 Testing with sample queries...")
        for query in test_queries:
            results = await scalable_rag.quick_search(query)
            if results:
                logger.info(f"✅ '{query}' -> Found {len(results)} results")
                logger.info(f"   Response: {results[0]['content'][:100]}...")
            else:
                logger.warning(f"⚠️ '{query}' -> No results")
        
        logger.info("✅ RAG system test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ RAG test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(setup_and_test())
    if success:
        print("\n🎉 Setup complete! You can now run:")
        print("   python ultra_fast_rag_agent.py dev")
    else:
        print("\n❌ Setup failed. Please check the errors above.")