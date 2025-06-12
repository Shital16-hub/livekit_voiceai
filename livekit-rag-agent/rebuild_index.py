#!/usr/bin/env python3
"""
Rebuild RAG index with fixed content processing
"""
import asyncio
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def rebuild_index():
    """Rebuild the RAG index with proper content formatting"""
    try:
        # Remove old index
        storage_dir = Path("rag_storage")
        if storage_dir.exists():
            logger.info("🗑️ Removing old RAG index...")
            shutil.rmtree(storage_dir)
        
        # Initialize fresh
        from scalable_fast_rag import scalable_rag
        
        logger.info("🔨 Rebuilding RAG index with fixed content processing...")
        success = await scalable_rag.initialize()
        
        if success:
            logger.info("✅ RAG index rebuilt successfully!")
            
            # Test a few queries
            test_queries = [
                "What services do you offer?",
                "What are your business hours?",
                "How much does it cost?"
            ]
            
            logger.info("🧪 Testing rebuilt index...")
            for query in test_queries:
                results = await scalable_rag.quick_search(query)
                if results:
                    content = results[0]["content"]
                    logger.info(f"✅ '{query}' -> {content[:100]}...")
                else:
                    logger.warning(f"⚠️ '{query}' -> No results")
        else:
            logger.error("❌ Failed to rebuild RAG index")
            
    except Exception as e:
        logger.error(f"❌ Rebuild failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(rebuild_index())