#!/usr/bin/env python3
"""
Knowledge Base Inspector - See what's currently indexed in your RAG system
"""
import asyncio
import json
import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def inspect_knowledge_base():
    """Inspect what's currently in the RAG knowledge base"""
    
    logger.info("🔍 KNOWLEDGE BASE INSPECTION")
    logger.info("=" * 60)
    
    # Check data directory
    data_dir = Path("data")
    if data_dir.exists():
        logger.info("📁 Data Directory Contents:")
        for file in data_dir.glob("*"):
            if file.is_file():
                size = file.stat().st_size
                logger.info(f"   📄 {file.name} ({size} bytes)")
    else:
        logger.warning("⚠️ No data directory found")
    
    # Check RAG storage
    storage_dir = Path("rag_storage")
    if storage_dir.exists():
        logger.info("\n💾 RAG Storage Contents:")
        for file in storage_dir.glob("*"):
            size = file.stat().st_size
            logger.info(f"   📄 {file.name} ({size} bytes)")
    else:
        logger.warning("⚠️ No RAG storage found")
    
    # Load and display current knowledge base
    await display_current_knowledge()
    
    # Test the RAG system
    await test_current_rag()

async def display_current_knowledge():
    """Display the current knowledge base contents"""
    logger.info("\n📚 CURRENT KNOWLEDGE BASE CONTENTS:")
    logger.info("=" * 50)
    
    # Check simple_knowledge.json
    knowledge_file = Path("data/simple_knowledge.json")
    if knowledge_file.exists():
        try:
            with open(knowledge_file, 'r') as f:
                knowledge = json.load(f)
            
            logger.info("📖 Main Knowledge Base (simple_knowledge.json):")
            for i, (key, value) in enumerate(knowledge.items(), 1):
                logger.info(f"\n{i}. 🏷️ Topic: {key}")
                logger.info(f"   📝 Content: {value[:150]}...")
                logger.info(f"   📏 Length: {len(value)} characters")
        
        except Exception as e:
            logger.error(f"❌ Error reading knowledge file: {e}")
    else:
        logger.warning("⚠️ No simple_knowledge.json found")
    
    # Check other data files
    data_dir = Path("data")
    if data_dir.exists():
        other_files = [f for f in data_dir.glob("*.json") if f.name != "simple_knowledge.json"]
        if other_files:
            logger.info("\n📋 Additional Data Files:")
            for file in other_files:
                try:
                    with open(file, 'r') as f:
                        data = json.load(f)
                    logger.info(f"   📄 {file.name}: {len(data)} items")
                except Exception as e:
                    logger.warning(f"   ⚠️ {file.name}: Error reading - {e}")
    
    # Check technical docs
    tech_dir = Path("technical_docs")
    if tech_dir.exists():
        tech_files = list(tech_dir.glob("*.txt"))
        if tech_files:
            logger.info("\n🔧 Technical Documentation:")
            for file in tech_files:
                size = file.stat().st_size
                logger.info(f"   📄 {file.name} ({size} bytes)")

async def test_current_rag():
    """Test what the RAG system can actually find"""
    logger.info("\n🧪 TESTING CURRENT RAG SYSTEM:")
    logger.info("=" * 50)
    
    try:
        from scalable_fast_rag import scalable_rag
        
        # Initialize
        success = await scalable_rag.initialize()
        if not success:
            logger.error("❌ RAG system failed to initialize")
            return
        
        logger.info(f"✅ RAG System Status: {len(scalable_rag.texts)} documents loaded")
        
        # Show all indexed documents
        logger.info("\n📚 All Indexed Documents:")
        for i, doc in enumerate(scalable_rag.texts):
            content = doc.get("content", "")
            source = doc.get("source", "unknown")
            logger.info(f"\n{i+1}. 📄 ID: {doc.get('id', 'unknown')}")
            logger.info(f"   📁 Source: {source}")
            logger.info(f"   📝 Content: {content[:100]}...")
            logger.info(f"   📏 Length: {len(content)} chars")
        
        # Test sample queries
        test_queries = [
            "What services do you offer?",
            "What are your business hours?",
            "How much does it cost?",
            "What features do you have?",
            "How do I contact you?",
            "What languages do you support?",
            "How do I integrate?",
            "What is your uptime?"
        ]
        
        logger.info("\n🎯 Testing Sample Queries:")
        working_queries = 0
        
        for query in test_queries:
            results = await scalable_rag.quick_search(query, top_k=1)
            if results:
                working_queries += 1
                result = results[0]
                logger.info(f"✅ '{query}' -> Score: {result['score']:.3f}")
                logger.info(f"   Response: {result['content'][:80]}...")
            else:
                logger.warning(f"❌ '{query}' -> No results")
        
        success_rate = (working_queries / len(test_queries)) * 100
        logger.info(f"\n📊 Query Success Rate: {working_queries}/{len(test_queries)} ({success_rate:.1f}%)")
        
    except Exception as e:
        logger.error(f"❌ RAG testing failed: {e}")

async def show_rag_statistics():
    """Show detailed RAG statistics"""
    logger.info("\n📊 RAG SYSTEM STATISTICS:")
    logger.info("=" * 50)
    
    try:
        from scalable_fast_rag import scalable_rag
        
        await scalable_rag.initialize()
        
        if not scalable_rag.texts:
            logger.warning("⚠️ No documents in RAG system")
            return
        
        # Document statistics
        total_docs = len(scalable_rag.texts)
        total_chars = sum(len(doc.get("content", "")) for doc in scalable_rag.texts)
        avg_chars = total_chars / total_docs if total_docs > 0 else 0
        
        logger.info(f"📄 Total Documents: {total_docs}")
        logger.info(f"📝 Total Characters: {total_chars:,}")
        logger.info(f"📏 Average Document Length: {avg_chars:.0f} chars")
        
        # Source breakdown
        sources = {}
        for doc in scalable_rag.texts:
            source = doc.get("source", "unknown")
            sources[source] = sources.get(source, 0) + 1
        
        logger.info("\n📁 Documents by Source:")
        for source, count in sources.items():
            logger.info(f"   {source}: {count} documents")
        
        # Content samples
        logger.info("\n📋 Sample Content:")
        for i, doc in enumerate(scalable_rag.texts[:3]):
            content = doc.get("content", "")[:200]
            logger.info(f"   {i+1}. {content}...")
        
    except Exception as e:
        logger.error(f"❌ Statistics generation failed: {e}")

def main():
    """Main inspection function"""
    asyncio.run(inspect_knowledge_base())
    asyncio.run(show_rag_statistics())
    
    print("\n" + "=" * 60)
    print("💡 HOW TO ADD MORE KNOWLEDGE:")
    print("1. Add files to data/ directory (JSON, TXT, MD)")
    print("2. Run: python data_ingestion_script.py --directory data --recursive")
    print("3. Or run: python setup_rag.py --all")
    print("4. Restart your agent")
    print("\n📖 SUPPORTED FORMATS:")
    print("- JSON files with key-value pairs")
    print("- Text files (.txt)")
    print("- Markdown files (.md)")
    print("- PDF files (.pdf) if PyPDF2 installed")

if __name__ == "__main__":
    main()