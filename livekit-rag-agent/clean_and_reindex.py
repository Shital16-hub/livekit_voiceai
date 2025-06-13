#!/usr/bin/env python3
"""
Clean and Reindex RAG System
1. Clean all existing indexes
2. Add your new PDF
3. Rebuild with proper content
"""
import asyncio
import shutil
import json
import logging
from pathlib import Path
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_everything():
    """Remove all existing RAG data"""
    logger.info("🧹 CLEANING ALL EXISTING RAG DATA")
    logger.info("=" * 50)
    
    # Remove RAG storage
    storage_dir = Path("rag_storage")
    if storage_dir.exists():
        shutil.rmtree(storage_dir)
        logger.info("✅ Removed rag_storage directory")
    
    # Remove cache
    cache_dir = Path("cache")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        logger.info("✅ Removed cache directory")
    
    # Clean data directory (optional)
    data_dir = Path("data")
    if data_dir.exists():
        # List current files
        logger.info("\n📁 Current data directory contents:")
        for file in data_dir.glob("*"):
            if file.is_file():
                logger.info(f"   📄 {file.name}")
        
        clean_data = input("\n🤔 Do you want to clean the data directory too? (y/N): ").strip().lower()
        if clean_data == 'y':
            shutil.rmtree(data_dir)
            data_dir.mkdir(exist_ok=True)
            logger.info("✅ Cleaned data directory")
        else:
            logger.info("⏭️ Keeping existing data directory")
    else:
        data_dir.mkdir(exist_ok=True)
        logger.info("✅ Created data directory")
    
    logger.info("\n🎉 Clean completed!")

def setup_pdf_processing():
    """Set up for PDF processing"""
    logger.info("\n📚 SETTING UP PDF PROCESSING")
    logger.info("=" * 40)
    
    # Check if PyPDF2 is installed
    try:
        import PyPDF2
        logger.info("✅ PyPDF2 is available for PDF processing")
        return True
    except ImportError:
        logger.warning("⚠️ PyPDF2 not installed")
        install = input("📦 Install PyPDF2 for PDF processing? (Y/n): ").strip().lower()
        if install != 'n':
            import subprocess
            try:
                subprocess.check_call(["pip", "install", "PyPDF2"])
                logger.info("✅ PyPDF2 installed successfully")
                return True
            except subprocess.CalledProcessError:
                logger.error("❌ Failed to install PyPDF2")
                return False
        return False

def add_pdf_files():
    """Help user add PDF files"""
    logger.info("\n📄 ADD YOUR PDF FILES")
    logger.info("=" * 30)
    
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    logger.info(f"📁 Data directory: {data_dir.absolute()}")
    logger.info("\n📋 Instructions:")
    logger.info("1. Copy your PDF files to the data/ directory")
    logger.info("2. Supported formats: .pdf, .txt, .md, .json")
    logger.info("3. Press Enter when ready to continue...")
    
    input()
    
    # List files in data directory
    files = list(data_dir.glob("*"))
    if files:
        logger.info("\n📄 Files found in data directory:")
        for file in files:
            if file.is_file():
                size = file.stat().st_size
                logger.info(f"   📄 {file.name} ({size} bytes)")
        return True
    else:
        logger.warning("⚠️ No files found in data directory")
        return False

async def rebuild_index():
    """Rebuild the RAG index with all available data"""
    logger.info("\n🔨 REBUILDING RAG INDEX")
    logger.info("=" * 30)
    
    try:
        from data_ingestion_script import DataIngestion
        
        # Process all files in data directory
        ingestion = DataIngestion()
        data_dir = Path("data")
        
        if not data_dir.exists():
            logger.error("❌ No data directory found")
            return False
        
        all_documents = []
        
        # Process all files
        for file_path in data_dir.glob("*"):
            if file_path.is_file():
                logger.info(f"📄 Processing: {file_path.name}")
                documents = ingestion.process_file(file_path)
                if documents:
                    logger.info(f"   ✅ Extracted {len(documents)} chunks")
                    all_documents.extend(documents)
                else:
                    logger.warning(f"   ⚠️ No content extracted from {file_path.name}")
        
        if not all_documents:
            logger.error("❌ No documents extracted from any files")
            return False
        
        logger.info(f"\n📚 Total documents extracted: {len(all_documents)}")
        
        # Save processed documents
        processed_file = data_dir / "processed_documents.json"
        with open(processed_file, 'w', encoding='utf-8') as f:
            json.dump(all_documents, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Saved processed documents to: {processed_file}")
        
        # Initialize the RAG system
        logger.info("\n🚀 Initializing RAG system...")
        from scalable_fast_rag import scalable_rag
        
        success = await scalable_rag.initialize()
        if success:
            logger.info(f"✅ RAG system initialized with {len(scalable_rag.texts)} documents")
            return True
        else:
            logger.error("❌ RAG system initialization failed")
            return False
    
    except Exception as e:
        logger.error(f"❌ Rebuild failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_new_index():
    """Test the newly built index"""
    logger.info("\n🧪 TESTING NEW INDEX")
    logger.info("=" * 25)
    
    try:
        from scalable_fast_rag import scalable_rag
        
        # Initialize
        await scalable_rag.initialize()
        
        logger.info(f"📊 Total documents: {len(scalable_rag.texts)}")
        
        # Show sample documents
        logger.info("\n📋 Sample indexed content:")
        for i, doc in enumerate(scalable_rag.texts[:3]):
            content = doc.get("content", "")
            source = doc.get("source", "unknown")
            logger.info(f"\n{i+1}. 📄 Source: {Path(source).name}")
            logger.info(f"   📝 Content: {content[:100]}...")
        
        # Test queries
        test_queries = [
            "What services do you offer?",
            "What are your business hours?",
            "How much does it cost?",
            "What features do you have?"
        ]
        
        logger.info("\n🎯 Testing queries:")
        working_queries = 0
        
        for query in test_queries:
            results = await scalable_rag.quick_search(query, top_k=1)
            if results:
                working_queries += 1
                result = results[0]
                logger.info(f"✅ '{query}' -> Score: {result['score']:.3f}")
                logger.info(f"   📝 Response: {result['content'][:80]}...")
            else:
                logger.warning(f"❌ '{query}' -> No results")
        
        success_rate = (working_queries / len(test_queries)) * 100
        logger.info(f"\n📊 Success Rate: {working_queries}/{len(test_queries)} ({success_rate:.1f}%)")
        
        if success_rate > 50:
            logger.info("🎉 RAG system is working well!")
            return True
        else:
            logger.warning("⚠️ RAG system needs improvement")
            return False
    
    except Exception as e:
        logger.error(f"❌ Testing failed: {e}")
        return False

def create_sample_pdf_content():
    """Create sample content if no PDFs are provided"""
    logger.info("\n📝 CREATING SAMPLE CONTENT")
    logger.info("=" * 35)
    
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Create comprehensive business knowledge
    business_knowledge = {
        "services": "We offer comprehensive AI voice assistant services including 24/7 customer support automation, voice-enabled information systems, call routing and transfer services, multi-language support, and integration with existing business systems.",
        "hours": "Our AI voice assistant is available 24 hours a day, 7 days a week to assist you. For complex issues requiring human support, our live agents are available Monday through Friday from 9 AM to 6 PM EST.",
        "pricing": "We offer flexible pricing plans to meet different business needs. Our basic plan starts at $99 per month, professional plan at $299 per month, and we provide custom enterprise pricing for large organizations.",
        "support": "Our AI provides instant support through natural voice interaction. For complex technical issues, we can immediately transfer you to a qualified human support specialist.",
        "features": "Our voice assistant includes advanced features such as natural language processing, sub-second response times, multi-language support, seamless call transfers, comprehensive knowledge base integration, and real-time voice interaction.",
        "contact": "You can reach us through this AI voice assistant 24/7 for immediate assistance. If you need to speak with a human representative, just ask and we can transfer you right away.",
        "company": "We are a leading provider of AI voice assistant technology, specializing in automated customer support and voice-enabled business solutions with enterprise-grade reliability.",
        "integration": "Our voice assistant integrates seamlessly with existing phone systems, CRM platforms, helpdesk software, and business applications through standard APIs and webhooks.",
        "technical": "Our system uses advanced AI technology including speech recognition, natural language processing, and text-to-speech synthesis to provide ultra-fast voice responses with 99.9% uptime.",
        "languages": "We support over 30 languages and regional dialects with native-speaking voice models for natural conversation experiences including English, Spanish, French, German, Italian, and many others.",
        "security": "We implement enterprise-grade security including end-to-end encryption, secure data handling, SOC 2 Type II compliance, and GDPR compliance for all voice interactions.",
        "implementation": "Getting started is simple - our implementation typically takes 1-2 weeks and includes system integration, comprehensive training, testing, and full technical support."
    }
    
    # Save to JSON
    with open(data_dir / "business_knowledge.json", 'w') as f:
        json.dump(business_knowledge, f, indent=2)
    
    # Create detailed FAQ
    detailed_faq = """
FREQUENTLY ASKED QUESTIONS

Q: What types of businesses can benefit from your AI voice assistant?
A: Our AI voice assistant is perfect for any business that receives phone calls - from small startups to large enterprises. We serve healthcare, finance, retail, real estate, legal services, and many other industries.

Q: How quickly can you implement the voice assistant for our business?
A: Implementation typically takes 1-2 weeks from start to finish. This includes system integration, training the AI on your specific business needs, testing, and full deployment with ongoing support.

Q: What happens if the AI can't answer a customer's question?
A: Our AI is designed to seamlessly transfer customers to human agents when needed. The transfer happens instantly and includes context about the customer's inquiry for a smooth handoff.

Q: How much does your AI voice assistant cost?
A: We offer flexible pricing starting at $99/month for basic plans, $299/month for professional features, and custom enterprise pricing. The cost depends on call volume, features needed, and integration requirements.

Q: Is our customer data secure with your voice assistant?
A: Absolutely. We implement enterprise-grade security including end-to-end encryption, secure data handling, and compliance with industry standards like GDPR and HIPAA where applicable.

Q: Can the AI handle multiple languages?
A: Yes! Our voice assistant supports over 30 languages and regional dialects, allowing you to serve international customers in their preferred language with natural conversation flow.

Q: What if we already have a phone system in place?
A: Our voice assistant integrates seamlessly with virtually all existing phone systems including traditional PBX, VoIP, and cloud-based systems through standard SIP protocols.

Q: How accurate is the speech recognition?
A: Our speech recognition achieves over 95% accuracy in typical business environments, with specialized models optimized for different industries, accents, and terminology.
"""
    
    with open(data_dir / "detailed_faq.txt", 'w') as f:
        f.write(detailed_faq)
    
    logger.info("✅ Created comprehensive sample content")
    logger.info("   📄 business_knowledge.json - 12 business topics")
    logger.info("   📄 detailed_faq.txt - 8 detailed FAQ items")

async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Clean and reindex RAG system")
    parser.add_argument("--skip-clean", action="store_true", help="Skip cleaning existing data")
    parser.add_argument("--auto", action="store_true", help="Run automatically with defaults")
    
    args = parser.parse_args()
    
    logger.info("🔄 RAG SYSTEM CLEAN AND REINDEX")
    logger.info("=" * 60)
    
    # Step 1: Clean everything
    if not args.skip_clean:
        clean_everything()
    
    # Step 2: Setup PDF processing
    pdf_ready = setup_pdf_processing()
    
    # Step 3: Add files
    if not args.auto:
        files_added = add_pdf_files()
        if not files_added:
            logger.info("📝 No files found. Creating sample content...")
            create_sample_pdf_content()
    else:
        logger.info("📝 Auto mode: Creating sample content...")
        create_sample_pdf_content()
    
    # Step 4: Rebuild index
    logger.info("\n🔨 Starting rebuild process...")
    success = await rebuild_index()
    
    if success:
        # Step 5: Test new index
        await test_new_index()
        
        logger.info("\n🎉 RAG SYSTEM REBUILD COMPLETE!")
        logger.info("✅ Your agent now has fresh, properly indexed content")
        logger.info("🚀 You can now restart your agent: python ultra_fast_rag_agent.py dev")
    else:
        logger.error("\n❌ RAG system rebuild failed")
        logger.info("💡 Check the logs above for specific errors")

if __name__ == "__main__":
    asyncio.run(main())