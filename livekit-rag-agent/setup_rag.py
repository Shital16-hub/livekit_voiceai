#!/usr/bin/env python3
"""
Setup Script for Scalable Fast RAG System
"""
import asyncio
import json
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_data():
    """Create sample data files for testing"""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # 1. Create comprehensive knowledge base
    knowledge_base = {
        "services": "We offer comprehensive AI voice assistant services including 24/7 customer support automation, voice-enabled information systems, call routing and transfer services, multi-language support, and integration with existing systems.",
        "hours": "We're available 24/7 to assist you with our AI voice assistant services.",
        "pricing": "For pricing information and custom quotes, please contact our sales team. We offer flexible plans for businesses of all sizes.",
        "support": "Our AI voice assistant provides 24/7 support. For complex issues, we can transfer you to a human agent.",
        "features": "Our AI voice assistant features include natural language processing, multi-language support, seamless call transfers, knowledge base integration, and real-time responses with sub-second latency.",
        "contact": "You can reach us through this voice assistant 24/7, or request to speak with a human agent for specialized assistance.",
        "company": "We specialize in AI voice assistant technology, providing automated customer support and voice-enabled solutions for businesses worldwide.",
        "technical": "Our voice assistant uses advanced AI technology including speech recognition, natural language processing, and text-to-speech synthesis for ultra-fast responses.",
        "integration": "Our voice assistant integrates seamlessly with existing phone systems, CRM platforms, and business applications through standard APIs.",
        "languages": "Our voice assistant supports multiple languages and can handle international customers with natural language processing capabilities.",
        "products": "Our main products include AI voice assistants, automated customer support systems, call routing solutions, and telephony integration platforms.",
        "benefits": "Benefits include 24/7 availability, reduced wait times, consistent service quality, significant cost savings, and improved customer satisfaction scores.",
        "implementation": "Implementation typically takes 1-2 weeks and includes system integration, comprehensive training, and testing phases with full technical support.",
        "reliability": "Our systems maintain 99.9% uptime with redundant infrastructure, automatic failover capabilities, and 24/7 monitoring.",
        "security": "We implement enterprise-grade security including end-to-end encryption, secure data handling, and compliance with industry standards like GDPR and HIPAA.",
        "customization": "Our voice assistants can be fully customized for specific industries, complex workflows, and unique business requirements.",
        "analytics": "We provide detailed analytics and reporting on call volumes, resolution rates, customer satisfaction metrics, and performance insights.",
        "training": "Our AI continuously learns and improves from interactions while maintaining strict data privacy and security protocols.",
        "scalability": "Our platform scales automatically to handle varying call volumes from small businesses to enterprise levels with millions of interactions.",
        "api": "We offer comprehensive REST and GraphQL APIs for integration with existing systems, CRM platforms, and custom applications."
    }
    
    with open(data_dir / "simple_knowledge.json", 'w') as f:
        json.dump(knowledge_base, f, indent=2)
    
    # 2. Create FAQ data
    faq_data = [
        {
            "question": "How quickly can your voice assistant respond?",
            "answer": "Our voice assistant typically responds in under 2 seconds, with most responses delivered in under 1 second for optimal conversation flow."
        },
        {
            "question": "What industries do you serve?",
            "answer": "We serve healthcare, finance, retail, telecommunications, insurance, real estate, and many other industries with specialized voice assistant solutions."
        },
        {
            "question": "Can you integrate with our existing phone system?",
            "answer": "Yes, we integrate with virtually all phone systems including traditional PBX, VoIP, and cloud-based systems through standard SIP protocols."
        },
        {
            "question": "How accurate is the speech recognition?",
            "answer": "Our speech recognition achieves over 95% accuracy in typical business environments, with specialized models for different industries and accents."
        },
        {
            "question": "Do you support international customers?",
            "answer": "Yes, we support over 30 languages and regional dialects, with native-speaking voice models for natural conversation experiences."
        }
    ]
    
    with open(data_dir / "faq.json", 'w') as f:
        json.dump(faq_data, f, indent=2)
    
    # 3. Create product documentation
    product_docs = """# AI Voice Assistant Platform

## Overview
Our AI Voice Assistant Platform provides enterprise-grade voice automation solutions for businesses of all sizes.

## Key Features
- **Ultra-Low Latency**: Sub-second response times for natural conversations
- **24/7 Availability**: Never miss a call with our always-on AI assistants
- **Multi-Language Support**: Serve customers in their preferred language
- **Seamless Integration**: Works with existing phone systems and business tools
- **Advanced Analytics**: Detailed insights into customer interactions and performance

## Use Cases
### Customer Support
Automate routine inquiries, provide instant answers, and escalate complex issues to human agents.

### Lead Qualification
Capture and qualify leads 24/7, ensuring no potential customer is missed.

### Appointment Scheduling
Allow customers to book, reschedule, or cancel appointments through natural voice interactions.

### Order Processing
Handle orders, check inventory, and process payments through voice commands.

## Technical Specifications
- Response Time: <2 seconds average, <1 second for cached responses
- Uptime: 99.9% guaranteed with redundant infrastructure
- Scalability: Handles 1 to 10,000+ concurrent calls
- Security: Enterprise-grade encryption and compliance certifications
"""
    
    with open(data_dir / "product_docs.md", 'w') as f:
        f.write(product_docs)
    
    logger.info("✅ Created sample data files:")
    logger.info("  - data/simple_knowledge.json (basic knowledge base)")
    logger.info("  - data/faq.json (frequently asked questions)")
    logger.info("  - data/product_docs.md (product documentation)")

def install_dependencies():
    """Install required dependencies"""
    try:
        import subprocess
        import sys
        
        dependencies = [
            "sentence-transformers",
            "faiss-cpu",
            "pandas",
            "PyPDF2",
            "markdown"
        ]
        
        logger.info("📦 Installing optional dependencies...")
        for dep in dependencies:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
                logger.info(f"✅ Installed {dep}")
            except subprocess.CalledProcessError:
                logger.warning(f"⚠️ Failed to install {dep} - some features may be limited")
        
        logger.info("✅ Dependency installation completed")
        
    except Exception as e:
        logger.error(f"❌ Dependency installation failed: {e}")

async def test_rag_system():
    """Test the RAG system with sample queries"""
    try:
        from scalable_fast_rag import scalable_rag
        
        logger.info("🔧 Testing RAG system...")
        
        # Initialize
        success = await scalable_rag.initialize()
        if not success:
            logger.error("❌ RAG initialization failed")
            return
        
        # Test queries
        test_queries = [
            "What services do you offer?",
            "What are your business hours?",
            "How much does it cost?",
            "Can you integrate with our phone system?",
            "What languages do you support?"
        ]
        
        logger.info("🧪 Running test queries...")
        for query in test_queries:
            results = await scalable_rag.quick_search(query)
            if results:
                logger.info(f"✅ '{query}' -> Found {len(results)} results")
            else:
                logger.warning(f"⚠️ '{query}' -> No results found")
        
        logger.info("✅ RAG system test completed")
        
    except Exception as e:
        logger.error(f"❌ RAG system test failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Setup RAG system")
    parser.add_argument("--sample-data", action="store_true", help="Create sample data files")
    parser.add_argument("--install-deps", action="store_true", help="Install optional dependencies")
    parser.add_argument("--test", action="store_true", help="Test the RAG system")
    parser.add_argument("--all", action="store_true", help="Do everything")
    
    args = parser.parse_args()
    
    if args.all:
        args.sample_data = True
        args.install_deps = True
        args.test = True
    
    if args.install_deps:
        install_dependencies()
    
    if args.sample_data:
        create_sample_data()
    
    if args.test:
        asyncio.run(test_rag_system())
    
    if not any([args.sample_data, args.install_deps, args.test, args.all]):
        parser.print_help()

if __name__ == "__main__":
    main()