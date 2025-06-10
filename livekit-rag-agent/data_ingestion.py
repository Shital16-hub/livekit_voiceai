"""
Data Ingestion Script for LiveKit RAG Agent
Prepares knowledge base from various document formats
"""
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any
import argparse

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Document,
    Settings
)
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding

from config import config, validate_config
from utils.performance_monitor import time_function

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataIngestionPipeline:
    """Data ingestion pipeline for knowledge base preparation"""
    
    def __init__(self):
        self.documents: List[Document] = []
        self.index: VectorStoreIndex = None
        
        # Setup LlamaIndex
        self._configure_llama_index()
    
    def _configure_llama_index(self):
        """Configure LlamaIndex for optimal performance"""
        Settings.embed_model = OpenAIEmbedding(
            model=config.embedding_model,
            api_key=config.openai_api_key
        )
        Settings.chunk_size = config.chunk_size
        Settings.chunk_overlap = config.chunk_overlap
        
        logger.info("✅ LlamaIndex configured")
    
    @time_function("document_loading")
    def load_documents(self, data_path: Path = None) -> bool:
        """Load documents from data directory"""
        try:
            if data_path is None:
                data_path = config.data_dir
            
            if not data_path.exists():
                logger.error(f"❌ Data directory not found: {data_path}")
                return False
            
            # Get all files in data directory
            files = list(data_path.rglob("*"))
            document_files = [f for f in files if f.is_file() and f.suffix.lower() in {
                '.txt', '.md', '.pdf', '.docx', '.doc', '.rtf'
            }]
            
            if not document_files:
                logger.error(f"❌ No supported documents found in {data_path}")
                logger.info("💡 Supported formats: .txt, .md, .pdf, .docx, .doc, .rtf")
                return False
            
            logger.info(f"📂 Found {len(document_files)} documents to process")
            
            # Load documents with metadata
            reader = SimpleDirectoryReader(
                input_dir=str(data_path),
                recursive=True,
                required_exts=[".txt", ".md", ".pdf", ".docx", ".doc", ".rtf"],
                file_metadata=lambda filename: {"source": str(filename)}
            )
            
            self.documents = reader.load_data()
            
            if not self.documents:
                logger.error("❌ No documents were loaded")
                return False
            
            logger.info(f"✅ Loaded {len(self.documents)} documents")
            self._print_document_summary()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading documents: {e}")
            return False
    
    def _print_document_summary(self):
        """Print summary of loaded documents"""
        logger.info("📋 Document Summary:")
        
        total_chars = 0
        for i, doc in enumerate(self.documents[:5]):  # Show first 5
            content_preview = doc.text[:100].replace('\n', ' ')
            source = doc.metadata.get('source', 'unknown')
            char_count = len(doc.text)
            total_chars += char_count
            
            logger.info(f"  {i+1}. {Path(source).name} ({char_count:,} chars)")
            logger.info(f"     Preview: {content_preview}...")
        
        if len(self.documents) > 5:
            remaining = len(self.documents) - 5
            logger.info(f"  ... and {remaining} more documents")
        
        logger.info(f"📊 Total content: {total_chars:,} characters")
    
    @time_function("index_creation")
    async def create_index(self) -> bool:
        """Create vector index from loaded documents"""
        try:
            if not self.documents:
                logger.error("❌ No documents loaded")
                return False
            
            logger.info("🔨 Creating vector index...")
            logger.info("⏳ This may take a few minutes for large datasets...")
            
            # Create index with optimized node parser
            node_parser = SentenceWindowNodeParser.from_defaults(
                window_size=3,  # Number of sentences for context
                window_metadata_key="window",
                original_text_metadata_key="original_text"
            )
            
            # Create index
            self.index = VectorStoreIndex.from_documents(
                self.documents,
                transformations=[node_parser]
            )
            
            logger.info("✅ Vector index created successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating index: {e}")
            return False
    
    @time_function("index_persistence")
    def save_index(self) -> bool:
        """Save index to storage directory"""
        try:
            if not self.index:
                logger.error("❌ No index to save")
                return False
            
            # Clear existing storage
            storage_dir = config.storage_dir
            if storage_dir.exists():
                import shutil
                shutil.rmtree(storage_dir)
                logger.info("🗑️ Cleared existing storage")
            
            storage_dir.mkdir(exist_ok=True)
            
            # Save index
            self.index.storage_context.persist(persist_dir=str(storage_dir))
            
            logger.info(f"💾 Index saved to {storage_dir}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving index: {e}")
            return False
    
    async def test_search(self, test_query: str = "What is this about?") -> bool:
        """Test the created index with a sample query"""
        try:
            if not self.index:
                logger.error("❌ No index available for testing")
                return False
            
            logger.info(f"🧪 Testing search with query: '{test_query}'")
            
            # Create query engine
            query_engine = self.index.as_query_engine(
                use_async=True,
                similarity_top_k=3
            )
            
            # Perform test query
            response = await query_engine.aquery(test_query)
            
            logger.info("✅ Search test successful!")
            logger.info(f"📝 Response preview: {str(response)[:200]}...")
            
            # Test retriever
            retriever = self.index.as_retriever(similarity_top_k=3)
            nodes = await retriever.aretrieve(test_query)
            
            logger.info(f"🔍 Retrieved {len(nodes)} relevant nodes")
            for i, node in enumerate(nodes):
                score = getattr(node, 'score', 'N/A')
                preview = node.get_content()[:100].replace('\n', ' ')
                logger.info(f"  {i+1}. Score: {score:.3f} - {preview}...")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Search test failed: {e}")
            return False
    
    async def run_full_pipeline(self, data_path: Path = None, test_query: str = None) -> bool:
        """Run the complete ingestion pipeline"""
        logger.info("🚀 Starting data ingestion pipeline...")
        
        # Step 1: Load documents
        if not self.load_documents(data_path):
            return False
        
        # Step 2: Create index
        if not await self.create_index():
            return False
        
        # Step 3: Save index
        if not self.save_index():
            return False
        
        # Step 4: Test search
        if test_query:
            if not await self.test_search(test_query):
                logger.warning("⚠️ Search test failed, but index was created")
        
        logger.info("🎉 Data ingestion pipeline completed successfully!")
        logger.info(f"📁 Index saved to: {config.storage_dir}")
        logger.info("💡 You can now run the RAG agent")
        
        return True

def create_sample_data():
    """Create sample data files for testing"""
    data_dir = config.data_dir
    data_dir.mkdir(exist_ok=True)
    
    sample_files = {
        "faq.txt": """
Frequently Asked Questions

Q: What services do we offer?
A: We provide comprehensive AI voice solutions including automated customer service, appointment scheduling, and information lookup services.

Q: How do I get started?
A: Simply call our number and speak naturally. Our AI assistant will guide you through the process.

Q: Can I speak to a human agent?
A: Yes! Just ask to speak to a human agent or say "transfer me to a person" and we'll connect you immediately.

Q: What are your business hours?
A: Our AI assistant is available 24/7. Human agents are available Monday-Friday 9AM-5PM.

Q: Is my information secure?
A: Absolutely. We use enterprise-grade security and never store personal information without consent.
        """,
        
        "product_info.md": """
# Product Information

## AI Voice Assistant Features

### Core Capabilities
- Natural language processing
- Real-time voice interaction
- Knowledge base integration
- Human agent transfer
- Multi-language support

### Technical Specifications
- Sub-2-second response time
- 99.9% uptime guarantee
- Enterprise security standards
- API integration available

### Use Cases
- Customer support automation
- Appointment scheduling
- Information lookup
- Order processing
- Technical support

### Pricing
- Basic Plan: $99/month
- Professional Plan: $299/month
- Enterprise Plan: Custom pricing

Contact us for a demo or more information.
        """,
        
        "policies.txt": """
Company Policies

Privacy Policy:
We protect your personal information and never share it with third parties without consent.

Refund Policy:
30-day money-back guarantee for all plans.

Service Level Agreement:
- 99.9% uptime guarantee
- Sub-2-second response time
- 24/7 AI availability
- Human backup support

Data Retention:
Call logs are kept for 30 days for quality assurance.

Cancellation Policy:
Cancel anytime with 30-day notice.
        """
    }
    
    for filename, content in sample_files.items():
        file_path = data_dir / filename
        if not file_path.exists():
            file_path.write_text(content.strip())
            logger.info(f"📄 Created sample file: {filename}")
    
    logger.info(f"✅ Sample data created in {data_dir}")

async def main():
    """Main function for data ingestion"""
    parser = argparse.ArgumentParser(description="Ingest data for LiveKit RAG Agent")
    parser.add_argument("--data-path", type=Path, help="Path to data directory")
    parser.add_argument("--test-query", default="What services do you offer?", 
                       help="Test query for search validation")
    parser.add_argument("--create-sample", action="store_true", 
                       help="Create sample data files")
    
    args = parser.parse_args()
    
    try:
        # Validate configuration
        validate_config()
        
        # Create sample data if requested
        if args.create_sample:
            create_sample_data()
        
        # Run ingestion pipeline
        pipeline = DataIngestionPipeline()
        success = await pipeline.run_full_pipeline(
            data_path=args.data_path,
            test_query=args.test_query
        )
        
        if success:
            logger.info("🎉 Data ingestion completed successfully!")
            logger.info("🚀 You can now run: python agent.py")
        else:
            logger.error("❌ Data ingestion failed")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))