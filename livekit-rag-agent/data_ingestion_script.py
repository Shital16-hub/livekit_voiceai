"""
Data Ingestion Script - Add any data to the RAG system
Supports: JSON, TXT, PDF, CSV, Markdown files
"""
import asyncio
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional imports for different file types
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    import markdown
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False

class DataIngestion:
    """Ingest various data formats into the RAG system"""
    
    def __init__(self):
        self.supported_formats = ['.json', '.txt', '.md', '.csv']
        if PDF_AVAILABLE:
            self.supported_formats.append('.pdf')
    
    def process_file(self, file_path: Path) -> List[Dict[str, str]]:
        """Process a single file and return documents"""
        try:
            if file_path.suffix.lower() == '.json':
                return self._process_json(file_path)
            elif file_path.suffix.lower() == '.txt':
                return self._process_txt(file_path)
            elif file_path.suffix.lower() == '.md':
                return self._process_markdown(file_path)
            elif file_path.suffix.lower() == '.csv':
                return self._process_csv(file_path)
            elif file_path.suffix.lower() == '.pdf' and PDF_AVAILABLE:
                return self._process_pdf(file_path)
            else:
                logger.warning(f"⚠️ Unsupported file format: {file_path.suffix}")
                return []
                
        except Exception as e:
            logger.error(f"❌ Error processing {file_path}: {e}")
            return []
    
    def _process_json(self, file_path: Path) -> List[Dict[str, str]]:
        """Process JSON file"""
        documents = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            # Handle key-value pairs
            for key, value in data.items():
                documents.append({
                    "id": f"{file_path.stem}_{key}",
                    "content": str(value),
                    "source": str(file_path),
                    "category": key,
                    "type": "json_entry"
                })
        elif isinstance(data, list):
            # Handle list of items
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    # If item has 'content' or 'text' field, use that
                    content = item.get('content') or item.get('text') or str(item)
                    title = item.get('title') or item.get('name') or f"item_{i}"
                else:
                    content = str(item)
                    title = f"item_{i}"
                
                documents.append({
                    "id": f"{file_path.stem}_{i}",
                    "content": content,
                    "source": str(file_path),
                    "category": title,
                    "type": "json_list_item"
                })
        
        logger.info(f"✅ Processed JSON: {len(documents)} documents from {file_path}")
        return documents
    
    def _process_txt(self, file_path: Path) -> List[Dict[str, str]]:
        """Process text file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        if not content:
            return []
        
        # Split into chunks for large files
        chunks = self._chunk_text(content)
        documents = []
        
        for i, chunk in enumerate(chunks):
            documents.append({
                "id": f"{file_path.stem}_chunk_{i}",
                "content": chunk,
                "source": str(file_path),
                "category": file_path.stem,
                "type": "text_chunk"
            })
        
        logger.info(f"✅ Processed TXT: {len(documents)} chunks from {file_path}")
        return documents
    
    def _process_markdown(self, file_path: Path) -> List[Dict[str, str]]:
        """Process Markdown file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            md_content = f.read().strip()
        
        documents = []
        
        # Split by headers if possible
        sections = self._split_markdown_sections(md_content)
        
        for i, (title, content) in enumerate(sections):
            if content.strip():
                documents.append({
                    "id": f"{file_path.stem}_section_{i}",
                    "content": content.strip(),
                    "source": str(file_path),
                    "category": title or f"section_{i}",
                    "type": "markdown_section"
                })
        
        logger.info(f"✅ Processed Markdown: {len(documents)} sections from {file_path}")
        return documents
    
    def _process_csv(self, file_path: Path) -> List[Dict[str, str]]:
        """Process CSV file"""
        if not PANDAS_AVAILABLE:
            logger.warning("⚠️ Pandas not available for CSV processing")
            return []
        
        documents = []
        df = pd.read_csv(file_path)
        
        for i, row in df.iterrows():
            # Create content from all columns
            content_parts = []
            for col, val in row.items():
                if pd.notna(val):
                    content_parts.append(f"{col}: {val}")
            
            content = "; ".join(content_parts)
            
            documents.append({
                "id": f"{file_path.stem}_row_{i}",
                "content": content,
                "source": str(file_path),
                "category": "csv_row",
                "type": "csv_entry"
            })
        
        logger.info(f"✅ Processed CSV: {len(documents)} rows from {file_path}")
        return documents
    
    def _process_pdf(self, file_path: Path) -> List[Dict[str, str]]:
        """Process PDF file"""
        if not PDF_AVAILABLE:
            logger.warning("⚠️ PyPDF2 not available for PDF processing")
            return []
        
        documents = []
        
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text().strip()
                if text:
                    # Split page into chunks if too long
                    chunks = self._chunk_text(text)
                    for chunk_num, chunk in enumerate(chunks):
                        documents.append({
                            "id": f"{file_path.stem}_page_{page_num}_chunk_{chunk_num}",
                            "content": chunk,
                            "source": str(file_path),
                            "category": f"page_{page_num}",
                            "type": "pdf_chunk"
                        })
        
        logger.info(f"✅ Processed PDF: {len(documents)} chunks from {file_path}")
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
            
        return [c for c in chunks if c.strip()]
    
    def _split_markdown_sections(self, content: str) -> List[tuple]:
        """Split markdown into sections by headers"""
        lines = content.split('\n')
        sections = []
        current_title = None
        current_content = []
        
        for line in lines:
            if line.startswith('#'):
                # Save previous section
                if current_content:
                    sections.append((current_title, '\n'.join(current_content)))
                
                # Start new section
                current_title = line.lstrip('#').strip()
                current_content = []
            else:
                current_content.append(line)
        
        # Save last section
        if current_content:
            sections.append((current_title, '\n'.join(current_content)))
        
        return sections
    
    def process_directory(self, dir_path: Path, recursive: bool = True) -> List[Dict[str, str]]:
        """Process all supported files in a directory"""
        all_documents = []
        
        if recursive:
            pattern = "**/*"
        else:
            pattern = "*"
        
        for file_path in dir_path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                documents = self.process_file(file_path)
                all_documents.extend(documents)
        
        logger.info(f"✅ Processed directory: {len(all_documents)} total documents")
        return all_documents

async def main():
    """Main ingestion function"""
    parser = argparse.ArgumentParser(description="Ingest data into RAG system")
    parser.add_argument("--file", type=str, help="Single file to process")
    parser.add_argument("--directory", type=str, help="Directory to process")
    parser.add_argument("--recursive", action="store_true", help="Process directory recursively")
    parser.add_argument("--output", type=str, default="data/processed_data.json", help="Output file")
    
    args = parser.parse_args()
    
    ingestion = DataIngestion()
    all_documents = []
    
    if args.file:
        file_path = Path(args.file)
        if file_path.exists():
            documents = ingestion.process_file(file_path)
            all_documents.extend(documents)
        else:
            logger.error(f"❌ File not found: {file_path}")
            return
    
    if args.directory:
        dir_path = Path(args.directory)
        if dir_path.exists():
            documents = ingestion.process_directory(dir_path, args.recursive)
            all_documents.extend(documents)
        else:
            logger.error(f"❌ Directory not found: {dir_path}")
            return
    
    if not all_documents:
        logger.error("❌ No documents processed")
        return
    
    # Save processed documents
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to simple format for RAG system
    rag_documents = []
    for doc in all_documents:
        rag_documents.append({
            "id": doc["id"],
            "content": doc["content"],
            "source": doc["source"],
            "category": doc["category"]
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(rag_documents, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ Saved {len(rag_documents)} documents to {output_path}")
    
    # Also create the simple knowledge format
    simple_knowledge = {}
    for doc in all_documents:
        category = doc["category"].lower().replace(" ", "_")
        if category not in simple_knowledge:
            simple_knowledge[category] = doc["content"]
    
    simple_path = Path("data/simple_knowledge.json")
    simple_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(simple_path, 'w', encoding='utf-8') as f:
        json.dump(simple_knowledge, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ Also saved simple format to {simple_path}")

if __name__ == "__main__":
    asyncio.run(main())