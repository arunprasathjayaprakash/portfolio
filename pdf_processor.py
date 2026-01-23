"""
PDF processing utilities for chunking documents
"""
from pypdf import PdfReader
from typing import List, Dict
import uuid
import hashlib


class PDFProcessor:
    """Handles PDF file processing and chunking"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize PDF processor
        
        Args:
            chunk_size: Number of characters per chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text content from PDF file"""
        try:
            reader = PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {e}")
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Split text into chunks with overlap
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text or len(text.strip()) == 0:
            return []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            # Calculate end position
            end = start + self.chunk_size
            
            # Extract chunk
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunk_metadata = {
                    "start_char": start,
                    "end_char": min(end, text_length),
                    **(metadata or {})
                }
                
                chunks.append({
                    "text": chunk_text,
                    "metadata": chunk_metadata
                })
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            
            # Prevent infinite loop
            if start >= text_length:
                break
        
        return chunks
    
    def process_pdf(self, pdf_file, filename: str) -> Dict:
        """
        Process PDF file and return chunks
        
        Args:
            pdf_file: PDF file object
            filename: Name of the PDF file
            
        Returns:
            Dictionary with document_id, text, chunks, and metadata
        """
        # Generate document ID
        document_id = str(uuid.uuid4())
        
        # Extract text
        text = self.extract_text_from_pdf(pdf_file)
        
        # Create metadata
        metadata = {
            "filename": filename,
            "total_chars": len(text)
        }
        
        # Chunk text
        chunks = self.chunk_text(text, metadata)
        
        return {
            "document_id": document_id,
            "filename": filename,
            "text": text,
            "chunks": chunks,
            "num_chunks": len(chunks),
            "file_size": len(text)
        }
