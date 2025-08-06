import fitz  # PyMuPDF
import re
from typing import List, Dict, Any
from config import Config

class PDFProcessor:
    def __init__(self):
        self.chunk_size = Config.CHUNK_SIZE
        self.chunk_overlap = Config.CHUNK_OVERLAP
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file using PyMuPDF"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page.get_text()
            
            doc.close()
            return self.clean_text(text)
        
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    def clean_text(self, text: str) -> str:
        """Clean text while preserving all types of formatting"""
        # First preserve code blocks and special formatting
        text = re.sub(r'```[\s\S]*?```', lambda m: m.group().replace('\n', '[NEWLINE]'), text)
        text = re.sub(r'`[^`]*`', lambda m: m.group().replace(' ', '[SPACE]'), text)
        
        # Preserve tables
        text = re.sub(r'\|\s*[-|]+\s*\|', lambda m: m.group().replace('\n', '[NEWLINE]'), text)
        text = re.sub(r'\|.*\|', lambda m: m.group().replace('\n', '[NEWLINE]'), text)
        
        # Preserve lists
        text = re.sub(r'^\s*[-*]\s', '• ', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\.\s', lambda m: f"{m.group().strip()} ", text, flags=re.MULTILINE)
        
        # Preserve code syntax
        text = re.sub(r'(\w+)\s+(\w+)\s*\(', r'\1 = \2(', text)
        text = re.sub(r'([a-zA-Z_]\w*)\s+([\'"].*?[\'"])', r'\1=\2', text)
        text = re.sub(r'(?<=[^=])=(?=[^=])', ' = ', text)
        
        # Remove excessive whitespace but preserve structure
        text = '\n'.join(line.rstrip() for line in text.split('\n'))
        
        # Restore special markers
        text = text.replace('[NEWLINE]', '\n').replace('[SPACE]', ' ')
        
        return text.strip()
    
    def chunk_text(self, text: str, filename: str) -> List[Dict[str, Any]]:
        """Split text into chunks while preserving formatting"""
        chunks = []
        
        # Split by natural boundaries first
        sections = re.split(r'\n\s*\n', text)
        
        current_chunk = []
        current_size = 0
        
        for section in sections:
            section_words = section.split()
            section_size = len(section_words)
            
            if current_size + section_size <= self.chunk_size:
                current_chunk.append(section)
                current_size += section_size
            else:
                if current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    chunks.append({
                        'text': chunk_text,
                        'metadata': {
                            'filename': filename,
                            'chunk_id': len(chunks),
                            'word_count': current_size,
                            'has_code': bool(re.search(r'```|\b(def|class|import|from)\b', chunk_text)),
                            'has_list': bool(re.search(r'^\s*[-*•]|\d+\.', chunk_text, re.M)),
                            'has_table': bool(re.search(r'\|.*\|', chunk_text))
                        }
                    })
                current_chunk = [section]
                current_size = section_size
        
        # Add the last chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'metadata': {
                    'filename': filename,
                    'chunk_id': len(chunks),
                    'word_count': current_size,
                    'has_code': bool(re.search(r'```|\b(def|class|import|from)\b', chunk_text)),
                    'has_list': bool(re.search(r'^\s*[-*•]|\d+\.', chunk_text, re.M)),
                    'has_table': bool(re.search(r'\|.*\|', chunk_text))
                }
            })
        
        return chunks
    
    def process_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Complete PDF processing pipeline"""
        filename = pdf_path.split('/')[-1] if '/' in pdf_path else pdf_path.split('\\')[-1]
        
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        
        if not text.strip():
            raise Exception("No text could be extracted from the PDF")
        
        # Create chunks
        chunks = self.chunk_text(text, filename)
        
        return chunks