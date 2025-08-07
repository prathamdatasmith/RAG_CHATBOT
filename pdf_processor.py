import fitz  # PyMuPDF
import re
import os
from typing import List, Dict, Any, Tuple
from config import Config
from PIL import Image
import uuid
import base64

class PDFProcessor:
    def __init__(self):
        self.chunk_size = Config.CHUNK_SIZE
        self.chunk_overlap = Config.CHUNK_OVERLAP
        self.images_dir = "extracted_images"
        os.makedirs(self.images_dir, exist_ok=True)
    
    def extract_text_and_images_from_pdf(self, pdf_path: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Extract both text and images from PDF file using PyMuPDF"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            images_info = []
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                
                # Extract text
                page_text = page.get_text()
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page_text
                
                # Extract images
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    # Get image data
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    # Generate unique filename
                    filename_base = os.path.splitext(os.path.basename(pdf_path))[0]
                    image_filename = f"{filename_base}_page_{page_num + 1}_img_{img_index + 1}.{image_ext}"
                    image_path = os.path.join(self.images_dir, image_filename)
                    
                    # Save image
                    with open(image_path, "wb") as img_file:
                        img_file.write(image_bytes)
                    
                    # Try to find associated caption/text near the image
                    caption = self._extract_image_caption(page_text, img_index, page_num + 1)
                    
                    # Store image info
                    images_info.append({
                        'image_path': image_path,
                        'image_filename': image_filename,
                        'page_number': page_num + 1,
                        'image_index': img_index + 1,
                        'caption': caption,
                        'pdf_filename': os.path.basename(pdf_path),
                        'image_id': str(uuid.uuid4()),
                        'width': base_image["width"],
                        'height': base_image["height"]
                    })
            
            doc.close()
            return self.clean_text(text), images_info
        
        except Exception as e:
            raise Exception(f"Error extracting text and images from PDF: {str(e)}")
    
    def _extract_image_caption(self, page_text: str, img_index: int, page_num: int) -> str:
        """Try to extract caption for an image"""
        # Common caption patterns
        caption_patterns = [
            r'Figure\s+\d+[.-]\d*[:\s]([^.\n]+)',
            r'Fig\s+\d+[.-]\d*[:\s]([^.\n]+)', 
            r'Image\s+\d+[:\s]([^.\n]+)',
            r'Diagram\s+\d+[:\s]([^.\n]+)',
            r'Chart\s+\d+[:\s]([^.\n]+)',
            r'Table\s+\d+[:\s]([^.\n]+)'
        ]
        
        for pattern in caption_patterns:
            matches = re.findall(pattern, page_text, re.IGNORECASE)
            if matches and len(matches) > img_index:
                return matches[img_index].strip()
            elif matches:
                return matches[0].strip()
        
        # Fallback: generic description
        return f"Image {img_index + 1} from page {page_num}"
    
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
    
    def process_pdf(self, pdf_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Complete PDF processing pipeline - returns both text chunks and images"""
        filename = pdf_path.split('/')[-1] if '/' in pdf_path else pdf_path.split('\\')[-1]
        
        # Extract text and images
        text, images_info = self.extract_text_and_images_from_pdf(pdf_path)
        
        if not text.strip():
            raise Exception("No text could be extracted from the PDF")
        
        # Create text chunks
        text_chunks = self.chunk_text(text, filename)
        
        return text_chunks, images_info