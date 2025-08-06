import asyncio
import os
from typing import List, Dict, Any
from pdf_processor import PDFProcessor
from embedding_service import EmbeddingService
from qdrant_service import QdrantService

class IngestionPipeline:
    def __init__(self):
        """Initialize the ingestion pipeline"""
        self.pdf_processor = PDFProcessor()
        self.embedding_service = EmbeddingService()
        self.qdrant_service = QdrantService()
    
    async def process_pdf_file(self, pdf_path: str) -> Dict[str, Any]:
        """Process a single PDF file through the complete pipeline"""
        try:
            print(f"Starting processing of: {pdf_path}")
            
            # Step 1: Extract and chunk text from PDF
            print("Step 1: Extracting text from PDF...")
            chunks = self.pdf_processor.process_pdf(pdf_path)
            print(f"Created {len(chunks)} chunks")
            
            # Step 2: Generate embeddings
            print("Step 2: Generating embeddings...")
            texts = [chunk['text'] for chunk in chunks]
            embeddings = self.embedding_service.embed_texts(texts)
            print(f"Generated {len(embeddings)} embeddings")
            
            # Step 3: Ensure collection exists
            print("Step 3: Setting up Qdrant collection...")
            await self.qdrant_service.create_collection_if_not_exists()
            
            # Step 4: Store in Qdrant
            print("Step 4: Storing documents in Qdrant...")
            await self.qdrant_service.add_documents(chunks, embeddings)
            
            filename = os.path.basename(pdf_path)
            print(f"Successfully processed: {filename}")
            
            return {
                'success': True,
                'filename': filename,
                'chunks_count': len(chunks),
                'message': f'Successfully processed {filename} with {len(chunks)} chunks'
            }
            
        except Exception as e:
            error_msg = f"Error processing {pdf_path}: {str(e)}"
            print(error_msg)
            return {
                'success': False,
                'filename': os.path.basename(pdf_path) if pdf_path else 'unknown',
                'error': error_msg
            }
    
    async def process_multiple_pdfs(self, pdf_paths: List[str]) -> List[Dict[str, Any]]:
        """Process multiple PDF files"""
        results = []
        
        for pdf_path in pdf_paths:
            result = await self.process_pdf_file(pdf_path)
            results.append(result)
        
        return results
    
    async def search_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents based on query"""
        try:
            print(f"Searching for: {query}")
            
            # Generate query embedding
            query_embedding = self.embedding_service.embed_single_text(query)
            
            # Search in Qdrant
            results = await self.qdrant_service.search_similar(query_embedding, top_k)
            
            return results
            
        except Exception as e:
            print(f"Error searching documents: {str(e)}")
            return []
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        return await self.qdrant_service.get_collection_info()
    
    def cleanup(self):
        """Clean up resources"""
        self.qdrant_service.close()

# Example usage and testing
async def main():
    """Example usage of the ingestion pipeline"""
    pipeline = IngestionPipeline()
    
    try:
        # Example: Process a PDF file
        # pdf_path = "path/to/your/document.pdf"
        # result = await pipeline.process_pdf_file(pdf_path)
        # print(result)
        
        # Example: Search documents
        # results = await pipeline.search_documents("your search query")
        # print(results)
        
        # Get collection stats
        stats = await pipeline.get_collection_stats()
        print(f"Collection stats: {stats}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        pipeline.cleanup()

if __name__ == "__main__":
    asyncio.run(main())