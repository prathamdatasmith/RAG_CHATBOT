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
        """Process a single PDF file through the complete multimodal pipeline"""
        try:
            print(f"Starting multimodal processing of: {pdf_path}")
            
            # Step 1: Extract text chunks and images from PDF
            print("Step 1: Extracting text and images from PDF...")
            text_chunks, images_info = self.pdf_processor.process_pdf(pdf_path)
            print(f"Created {len(text_chunks)} text chunks and extracted {len(images_info)} images")
            
            # Step 2: Generate embeddings for text chunks
            print("Step 2: Generating text embeddings...")
            texts = [chunk['text'] for chunk in text_chunks]
            text_embeddings = self.embedding_service.embed_texts(texts)
            print(f"Generated {len(text_embeddings)} text embeddings")
            
            # Step 3: Generate embeddings for image captions
            print("Step 3: Generating image caption embeddings...")
            image_captions = [img['caption'] for img in images_info]
            if image_captions:
                caption_embeddings = self.embedding_service.embed_texts(image_captions)
                print(f"Generated {len(caption_embeddings)} caption embeddings")
            else:
                caption_embeddings = []
            
            # Step 4: Ensure collections exist
            print("Step 4: Setting up Qdrant collections...")
            await self.qdrant_service.create_collections_if_not_exist()
            
            # Step 5: Store text documents in Qdrant
            print("Step 5: Storing text documents in Qdrant...")
            await self.qdrant_service.add_documents(text_chunks, text_embeddings)
            
            # Step 6: Store image information in Qdrant
            if images_info and caption_embeddings:
                print("Step 6: Storing image information in Qdrant...")
                await self.qdrant_service.add_images(images_info, caption_embeddings)
            
            filename = os.path.basename(pdf_path)
            print(f"Successfully processed: {filename}")
            
            return {
                'success': True,
                'filename': filename,
                'chunks_count': len(text_chunks),
                'images_count': len(images_info),
                'message': f'Successfully processed {filename} with {len(text_chunks)} text chunks and {len(images_info)} images'
            }
            
        except Exception as e:
            error_msg = f"Error processing {pdf_path}: {str(e)}"
            print(error_msg)
            return {
                'success': False,
                'filename': os.path.basename(pdf_path) if pdf_path else 'unknown',
                'error': error_msg
            }
    
    def _is_visual_query(self, query: str) -> bool:
        """Determine if a query is asking for visual content"""
        visual_keywords = [
            'figure', 'fig', 'diagram', 'chart', 'graph', 'image', 'picture', 
            'illustration', 'screenshot', 'photo', 'table', 'show me', 
            'display', 'visual', 'drawing', 'plot', 'map'
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in visual_keywords)
    
    async def search_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Enhanced search that handles both text and visual queries"""
        try:
            print(f"Searching for: {query}")
            
            # Generate query embedding
            query_embedding = self.embedding_service.embed_single_text(query)
            
            # Check if this is a visual query
            if self._is_visual_query(query):
                print("Detected visual query - searching both text and images")
                # Search both text and images
                multimodal_results = await self.qdrant_service.search_multimodal(
                    query_embedding, 
                    text_limit=top_k,
                    image_limit=5
                )
                
                # Combine and rank results
                combined_results = []
                
                # Add text results
                for result in multimodal_results['text']:
                    combined_results.append(result)
                
                # Add image results with higher priority for visual queries
                for result in multimodal_results['images']:
                    # Boost image scores for visual queries
                    result['score'] = min(result['score'] * 1.2, 1.0)
                    combined_results.append(result)
                
                # Sort by score
                combined_results.sort(key=lambda x: x['score'], reverse=True)
                return combined_results[:top_k + 5]  # Return more for visual queries
                
            else:
                print("Text-based query - searching text documents")
                # Regular text search
                results = await self.qdrant_service.search_text_similar(query_embedding, top_k)
                return results
            
        except Exception as e:
            print(f"Error searching documents: {str(e)}")
            return []
    
    async def search_images_only(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search specifically for images"""
        try:
            print(f"Searching images for: {query}")
            query_embedding = self.embedding_service.embed_single_text(query)
            results = await self.qdrant_service.search_images_similar(query_embedding, top_k)
            return results
        except Exception as e:
            print(f"Error searching images: {str(e)}")
            return []
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics for both text and images"""
        return await self.qdrant_service.get_collection_info()
    
    def cleanup(self):
        """Clean up resources"""
        self.qdrant_service.close()

# Example usage and testing
async def main():
    """Example usage of the enhanced multimodal ingestion pipeline"""
    pipeline = IngestionPipeline()
    
    try:
        # Get collection stats
        stats = await pipeline.get_collection_stats()
        print(f"Collection stats: {stats}")
        
        # Example: Search for both text and images
        # results = await pipeline.search_documents("show me figure 8-6")
        # print(f"Multimodal search results: {len(results)}")
        
        print("Enhanced Multimodal Pipeline initialized successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        pipeline.cleanup()

if __name__ == "__main__":
    asyncio.run(main())