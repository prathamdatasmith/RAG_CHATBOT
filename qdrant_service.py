import asyncio
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import uuid
from config import Config

class QdrantService:
    def __init__(self):
        """Initialize Qdrant client"""
        self.client = QdrantClient(
            url=Config.QDRANT_URL,
            api_key=Config.QDRANT_API_KEY,
            timeout=60
        )
        self.text_collection_name = Config.TEXT_COLLECTION_NAME
        self.image_collection_name = Config.IMAGES_COLLECTION_NAME
        
    async def create_collections_if_not_exist(self):
        """Create both text and image collections if they don't exist"""
        try:
            # Check existing collections
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            # Create text collection
            if self.text_collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.text_collection_name,
                    vectors_config=VectorParams(
                        size=Config.VECTOR_SIZE,
                        distance=Distance.COSINE
                    )
                )
                print(f"Created text collection: {self.text_collection_name}")
            
            # Create image collection
            if self.image_collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.image_collection_name,
                    vectors_config=VectorParams(
                        size=Config.VECTOR_SIZE,
                        distance=Distance.COSINE
                    )
                )
                print(f"Created image collection: {self.image_collection_name}")
                
        except Exception as e:
            raise Exception(f"Error creating collections: {str(e)}")
    
    async def create_collection_if_not_exists(self):
        """Backward compatibility - creates text collection only"""
        await self.create_collections_if_not_exist()
    
    async def add_documents(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        """Add text documents with embeddings to Qdrant"""
        try:
            points = []
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        'text': chunk['text'],
                        'filename': chunk['metadata']['filename'],
                        'chunk_id': chunk['metadata']['chunk_id'],
                        'word_count': chunk['metadata']['word_count'],
                        'content_type': 'text'
                    }
                )
                points.append(point)
            
            # Batch insert points
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.text_collection_name,
                    points=batch
                )
            
            print(f"Successfully added {len(points)} text documents to Qdrant")
            
        except Exception as e:
            raise Exception(f"Error adding documents to Qdrant: {str(e)}")
    
    async def add_images(self, images: List[Dict[str, Any]], caption_embeddings: List[List[float]]):
        """Add image information with caption embeddings to Qdrant"""
        try:
            points = []
            
            for i, (image_info, embedding) in enumerate(zip(images, caption_embeddings)):
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        'image_path': image_info['image_path'],
                        'caption': image_info['caption'],
                        'page_number': image_info['page_number'],
                        'pdf_filename': image_info['pdf_filename']
                    }
                )
                points.append(point)
            
            # Batch insert points
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.image_collection_name,
                    points=batch
                )
            
            print(f"Successfully added {len(points)} images to Qdrant")
            
        except Exception as e:
            raise Exception(f"Error adding images to Qdrant: {str(e)}")
    
    async def search_text_similar(self, query_embedding: List[float], limit: int = Config.TOP_K) -> List[Dict[str, Any]]:
        """Search for similar text documents"""
        try:
            search_results = self.client.search(
                collection_name=self.text_collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=Config.SCORE_THRESHOLD
            )
            
            results = []
            for result in search_results:
                results.append({
                    'text': result.payload['text'],
                    'filename': result.payload['filename'],
                    'chunk_id': result.payload['chunk_id'],
                    'score': result.score,
                    'word_count': result.payload.get('word_count', 0),
                    'content_type': 'text'
                })
            
            return results
            
        except Exception as e:
            raise Exception(f"Error searching text documents: {str(e)}")
    
    async def search_images_similar(self, query_embedding: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        """Search for similar images based on caption embeddings"""
        try:
            search_results = self.client.search(
                collection_name=self.image_collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=0.3  # Lower threshold for images
            )
            
            results = []
            for result in search_results:
                results.append({
                    'image_path': result.payload['image_path'],
                    'image_filename': result.payload['image_filename'],
                    'caption': result.payload['caption'],
                    'page_number': result.payload['page_number'],
                    'pdf_filename': result.payload['pdf_filename'],
                    'score': result.score,
                    'image_id': result.payload['image_id'],
                    'width': result.payload['width'],
                    'height': result.payload['height'],
                    'content_type': 'image'
                })
            
            return results
            
        except Exception as e:
            raise Exception(f"Error searching images: {str(e)}")
    
    async def search_similar(self, query_embedding: List[float], limit: int = Config.TOP_K) -> List[Dict[str, Any]]:
        """Backward compatibility - searches only text documents"""
        return await self.search_text_similar(query_embedding, limit)
    
    async def search_multimodal(self, query_embedding: List[float], text_limit: int = None, image_limit: int = None) -> Dict[str, List[Dict[str, Any]]]:
        """Search both text and images"""
        try:
            if text_limit is None:
                text_limit = Config.TOP_K
            if image_limit is None:
                image_limit = 5
            
            # Search text and images concurrently
            text_results = await self.search_text_similar(query_embedding, text_limit)
            image_results = await self.search_images_similar(query_embedding, image_limit)
            
            return {
                'text': text_results,
                'images': image_results
            }
            
        except Exception as e:
            raise Exception(f"Error in multimodal search: {str(e)}")
    
    async def get_collection_info(self) -> Dict[str, Any]:
        """Get information about both collections"""
        try:
            text_info = self.client.get_collection(self.text_collection_name)
            
            try:
                image_info = self.client.get_collection(self.image_collection_name)
                image_stats = {
                    'vectors_count': image_info.vectors_count,
                    'points_count': image_info.points_count,
                    'status': image_info.status
                }
            except:
                image_stats = {'vectors_count': 0, 'points_count': 0, 'status': 'not_found'}
            
            return {
                'text_collection': {
                    'vectors_count': text_info.vectors_count,
                    'points_count': text_info.points_count,
                    'status': text_info.status
                },
                'image_collection': image_stats
            }
        except Exception as e:
            return {'error': str(e)}
    
    async def delete_collection(self):
        """Delete both collections"""
        try:
            self.client.delete_collection(self.text_collection_name)
            print(f"Deleted text collection: {self.text_collection_name}")
        except Exception as e:
            print(f"Error deleting text collection: {str(e)}")
        
        try:
            self.client.delete_collection(self.image_collection_name)
            print(f"Deleted image collection: {self.image_collection_name}")
        except Exception as e:
            print(f"Error deleting image collection: {str(e)}")
    
    def close(self):
        """Close the client connection"""
        if hasattr(self.client, 'close'):
            self.client.close()