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
        self.collection_name = Config.COLLECTION_NAME
        
    async def create_collection_if_not_exists(self):
        """Create collection if it doesn't exist"""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                # Create collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=Config.VECTOR_SIZE,
                        distance=Distance.COSINE
                    )
                )
                print(f"Created collection: {self.collection_name}")
            else:
                print(f"Collection {self.collection_name} already exists")
                
        except Exception as e:
            raise Exception(f"Error creating collection: {str(e)}")
    
    async def add_documents(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        """Add documents with embeddings to Qdrant"""
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
                        'word_count': chunk['metadata']['word_count']
                    }
                )
                points.append(point)
            
            # Batch insert points
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
            
            print(f"Successfully added {len(points)} documents to Qdrant")
            
        except Exception as e:
            raise Exception(f"Error adding documents to Qdrant: {str(e)}")
    
    async def search_similar(self, query_embedding: List[float], limit: int = Config.TOP_K) -> List[Dict[str, Any]]:
        """Comprehensive search - gets results from ANYWHERE in large documents"""
        try:
            all_results = []
            
            # Search 1: Try with configured threshold
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=Config.SCORE_THRESHOLD
            )
            all_results.extend(search_results)
            
            # Search 2: If few results, try with lower threshold
            if len(search_results) < limit // 2:
                search_results2 = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    limit=limit * 2,
                    score_threshold=0.05  # Very low threshold
                )
                all_results.extend(search_results2)
            
            # Search 3: If still few results, get everything with minimal threshold
            if len(all_results) < limit:
                search_results3 = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    limit=limit * 3,
                    score_threshold=0.01  # Almost no threshold
                )
                all_results.extend(search_results3)
            
            # Convert to our format and remove duplicates
            seen_ids = set()
            results = []
            
            for result in all_results:
                result_id = f"{result.payload['filename']}_{result.payload['chunk_id']}"
                if result_id not in seen_ids:
                    seen_ids.add(result_id)
                    results.append({
                        'text': result.payload['text'],
                        'filename': result.payload['filename'],
                        'chunk_id': result.payload['chunk_id'],
                        'score': result.score,
                        'word_count': result.payload.get('word_count', 0)
                    })
            
            # Sort by score and return
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:limit]
            
        except Exception as e:
            raise Exception(f"Error in comprehensive search: {str(e)}")
    
    async def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                'vectors_count': info.vectors_count,
                'points_count': info.points_count,
                'status': info.status
            }
        except Exception as e:
            return {'error': str(e)}
    
    async def delete_collection(self):
        """Delete the collection"""
        try:
            self.client.delete_collection(self.collection_name)
            print(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            print(f"Error deleting collection: {str(e)}")
    
    def close(self):
        """Close the client connection"""
        if hasattr(self.client, 'close'):
            self.client.close()