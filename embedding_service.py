from fastembed import TextEmbedding
from typing import List
import numpy as np
from config import Config

class EmbeddingService:
    def __init__(self):
        """Initialize FastEmbed embedding model"""
        try:
            self.embedding_model = TextEmbedding(Config.EMBEDDING_MODEL)
            print(f"Initialized embedding model: {Config.EMBEDDING_MODEL}")
        except Exception as e:
            raise Exception(f"Failed to initialize embedding model: {str(e)}")
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        try:
            # FastEmbed returns generator, so we convert to list
            embeddings = list(self.embedding_model.embed(texts))
            
            # Convert numpy arrays to lists if needed
            embeddings_list = []
            for embedding in embeddings:
                if isinstance(embedding, np.ndarray):
                    embeddings_list.append(embedding.tolist())
                else:
                    embeddings_list.append(list(embedding))
            
            return embeddings_list
            
        except Exception as e:
            raise Exception(f"Error generating embeddings: {str(e)}")
    
    def embed_single_text(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        embeddings = self.embed_texts([text])
        return embeddings[0] if embeddings else []
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        return Config.VECTOR_SIZE