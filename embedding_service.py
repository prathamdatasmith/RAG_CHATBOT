from google import genai
from google.genai import types
from typing import List
import numpy as np
from config import Config

class EmbeddingService:
    def __init__(self):
        """Initialize Gemini embedding client"""
        try:
            self.client = genai.Client(api_key=Config.GEMINI_API_KEY)
            print(f"Initialized Gemini embedding model: {Config.EMBEDDING_MODEL}")
        except Exception as e:
            raise Exception(f"Failed to initialize Gemini embedding client: {str(e)}")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        try:
            result = self.client.models.embed_content(
                model=Config.EMBEDDING_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type=Config.EMBEDDING_TASK_TYPE,
                    output_dimensionality=Config.EMBEDDING_OUTPUT_DIM
                )
            )
            
            embeddings = []
            for embedding_obj in result.embeddings:
                embedding = list(embedding_obj.values)
                if Config.NORMALIZE_EMBEDDINGS:
                    embedding = self._normalize_embedding(embedding)
                embeddings.append(embedding)
            return embeddings
            
        except Exception as e:
            raise Exception(f"Error generating embeddings: {str(e)}")

    def _normalize_embedding(self, embedding: List[float]) -> List[float]:
        """Normalize embedding vector to unit length"""
        arr = np.array(embedding)
        norm = np.linalg.norm(arr)
        if norm > 0:
            return (arr / norm).tolist()
        return arr.tolist()