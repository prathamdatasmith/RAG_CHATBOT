import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "chatbot_docs")
    
    # FastEmbed settings
    EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
    VECTOR_SIZE = 384  # This model produces 384-dimensional vectors
    
    # Chunking settings for formatted content
    CHUNK_SIZE = 2000  # Smaller chunks to preserve formatting
    CHUNK_OVERLAP = 400  # Overlap for context
    
    # Retrieval settings
    TOP_K = 25  # Get more chunks for better context
    SCORE_THRESHOLD = 0  # Slightly higher threshold for relevance
    MAX_CONTEXT_CHUNKS = 12  # Balance between context and response quality
    
    # Gemini API settings
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")