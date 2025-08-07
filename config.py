import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Qdrant settings
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # Can be None for local
    TEXT_COLLECTION_NAME = os.getenv("TEXT_COLLECTION_NAME", "chatbot_docs")
    IMAGES_COLLECTION_NAME = os.getenv("IMAGES_COLLECTION_NAME", "chatbot_docs_images") 
    
    # Gemini Embedding settings
    EMBEDDING_MODEL = "gemini-embedding-001"
    VECTOR_SIZE = 1536  # Recommended dimension for Gemini
    EMBEDDING_OUTPUT_DIM = 1536  # Match vector size
    EMBEDDING_TASK_TYPE = "RETRIEVAL_DOCUMENT"  # For document search
    NORMALIZE_EMBEDDINGS = True  # For non-3072 dimensions
    
    # Chunking settings for formatted content
    CHUNK_SIZE = 2000  # Smaller chunks to preserve formatting
    CHUNK_OVERLAP = 400  # Overlap for context
    
    # Retrieval settings
    TOP_K = 25  # Get more chunks for better context
    SCORE_THRESHOLD = 0  # Slightly higher threshold for relevance
    MAX_CONTEXT_CHUNKS = 12  # Balance between context and response quality
    
    # Multimodal settings
    IMAGES_DIR = "extracted_images"
    MAX_IMAGES_PER_QUERY = 5
    IMAGE_SCORE_BOOST = 1.2  # Boost image scores for visual queries
    
    # Gemini API settings
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")