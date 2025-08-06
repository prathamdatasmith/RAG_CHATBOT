import asyncio
import os
from ingestion_pipeline import IngestionPipeline
from rag_service import RAGService

async def test_pipeline():
    """Test the complete pipeline"""
    print("🧪 Testing RAG Pipeline...")
    
    # Initialize services
    pipeline = IngestionPipeline()
    rag_service = RAGService()
    
    try:
        # Test 1: Check Qdrant connection
        print("\n1. Testing Qdrant connection...")
        stats = await pipeline.get_collection_stats()
        print(f"✅ Connection successful: {stats}")
        
        # Test 2: Test embedding service
        print("\n2. Testing embedding service...")
        test_text = "This is a test sentence for embedding."
        embedding = pipeline.embedding_service.embed_single_text(test_text)
        print(f"✅ Embedding generated: dimension = {len(embedding)}")
        
        # Test 3: Test search (if collection has data)
        print("\n3. Testing search functionality...")
        results = await rag_service.generate_answer("What is this about?")
        print(f"✅ Search completed: {results['answer'][:100]}...")
        
        print("\n🎉 All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
    
    finally:
        pipeline.cleanup()
        rag_service.cleanup()

if __name__ == "__main__":
    asyncio.run(test_pipeline())