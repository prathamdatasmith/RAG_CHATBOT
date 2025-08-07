import asyncio
import os
from typing import List, Dict, Any, Optional
from ingestion_pipeline import IngestionPipeline
from google import genai
from google.genai import types
from config import Config

class RAGService:
    def __init__(self):
        """Initialize RAG service"""
        self.pipeline = IngestionPipeline()
        # Initialize Gemini client
        self.gemini_client = genai.Client(
            api_key=Config.GEMINI_API_KEY,
        )
    
    def _is_visual_query(self, query: str) -> bool:
        """Determine if a query is asking for visual content"""
        visual_keywords = [
            'figure', 'fig', 'diagram', 'chart', 'graph', 'image', 'picture', 
            'illustration', 'screenshot', 'photo', 'table', 'show me', 
            'display', 'visual', 'drawing', 'plot', 'map', 'screenshot'
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in visual_keywords)
    
    def _is_greeting(self, text: str) -> bool:
        """Check if the text is a greeting"""
        greetings = {'hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening'}
        return text.lower().strip() in greetings
    
    async def generate_answer(self, question: str, top_k: int = None) -> Dict[str, Any]:
        """Generate multimodal answer based on retrieved documents and images"""
        try:
            # Handle greetings and simple queries
            if self._is_greeting(question):
                return {
                    'answer': 'Hello! I am an AI assistant. How can I help you today?',
                    'sources': [],
                    'confidence': 1.0
                }

            if top_k is None:
                top_k = Config.TOP_K
                
            # Step 1: Determine query type and retrieve relevant content
            is_visual = self._is_visual_query(question)
            relevant_docs = await self.pipeline.search_documents(question, top_k)
            
            if not relevant_docs:
                return {
                    'answer': "I couldn't find any relevant information in the uploaded documents to answer your question.",
                    'sources': [],
                    'images': [],
                    'confidence': 0.0,
                    'response_type': 'text'
                }
            
            # Step 2: Separate text and image results
            text_docs = [doc for doc in relevant_docs if doc.get('content_type') == 'text']
            image_docs = [doc for doc in relevant_docs if doc.get('content_type') == 'image']
            
            # Step 3: Prepare context from text documents
            context_parts = []
            text_sources = []
            
            max_chunks = min(Config.MAX_CONTEXT_CHUNKS, len(text_docs))
            
            for i, doc in enumerate(text_docs[:max_chunks]):
                context_parts.append(f"Text Source {i+1} (Score: {doc['score']:.3f}, File: {doc['filename']}): {doc['text']}")
                text_sources.append({
                    'filename': doc['filename'],
                    'chunk_id': doc.get('chunk_id', 0),
                    'score': doc['score'],
                    'content_type': 'text'
                })
            
            # Step 4: Prepare image information
            image_info_parts = []
            image_sources = []
            
            for i, img_doc in enumerate(image_docs[:5]):  # Limit to 5 images
                image_info_parts.append(
                    f"Image {i+1} (Score: {img_doc['score']:.3f}, File: {img_doc['pdf_filename']}): "
                    f"Caption: {img_doc['caption']}, Page: {img_doc['page_number']}"
                )
                image_sources.append({
                    'filename': img_doc['pdf_filename'],
                    'image_path': img_doc['image_path'],
                    'caption': img_doc['caption'],
                    'page_number': img_doc['page_number'],
                    'score': img_doc['score'],
                    'content_type': 'image',
                    'image_id': img_doc['image_id']
                })
            
            # Step 5: Generate answer based on content type
            if is_visual and image_docs:
                # Visual query with images found
                answer = await self._generate_multimodal_response(question, context_parts, image_info_parts)
                response_type = 'multimodal'
            elif text_docs:
                # Text-based response
                answer = await self._generate_text_response(question, context_parts)
                response_type = 'text'
            else:
                answer = "I found some relevant content but couldn't generate a comprehensive answer."
                response_type = 'text'
            
            # Calculate confidence
            all_scores = [doc['score'] for doc in relevant_docs[:max_chunks + 5]]
            avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
            
            return {
                'answer': answer,
                'sources': text_sources,
                'images': image_sources,
                'confidence': avg_score,
                'response_type': response_type,
                'retrieved_docs_count': len(text_docs),
                'retrieved_images_count': len(image_docs),
                'is_visual_query': is_visual
            }
            
        except Exception as e:
            return {
                'answer': f"An error occurred while processing your question: {str(e)}",
                'sources': [],
                'images': [],
                'confidence': 0.0,
                'response_type': 'error'
            }
    
    async def _generate_multimodal_response(self, question: str, text_context: List[str], image_context: List[str]) -> str:
        """Generate response that incorporates both text and image information"""
        try:
            context = "\n\n".join(text_context)
            image_info = "\n\n".join(image_context)
            
            prompt = f"""You are a helpful AI assistant that answers questions using both text and visual content from documents.

TEXT CONTEXT:
{context}

AVAILABLE IMAGES/FIGURES:
{image_info}

QUESTION: {question}

INSTRUCTIONS:
- Answer the question using information from both the text context and available images
- If the question asks for a specific figure/image, reference it clearly
- Mention which images are relevant to the answer
- If showing code, tables, or technical content, preserve exact formatting
- Be specific about which image(s) answer the user's question
- If no specific image matches the query, explain what images are available that might be related

ANSWER:"""

            # Generate with Gemini
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)],
                ),
            ]
            
            generate_content_config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=-1),
            )
            
            answer = ""
            for chunk in self.gemini_client.models.generate_content_stream(
                model="gemini-2.5-pro",
                contents=contents,
                config=generate_content_config,
            ):
                if chunk.text:
                    answer += chunk.text
            
            return answer.strip() if answer.strip() else "I couldn't generate a comprehensive multimodal answer."
            
        except Exception as e:
            return f"Error generating multimodal response: {str(e)}"
    
    async def _generate_text_response(self, question: str, context_parts: List[str]) -> str:
        """Generate text-only response"""
        try:
            context = "\n\n".join(context_parts)
            
            prompt = f"""You are a helpful AI assistant that answers questions based on provided document context.

CONTEXT FROM DOCUMENTS:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Answer the question using ONLY the information provided in the context above
- PRESERVE ALL ORIGINAL FORMATTING including:
  * Code blocks (maintain exact indentation and syntax)
  * Lists (preserve bullets and numbering)  
  * Tables (maintain table structure)
  * Special characters and symbols
  * Line breaks and paragraph structure
- If showing code or configuration, use proper formatting with ```
- For lists, preserve the original bullet style (-, *, •) or numbering
- For tables, maintain the original table structure with | characters
- Keep all technical notation exactly as it appears
- If unsure about any content, respond with "I don't have enough information"

ANSWER:"""

            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)],
                ),
            ]
            
            generate_content_config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=-1),
            )
            
            answer = ""
            for chunk in self.gemini_client.models.generate_content_stream(
                model="gemini-2.5-pro",
                contents=contents,
                config=generate_content_config,
            ):
                if chunk.text:
                    answer += chunk.text
            
            return answer.strip() if answer.strip() else "I couldn't generate an answer based on the provided context."
            
        except Exception as e:
            return f"Error generating text response: {str(e)}"
    
    async def search_images_specifically(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Search specifically for images based on query"""
        try:
            # Get embedding for the question
            question_embedding = self.embedding_service.embed_single_text(question)
            
            # Search for similar images
            similar_images = await self.pipeline.qdrant_service.search_images(
                query_embedding=question_embedding,
                limit=top_k
            )
            
            # Format results
            results = {
                'images': [
                    {
                        'image_path': img['image_path'],
                        'caption': img['caption'],
                        'page_number': img['page_number'],
                        'pdf_filename': img['pdf_filename'],
                        'score': img['score']
                    } 
                    for img in similar_images
                    if os.path.exists(img['image_path'])  # Only include existing images
                ]
            }
            return results
        except Exception as e:
            print(f"Error in search_images_specifically: {str(e)}")
            return {'images': []}
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        return await self.pipeline.get_collection_stats()
    
    async def get_document_summary(self, filename: Optional[str] = None) -> str:
        """Get summary of uploaded documents"""
        try:
            stats = await self.pipeline.get_collection_stats()
            
            if filename:
                # Search for specific file content
                results = await self.pipeline.search_documents(f"filename:{filename}", top_k=3)
                if results:
                    content_preview = results[0]['text'][:300] + "..."
                    return f"Document: {filename}\nPreview: {content_preview}\nTotal chunks: {len(results)}"
                else:
                    return f"No content found for file: {filename}"
            else:
                text_stats = stats.get('text_collection', {})
                image_stats = stats.get('image_collection', {})
                return (f"Text documents: {text_stats.get('points_count', 0)} chunks\n"
                       f"Images: {image_stats.get('points_count', 0)} images")
                
        except Exception as e:
            return f"Error retrieving document summary: {str(e)}"
    
    def cleanup(self):
        """Clean up resources"""
        self.pipeline.cleanup()

# Example usage
async def main():
    rag = RAGService()
    
    try:
        # Test connection and setup
        stats = await rag.get_collection_stats()
        print(f"Collection stats: {stats}")
        print("Multimodal RAG Service initialized successfully!")
        print("Ready to handle both text and visual queries!")
        
        # Example visual query
        # result = await rag.generate_answer("Show me figure 8-6")
        # print(f"Visual query result: {result}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        rag.cleanup()

if __name__ == "__main__":
    asyncio.run(main())