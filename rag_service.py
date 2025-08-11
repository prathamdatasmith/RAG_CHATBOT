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
    
    async def generate_answer(self, question: str, top_k: int = None) -> Dict[str, Any]:
        """Generate answer based on retrieved documents with enhanced context"""
        try:
            if top_k is None:
                top_k = Config.TOP_K
                
            # Step 1: Enhanced retrieval for large documents
            relevant_docs = await self._smart_retrieval(question, top_k)
            
            if not relevant_docs:
                return {
                    'answer': "I couldn't find any relevant information in the uploaded documents to answer your question.",
                    'sources': [],
                    'confidence': 0.0
                }
            
            # Step 2: Prepare enhanced context from more documents
            context_parts = []
            sources = []
            
            # Use more chunks for better context (up to MAX_CONTEXT_CHUNKS)
            max_chunks = min(Config.MAX_CONTEXT_CHUNKS, len(relevant_docs))
            
            for i, doc in enumerate(relevant_docs[:max_chunks]):
                context_parts.append(f"Source {i+1} (Score: {doc['score']:.3f}, File: {doc['filename']}): {doc['text']}")
                sources.append({
                    'filename': doc['filename'],
                    'chunk_id': doc['chunk_id'],
                    'score': doc['score']
                })
            
            context = "\n\n".join(context_parts)
            
            # Step 3: Generate answer using Gemini model with enhanced context
            answer = await self._generate_with_gemini(question, context, relevant_docs[:max_chunks])
            
            # Calculate confidence based on similarity scores
            avg_score = sum(doc['score'] for doc in relevant_docs[:max_chunks]) / max_chunks
            
            return {
                'answer': answer,
                'sources': sources,
                'confidence': avg_score,
                'retrieved_docs_count': len(relevant_docs),
                'context_chunks_used': max_chunks
            }
            
        except Exception as e:
            return {
                'answer': f"An error occurred while processing your question: {str(e)}",
                'sources': [],
                'confidence': 0.0
            }
    
    async def _smart_retrieval(self, question: str, top_k: int) -> List[Dict[str, Any]]:
        """Comprehensive retrieval for large documents - searches EVERYWHERE"""
        try:
            all_results = []
            
            # Strategy 1: Cast a VERY wide net - get tons of results
            broad_results = await self.pipeline.qdrant_service.search_similar(
                query_embedding=self.pipeline.embedding_service.embed_single_text(question),
                limit=top_k * 4  # Get 4x more results initially
            )
            all_results.extend(broad_results)
            
            # Strategy 2: Keyword-based search for comprehensive coverage
            keywords = self._extract_keywords(question)
            if keywords:
                for keyword in keywords:
                    keyword_results = await self.pipeline.search_documents(keyword, top_k * 2)
                    all_results.extend(keyword_results)
            
            # Strategy 3: Search for any numbers/chapters/sections mentioned
            numbers_terms = self._extract_any_references(question)
            if numbers_terms:
                for term in numbers_terms:
                    ref_results = await self.pipeline.search_documents(term, top_k)
                    all_results.extend(ref_results)
            
            # Strategy 4: If still no results, search with MINIMUM threshold
            if not all_results:
                emergency_results = await self.pipeline.qdrant_service.client.search(
                    collection_name=self.pipeline.qdrant_service.collection_name,
                    query_vector=self.pipeline.embedding_service.embed_single_text(question),
                    limit=top_k * 5,
                    score_threshold=0.01  # Almost no threshold - get everything
                )
                
                for result in emergency_results:
                    all_results.append({
                        'text': result.payload['text'],
                        'filename': result.payload['filename'],
                        'chunk_id': result.payload['chunk_id'],
                        'score': result.score,
                        'word_count': result.payload.get('word_count', 0)
                    })
            
            # Remove duplicates but keep MORE results
            seen_chunks = set()
            unique_results = []
            for result in all_results:
                chunk_id = f"{result['filename']}_{result['chunk_id']}"
                if chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)
                    unique_results.append(result)
            
            # Sort by score and return MANY more results
            unique_results.sort(key=lambda x: x['score'], reverse=True)
            return unique_results[:top_k * 3]  # Return 3x more results
            
        except Exception as e:
            print(f"Error in comprehensive retrieval: {str(e)}")
            return []
    
    def _extract_any_references(self, question: str) -> List[str]:
        """Extract ANY kind of reference (chapter, section, page, part, etc.)"""
        import re
        references = []
        
        # Look for ANY number-based references
        patterns = [
            r'chapter\s*(\d+)',
            r'ch\s*(\d+)', 
            r'section\s*(\d+)',
            r'part\s*(\d+)',
            r'page\s*(\d+)',
            r'p\s*(\d+)',
            r'lesson\s*(\d+)',
            r'unit\s*(\d+)',
            r'module\s*(\d+)',
            r'exercise\s*(\d+)',
            r'(\d+)'  # Any standalone number
        ]
        
        question_lower = question.lower()
        for pattern in patterns:
            matches = re.findall(pattern, question_lower)
            for match in matches:
                references.extend([
                    f"chapter {match}",
                    f"section {match}",
                    f"part {match}",
                    f"page {match}",
                    f"lesson {match}",
                    f"unit {match}",
                    f"module {match}",
                    match  # Just the number
                ])
        
        return list(set(references))  # Remove duplicates
    
    def _extract_keywords(self, question: str) -> List[str]:
        """Extract important keywords from question"""
        # Simple keyword extraction
        stop_words = {'what', 'is', 'this', 'document', 'about', 'who', 'how', 'when', 'where', 'why', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
        words = question.lower().split()
        keywords = [word.strip('.,!?;:"()') for word in words if word.lower() not in stop_words and len(word) > 2]
        return keywords
    
    async def _generate_with_gemini(self, question: str, context: str, docs: List[Dict]) -> str:
        """Generate answer using Gemini model with enhanced formatting preservation"""
        try:
            # Create enhanced prompt template
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
- For mathematical content, preserve all operators and formatting
- If unsure about any content, respond with "I don't have enough information"

ANSWER:"""

            # Prepare content for Gemini
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt),
                    ],
                ),
            ]
            
            # Generate content config
            generate_content_config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    thinking_budget=-1,
                ),
            )
            
            # Generate response using Gemini model
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
            return f"Error generating answer with Gemini: {str(e)}"
    
    def _generate_contextual_answer(self, question: str, context: str, docs: List[Dict]) -> str:
        """Generate answer based on context (simple extraction-based approach)"""
        # Simple approach: find the most relevant sentences
        question_lower = question.lower()
        
        # Keywords from question
        question_words = set(word.strip('.,!?;:"()').lower() for word in question.split() 
                           if len(word) > 3 and word.lower() not in ['what', 'how', 'when', 'where', 'why', 'who', 'which', 'that', 'this', 'these', 'those'])
        
        best_sentences = []
        
        for doc in docs[:3]:  # Use top 3 documents
            sentences = doc['text'].split('. ')
            for sentence in sentences:
                sentence_words = set(word.strip('.,!?;:"()').lower() for word in sentence.split())
                
                # Calculate word overlap
                overlap = len(question_words.intersection(sentence_words))
                if overlap > 0:
                    best_sentences.append((sentence.strip(), overlap, doc['score']))
        
        if not best_sentences:
            # Fallback: return first few sentences from top document
            top_doc = docs[0]['text']
            sentences = top_doc.split('. ')[:3]
            return '. '.join(sentences) + '.'
        
        # Sort by overlap and score
        best_sentences.sort(key=lambda x: (x[1], x[2]), reverse=True)
        
        # Take top sentences and form answer
        answer_sentences = []
        seen_content = set()
        
        for sentence, overlap, score in best_sentences[:3]:
            if sentence and sentence not in seen_content and len(sentence) > 20:
                answer_sentences.append(sentence)
                seen_content.add(sentence)
        
        if answer_sentences:
            answer = '. '.join(answer_sentences)
            if not answer.endswith('.'):
                answer += '.'
            return answer
        else:
            # Final fallback
            return docs[0]['text'][:500] + "..."
    
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
                return f"Total documents in collection: {stats.get('points_count', 0)} chunks"
                
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
        stats = await rag.pipeline.get_collection_stats()
        print(f"Collection stats: {stats}")
        print("RAG Service initialized successfully!")
        print("Ready to answer questions through Streamlit app!")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        rag.cleanup()

if __name__ == "__main__":
    asyncio.run(main())