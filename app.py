import streamlit as st
import asyncio
import tempfile
import os
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime

# Import our services
from ingestion_pipeline import IngestionPipeline
from rag_service import RAGService

# Page configuration
st.set_page_config(
    page_title="RAG Chatbot with PDF Upload",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'rag_service' not in st.session_state:
    st.session_state.rag_service = None
if 'uploaded_files_info' not in st.session_state:
    st.session_state.uploaded_files_info = []

def initialize_rag_service():
    """Initialize RAG service if not already done"""
    if st.session_state.rag_service is None:
        st.session_state.rag_service = RAGService()
    return st.session_state.rag_service

async def process_uploaded_file(uploaded_file, rag_service):
    """Process uploaded PDF file"""
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Process the PDF
        result = await rag_service.pipeline.process_pdf_file(tmp_path)
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        return result
    except Exception as e:
        return {
            'success': False,
            'filename': uploaded_file.name,
            'error': str(e)
        }

async def get_answer(question: str, rag_service):
    """Get answer from RAG service"""
    return await rag_service.generate_answer(question)

def main():
    st.title("📚 RAG Chatbot with PDF Upload")
    st.markdown("Upload PDF documents and ask questions about their content!")
    
    # Sidebar for document upload and management
    with st.sidebar:
        st.header("📁 Document Management")
        
        # Initialize RAG service
        rag_service = initialize_rag_service()
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=['pdf'],
            accept_multiple_files=True,
            key="pdf_uploader"
        )
        
        if uploaded_files:
            if st.button("🔄 Process Documents", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {uploaded_file.name}...")
                    
                    # Run async function
                    result = asyncio.run(process_uploaded_file(uploaded_file, rag_service))
                    
                    if result['success']:
                        st.success(f"✅ {result['filename']}: {result['chunks_count']} chunks")
                        
                        # Add to uploaded files info
                        file_info = {
                            'filename': result['filename'],
                            'chunks_count': result['chunks_count'],
                            'upload_time': datetime.now().strftime("%H:%M:%S"),
                            'status': 'Success'
                        }
                        
                        # Check if file already exists in session state
                        existing_files = [f['filename'] for f in st.session_state.uploaded_files_info]
                        if result['filename'] not in existing_files:
                            st.session_state.uploaded_files_info.append(file_info)
                        
                    else:
                        st.error(f"❌ {result['filename']}: {result['error']}")
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                status_text.text("Processing complete!")
                st.rerun()
        
        # Display uploaded files information
        if st.session_state.uploaded_files_info:
            st.header("📋 Uploaded Documents")
            
            for file_info in st.session_state.uploaded_files_info:
                with st.expander(f"📄 {file_info['filename']}"):
                    st.write(f"**Chunks:** {file_info['chunks_count']}")
                    st.write(f"**Upload Time:** {file_info['upload_time']}")
                    st.write(f"**Status:** {file_info['status']}")
        
        # Collection statistics
        if st.button("📊 Collection Stats"):
            try:
                stats = asyncio.run(rag_service.get_collection_stats())
                st.json(stats)
            except Exception as e:
                st.error(f"Error getting stats: {str(e)}")
        
        # Clear collection button
        if st.button("🗑️ Clear All Documents", type="secondary"):
            if st.checkbox("I understand this will delete all documents"):
                try:
                    asyncio.run(rag_service.pipeline.qdrant_service.delete_collection())
                    st.session_state.uploaded_files_info = []
                    st.session_state.messages = []
                    st.success("Collection cleared!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error clearing collection: {str(e)}")
    
    # Main chat interface
    st.header("💬 Chat with your documents")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if available
            if message["role"] == "assistant" and "sources" in message:
                if message["sources"]:
                    with st.expander("📚 Sources"):
                        for i, source in enumerate(message["sources"], 1):
                            st.write(f"**Source {i}:** {source['filename']} (Score: {source['score']:.3f})")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        if not st.session_state.uploaded_files_info:
            st.warning("Please upload and process some PDF documents first!")
            return
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = asyncio.run(get_answer(prompt, rag_service))
                    
                    st.markdown(response['answer'])
                    
                    # Show confidence and sources
                    col1, col2 = st.columns(2)
                    with col1:
                        confidence = response['confidence']
                        if confidence > 0.8:
                            st.success(f"Confidence: {confidence:.2%}")
                        elif confidence > 0.6:
                            st.warning(f"Confidence: {confidence:.2%}")
                        else:
                            st.error(f"Confidence: {confidence:.2%}")
                    
                    with col2:
                        st.info(f"Sources found: {len(response['sources'])}")
                    
                    # Add assistant message to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response['answer'],
                        "sources": response['sources'],
                        "confidence": confidence
                    })
                    
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg
                    })

if __name__ == "__main__":
    main()