import streamlit as st
import asyncio
import tempfile
import os
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime
from PIL import Image
import shutil

# Import our services
from ingestion_pipeline import IngestionPipeline
from rag_service import RAGService

# Page configuration
st.set_page_config(
    page_title="Multimodal RAG Chatbot",
    page_icon="📚🖼️",
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
    """Process uploaded PDF file with multimodal extraction"""
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Process the PDF (now extracts both text and images)
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
    """Get multimodal answer from RAG service"""
    return await rag_service.generate_answer(question)

def display_image_safely(image_path: str, caption: str = "", max_width: int = 600):
    """Safely display an image with error handling"""
    try:
        if os.path.exists(image_path):
            # Load and display image
            img = Image.open(image_path)
            st.image(img, caption=caption, use_column_width=False, width=min(max_width, img.width))
            return True
        else:
            st.error(f"Image not found: {image_path}")
            return False
    except Exception as e:
        st.error(f"Error displaying image: {str(e)}")
        return False

def main():
    st.title("📚🖼️ Multimodal RAG Chatbot")
    st.markdown("Upload PDF documents and ask questions about both text and visual content!")
    
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
                        st.success(f"✅ {result['filename']}: {result['chunks_count']} chunks, {result.get('images_count', 0)} images")
                        
                        # Add to uploaded files info
                        file_info = {
                            'filename': result['filename'],
                            'chunks_count': result['chunks_count'],
                            'images_count': result.get('images_count', 0),
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
                    st.write(f"**Text Chunks:** {file_info['chunks_count']}")
                    st.write(f"**Images:** {file_info.get('images_count', 0)}")
                    st.write(f"**Upload Time:** {file_info['upload_time']}")
                    st.write(f"**Status:** {file_info['status']}")
        
        # Collection statistics
        if st.button("📊 Collection Stats"):
            try:
                stats = asyncio.run(rag_service.get_collection_stats())
                st.json(stats)
            except Exception as e:
                st.error(f"Error getting stats: {str(e)}")
        
        # Query type selector
        st.header("🔍 Query Options")
        query_type = st.selectbox(
            "Choose query type:",
            ["Auto-detect", "Text only", "Images only", "Multimodal"],
            help="Auto-detect will determine if you're asking for visual content"
        )
        
        # Clear collection button
        if st.button("🗑️ Clear All Documents", type="secondary"):
            if st.checkbox("I understand this will delete all documents and images"):
                try:
                    asyncio.run(rag_service.pipeline.qdrant_service.delete_collection())
                    # Also clear extracted images directory
                    if os.path.exists("extracted_images"):
                        shutil.rmtree("extracted_images")
                        os.makedirs("extracted_images", exist_ok=True)
                    
                    st.session_state.uploaded_files_info = []
                    st.session_state.messages = []
                    st.success("Collection and images cleared!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error clearing collection: {str(e)}")
    
    # Main chat interface
    st.header("💬 Chat with your documents")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show images if available
            if message["role"] == "assistant" and "images" in message and message["images"]:
                st.subheader("📸 Related Images:")
                
                # Display images in columns if multiple
                if len(message["images"]) > 1:
                    cols = st.columns(min(len(message["images"]), 3))
                    for i, image_info in enumerate(message["images"][:6]):  # Limit to 6 images
                        with cols[i % 3]:
                            display_image_safely(
                                image_info['image_path'], 
                                f"{image_info['caption']} (Page {image_info['page_number']})",
                                max_width=300
                            )
                            st.caption(f"Score: {image_info['score']:.3f}")
                else:
                    # Single image, display larger
                    for image_info in message["images"]:
                        display_image_safely(
                            image_info['image_path'], 
                            f"{image_info['caption']} (Page {image_info['page_number']})",
                            max_width=600
                        )
                        st.caption(f"Score: {image_info['score']:.3f}")
            
            # Show text sources if available
            if message["role"] == "assistant" and "sources" in message:
                if message["sources"]:
                    with st.expander("📚 Text Sources"):
                        for i, source in enumerate(message["sources"], 1):
                            st.write(f"**Source {i}:** {source['filename']} (Score: {source['score']:.3f})")
    
    # Chat input
    if prompt := st.chat_input("Ask about text or visual content in your documents..."):
        if not st.session_state.uploaded_files_info:
            st.warning("Please upload and process some PDF documents first!")
            return
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing documents and images..."):
                try:
                    if query_type == "Images only":
                        # Search only images
                        image_results = asyncio.run(rag_service.search_images_specifically(prompt))
                        
                        if image_results['images']:
                            st.markdown("Here are the relevant images I found:")
                            
                            # Display found images
                            for image_info in image_results['images']:
                                display_image_safely(
                                    image_info['image_path'], 
                                    f"{image_info['caption']} (Page {image_info['page_number']})"
                                )
                                st.caption(f"From: {image_info['pdf_filename']} | Score: {image_info['score']:.3f}")
                            
                            response_content = f"Found {len(image_results['images'])} relevant images."
                            
                            # Add to message history
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": response_content,
                                "images": image_results['images'],
                                "sources": [],
                                "response_type": "images_only"
                            })
                        else:
                            error_msg = "No relevant images found for your query."
                            st.markdown(error_msg)
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": error_msg,
                                "images": [],
                                "sources": []
                            })
                    
                    else:
                        # Regular multimodal response
                        response = asyncio.run(get_answer(prompt, rag_service))
                        
                        st.markdown(response['answer'])
                        
                        # Display images if found
                        if response.get('images'):
                            st.subheader("📸 Related Images:")
                            
                            if len(response['images']) > 1:
                                cols = st.columns(min(len(response['images']), 3))
                                for i, image_info in enumerate(response['images'][:6]):
                                    with cols[i % 3]:
                                        display_image_safely(
                                            image_info['image_path'], 
                                            f"{image_info['caption']} (Page {image_info['page_number']})",
                                            max_width=300
                                        )
                                        st.caption(f"Score: {image_info['score']:.3f}")
                            else:
                                for image_info in response['images']:
                                    display_image_safely(
                                        image_info['image_path'], 
                                        f"{image_info['caption']} (Page {image_info['page_number']})",
                                        max_width=600
                                    )
                                    st.caption(f"Score: {image_info['score']:.3f}")
                        
                        # Show confidence and statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            confidence = response['confidence']
                            if confidence > 0.8:
                                st.success(f"Confidence: {confidence:.2%}")
                            elif confidence > 0.6:
                                st.warning(f"Confidence: {confidence:.2%}")
                            else:
                                st.error(f"Confidence: {confidence:.2%}")
                        
                        with col2:
                            st.info(f"Text sources: {len(response['sources'])}")
                        
                        with col3:
                            st.info(f"Images found: {len(response.get('images', []))}")
                        
                        # Show response type
                        if response.get('is_visual_query'):
                            st.caption("🖼️ Visual query detected")
                        
                        # Add assistant message to chat history
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response['answer'],
                            "sources": response['sources'],
                            "images": response.get('images', []),
                            "confidence": confidence,
                            "response_type": response.get('response_type', 'text')
                        })
                    
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg,
                        "sources": [],
                        "images": []
                    })

if __name__ == "__main__":
    main()