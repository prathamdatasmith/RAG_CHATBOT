#!/usr/bin/env python3
"""
Setup script for RAG Chatbot Pipeline
"""

import os
import subprocess
import sys
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        sys.exit(1)

def check_env_file():
    """Check if .env file exists with required variables"""
    env_path = Path(".env")
    
    if not env_path.exists():
        print("❌ .env file not found!")
        print("Please create a .env file with the following variables:")
        print("""
QDRANT_URL="https://f5279c79-7150-46bc-ac4e-32cc3b0be830.us-west-1-0.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY="your_api_key_here"
COLLECTION_NAME="chatbot_docs"
        """)
        return False
    
    # Read and check required variables
    required_vars = ["QDRANT_URL", "QDRANT_API_KEY", "COLLECTION_NAME"]
    env_content = env_path.read_text()
    
    missing_vars = []
    for var in required_vars:
        if var not in env_content:
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        return False
    
    print("✅ Environment file looks good!")
    return True

def create_directories():
    """Create necessary directories"""
    directories = ["uploads", "logs", "temp"]
    
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
    
    print("✅ Directories created!")

def init_collections():
    """Initialize Qdrant collections"""
    from qdrant_client import QdrantClient
    from config import Config
    
    client = QdrantClient(url=Config.QDRANT_URL, api_key=Config.QDRANT_API_KEY)
    
    # Initialize text collection
    try:
        client.create_collection(
            collection_name=Config.TEXT_COLLECTION_NAME,
            vectors_config={
                "text_vector": dict(
                    size=Config.VECTOR_SIZE,
                    distance="Cosine"
                )
            }
        )
        print(f"✅ Created text collection: {Config.TEXT_COLLECTION_NAME}")
    except Exception as e:
        print(f"ℹ️ Text collection already exists or error: {str(e)}")
    
    # Initialize images collection
    try:
        client.create_collection(
            collection_name=Config.IMAGES_COLLECTION_NAME,
            vectors_config={
                "image_vector": dict(size=Config.IMAGE_VECTOR_SIZE, distance="Cosine")
            }
        )
        print(f"✅ Created images collection: {Config.IMAGES_COLLECTION_NAME}")
    except Exception as e:
        print(f"ℹ️ Images collection already exists or error: {str(e)}")

def main():
    """Main setup function"""
    print("🚀 Setting up RAG Chatbot Pipeline...\n")
    
    # Step 1: Check environment file
    if not check_env_file():
        print("Please fix the .env file and run setup again.")
        sys.exit(1)
    
    # Step 2: Install requirements
    install_requirements()
    
    # Step 3: Create directories
    create_directories()
    
    # Step 4: Initialize collections
    init_collections()
    
    print("\n🎉 Setup completed successfully!")
    print("\nTo start the application:")
    print("1. Test the pipeline: python test_pipeline.py")
    print("2. Run the Streamlit app: streamlit run app.py")
    print("\nHappy chatting! 🤖📚")

if __name__ == "__main__":
    main()