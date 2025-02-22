"""
Script to generate embeddings for books in the database.
This script uses Ollama to generate embeddings and stores them in Milvus.
"""

import sys
import json
import logging
import logging.config
from pathlib import Path
from tqdm import tqdm
import numpy as np
import time
import os
import requests
from typing import Dict, List
from dotenv import load_dotenv

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from sqlalchemy import create_engine, select, text
from app.models.book import Book
from app.core.config import DATABASE, OLLAMA, LOGGING, MILVUS_CONNECT_CONFIG
from app.services.milvus_store import MilvusVectorStore

# Initialize logging with the configuration from config.py
logging.config.dictConfig(LOGGING)
logger = logging.getLogger(__name__)
logger.debug("Logging initialized for generate_embeddings.py")

def check_ollama_service():
    """Check if Ollama service is running and the model is available"""
    ollama_host = OLLAMA['HOST']
    ollama_port = OLLAMA['PORT']
    base_url = f'http://{ollama_host}:{ollama_port}'
    
    print("\nChecking Ollama service...")
    try:
        # Check if service is running
        response = requests.get(f"{base_url}/api/version")
        if response.status_code != 200:
            print("❌ Ollama service is not running!")
            print("Please start Ollama by running this command in a separate terminal:")
            print(f"    ollama run {OLLAMA['MODEL']}")
            return False
            
        # Check if model is available
        response = requests.post(
            f"{base_url}/api/embeddings", 
            json={"model": OLLAMA['MODEL'], "prompt": "test"},
            timeout=30
        )
        if response.status_code != 200:
            print(f"❌ {OLLAMA['MODEL']} model is not available!")
            print("Please run this command in a separate terminal:")
            print(f"    ollama pull {OLLAMA['MODEL']}")
            return False
            
        print("✓ Ollama service is running and model is available")
        return True
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama service!")
        print("Please start Ollama by running this command in a separate terminal:")
        print(f"    ollama run {OLLAMA['MODEL']}")
        return False

def get_embedding(text: str) -> List[float]:
    """Get embedding for a text using Ollama API"""
    ollama_host = OLLAMA['HOST']
    ollama_port = OLLAMA['PORT']
    base_url = f'http://{ollama_host}:{ollama_port}'
    
    try:
        response = requests.post(
            f"{base_url}/api/embeddings",
            json={"model": OLLAMA['MODEL'], "prompt": text},
            timeout=30
        )
        response.raise_for_status()
        return response.json()['embedding']
    except Exception as e:
        logger.error(f"Failed to get embedding: {e}")
        raise

def generate_embeddings():
    """Generate embeddings for all books and store in Milvus."""
    print("\n=== Book Embedding Generation and Cloud Storage ===\n")
    
    try:
        # Run pre-flight checks
        print("Checking Ollama service...")
        if not check_ollama_service():
            print("❌ Ollama service check failed")
            sys.exit(1)
        print("✓ Ollama service check passed")
        
        # Create database engine and Milvus store
        print("\nInitializing services...")
        print("- Creating database connection...")
        engine = create_engine(DATABASE['default']['URL'], echo=False)
        print("✓ Database connection created")
        
        print("- Connecting to Milvus cloud storage...")
        try:
            milvus_store = MilvusVectorStore()
            print("✓ Successfully connected to Milvus cloud")
        except Exception as e:
            print(f"❌ Failed to connect to Milvus: {str(e)}")
            print("Please check your Milvus configuration and credentials.")
            sys.exit(1)
        
        # Get all books
        print("\nFetching books from database...")
        try:
            with engine.connect() as conn:
                print("- Executing database query...")
                result = conn.execute(select(Book))
                print("- Processing query results...")
                books = [dict(
                    id=book.id,
                    title=book.title,
                    description=book.description or '',
                    authors=book.authors or '',
                    categories=book.categories or '',
                    published_year=book.published_year,
                    average_rating=book.average_rating,
                    num_pages=book.num_pages,
                    ratings_count=book.ratings_count
                ) for book in result]
                
            total_count = len(books)
            print(f"✓ Found {total_count} books to process")
            
        except Exception as e:
            print(f"❌ Database error: {str(e)}")
            raise
        
        print("\nStarting embedding generation and storage:")
        print("----------------------------------------------")
        
        successful_uploads = 0
        failed_uploads = 0
        start_time = time.time()
        
        # Process books in batches
        batch_size = 5  # Reduced batch size to match Milvus batch size
        
        # Overall progress bar
        with tqdm(total=total_count, desc="Processing Books", unit="book") as pbar:
            for i in range(0, len(books), batch_size):
                batch = books[i:i + batch_size]
                
                try:
                    # Generate embeddings for the batch
                    batch_embeddings = []
                    for book in batch:
                        # Combine relevant text fields for embedding
                        text_for_embedding = f"{book['title']} {book['description']} {book['authors']} {book['categories']}"
                        embedding = get_embedding(text_for_embedding)
                        batch_embeddings.append(embedding)
                    
                    # Add books to Milvus with their embeddings
                    milvus_store.add_books(batch, batch_embeddings)
                    successful_uploads += len(batch)
                    
                    # Get collection information after each batch
                    try:
                        num_entities = milvus_store.collection.num_entities
                        logger.info(f"Current collection size: {num_entities} entities")
                    except Exception as e:
                        logger.warning(f"Could not get collection stats: {e}")
                    
                except Exception as e:
                    logger.error(f"Failed to process batch: {e}")
                    failed_uploads += len(batch)
                    continue
                finally:
                    pbar.update(len(batch))
        
        # Calculate timing
        total_time = time.time() - start_time
        
        # Final summary
        print("\n=== Upload Summary ===")
        print(f"Total books processed: {total_count}")
        print(f"Successfully uploaded: {successful_uploads}")
        print(f"Failed uploads: {failed_uploads}")
        print(f"Success rate: {(successful_uploads/total_count)*100:.2f}%")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average time per book: {total_time/total_count:.2f} seconds")
        print(f"Average time per batch: {total_time/(total_count/batch_size):.2f} seconds")
        
        if successful_uploads > 0:
            print("\n✓ Embeddings have been successfully stored in Milvus cloud")
            print("✓ The collection is ready for similarity search")
        else:
            print("\n❌ No embeddings were successfully stored")
            print("Please check the logs for errors and try again")
        
    except Exception as e:
        logger.error(f"Error in embedding generation: {e}")
        raise
    finally:
        # Clean up connections
        try:
            if 'engine' in locals():
                engine.dispose()
            if 'milvus_store' in locals():
                del milvus_store  # This will trigger cleanup in __del__
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

if __name__ == "__main__":
    print("Starting embedding generation script...")
    try:
        generate_embeddings()
        print("Embedding generation completed successfully!")
    except Exception as e:
        print(f"Error during embedding generation: {str(e)}")
        sys.exit(1) 