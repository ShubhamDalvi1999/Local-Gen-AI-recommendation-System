"""
Minimal POC script to test Milvus connection and data insertion.
"""

import os
import sys
import logging
import numpy as np
from pathlib import Path
from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema
from dotenv import load_dotenv

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Now we can import from app
from app.core.config import MILVUS_HOST, MILVUS_TOKEN, OLLAMA

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

COLLECTION_NAME = "minimal_test_collection"
DIMENSION = OLLAMA['DIMENSION']  # Using the same dimension as in main app

def create_minimal_collection(client):
    """Creates a minimal Milvus collection for testing."""
    try:
        # Drop collection if it exists
        collections = client.list_collections()
        if COLLECTION_NAME in collections:
            logger.info(f"Found existing collection '{COLLECTION_NAME}', dropping it...")
            client.drop_collection(COLLECTION_NAME)
            logger.info(f"Dropped existing collection '{COLLECTION_NAME}'")

        # Define minimal schema
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.INT64,
                description="Unique ID for each chunk",
                is_primary=True,
                auto_id=True
            ),
            FieldSchema(
                name="title",
                dtype=DataType.VARCHAR,
                max_length=200,
                description="Book Title"
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=DIMENSION,
                description="Text embedding vector"
            )
        ]
        schema = CollectionSchema(
            fields=fields,
            description="Minimal collection for testing auto_id"
        )

        # Create collection
        client.create_collection(
            COLLECTION_NAME, # collection_name (positional argument 1)
            dimension=DIMENSION, # dimension (positional argument 2 - corrected)
            schema=schema # schema (keyword argument)
        )
        logger.info(f"Created collection '{COLLECTION_NAME}' with auto_id=True")

    except Exception as e:
        logger.error(f"Error during collection creation: {e}")
        raise

def insert_minimal_data(client):
    """Inserts a single, minimal data entity."""
    try:
        # Prepare a single, minimal entity
        entity = {
            "title": "Test Book Title",  # Single string value
            "embedding": np.random.rand(DIMENSION).astype(np.float32).tolist()  # Random embedding as list of floats
        }
        
        # Insert the entity
        result = client.insert(
            collection_name=COLLECTION_NAME,
            data=[entity]  # Data should be a list of dictionaries
        )
        logger.info(f"Insertion result: {result}")
        logger.info("Successfully inserted one entity")
        
    except Exception as e:
        logger.error(f"Error during insertion: {e}")
        raise

def main():
    """Main execution function."""
    try:
        # Connect to Milvus
        logger.info(f"Connecting to Milvus at {MILVUS_HOST}...")
        client = MilvusClient(
            uri=MILVUS_HOST,
            token=MILVUS_TOKEN
        )
        logger.info("Successfully connected to Milvus")
        
        # Create collection
        create_minimal_collection(client)
        
        # Insert test data
        insert_minimal_data(client)
        
        # Test search functionality
        search_params = {
            "data": [np.random.rand(DIMENSION).astype(np.float32).tolist()],  # Random query vector
            "limit": 1,
            "output_fields": ["title"]
        }
        
        results = client.search(
            collection_name=COLLECTION_NAME,
            **search_params
        )
        
        logger.info("=== Search Results ===")
        for hits in results:
            for hit in hits:
                logger.info(f"Found document: {hit}")
        logger.info("=====================")
        
        logger.info("Minimal Milvus test completed successfully")
        
    except Exception as e:
        logger.error(f"Minimal Milvus test failed: {e}")
        raise

if __name__ == "__main__":
    main()
