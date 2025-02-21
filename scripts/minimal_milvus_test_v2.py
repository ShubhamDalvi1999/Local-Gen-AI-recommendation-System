"""
Minimal test script for Milvus using lower-level API.
"""

import os
import sys
import logging
from pathlib import Path
from pymilvus import connections, Collection, FieldSchema, DataType, CollectionSchema, utility
from dotenv import load_dotenv
import time
import numpy as np
import requests

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Now we can import from app
from app.core.config import MILVUS_CONNECT_CONFIG, OLLAMA, OLLAMA_HOST, OLLAMA_PORT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

COLLECTION_NAME = "minimal_test_collection_v2"
DIMENSION = OLLAMA['DIMENSION']
MAX_RETRIES = 3
TIMEOUT = 30  # seconds

def get_ollama_embedding(text: str) -> np.ndarray:
    """Generate embedding using Ollama API"""
    try:
        response = requests.post(
            f'http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/embeddings',
            json={'model': OLLAMA['MODEL'], 'prompt': text},
            timeout=TIMEOUT
        )
        if response.status_code == 200:
            embedding = np.array(response.json()['embedding'][:DIMENSION], dtype=np.float32)
            # Normalize for cosine similarity
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding
        raise ValueError(f"Failed to get embedding: {response.text}")
    except Exception as e:
        logger.error(f"Error getting embedding: {e}")
        raise

def create_minimal_collection():
    """Creates a minimal Milvus collection for testing."""
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Attempting to connect to Milvus (attempt {attempt + 1}/{MAX_RETRIES})")
            
            # Parse host without protocol for lower-level API
            host = MILVUS_CONNECT_CONFIG['host']
            if host.startswith('https://'):
                host = host[8:]
            elif host.startswith('http://'):
                host = host[7:]
            
            logger.info(f"Connecting to Milvus host: {host}")
            
            # Connect to Milvus with the correct configuration
            connections.connect(
                alias="default",
                host=host,  # Host without protocol
                port=MILVUS_CONNECT_CONFIG['port'],
                token=MILVUS_CONNECT_CONFIG['token'],
                secure=True,  # Always true for cloud
                timeout=MILVUS_CONNECT_CONFIG['timeout']
            )
            logger.info("Successfully connected to Milvus")

            # Drop collection if it exists
            if utility.has_collection(COLLECTION_NAME):
                utility.drop_collection(COLLECTION_NAME)
                logger.info(f"Dropped existing collection '{COLLECTION_NAME}'")

            # Define fields with explicit types
            fields = [
                FieldSchema(
                    name="id",
                    dtype=DataType.INT64,
                    is_primary=True,
                    auto_id=False,
                    description="Book ID"
                ),
                FieldSchema(
                    name="title",
                    dtype=DataType.VARCHAR,
                    max_length=512,
                    description="Book title"
                ),
                FieldSchema(
                    name="description",
                    dtype=DataType.VARCHAR,
                    max_length=65535,
                    description="Book description"
                ),
                FieldSchema(
                    name="embedding",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=DIMENSION,
                    description="Book embedding vector"
                )
            ]

            # Create schema
            schema = CollectionSchema(
                fields=fields,
                description="Minimal book collection for testing",
                enable_dynamic_field=False
            )

            # Create collection
            collection = Collection(
                name=COLLECTION_NAME,
                schema=schema
            )
            logger.info(f"Created collection '{COLLECTION_NAME}'")

            # Create index
            index_params = {
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {"M": 16, "efConstruction": 200}
            }
            collection.create_index(
                field_name="embedding",
                index_params=index_params
            )
            logger.info("Created index on embedding field")

            return collection

        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                wait_time = (attempt + 1) * 2  # Exponential backoff
                logger.info(f"Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
                try:
                    connections.disconnect("default")
                except:
                    pass
            else:
                logger.error(f"Error during collection creation after {MAX_RETRIES} attempts: {e}")
                raise

def insert_minimal_data(collection):
    """Inserts a single test book."""
    try:
        # Test book data
        test_book = {
            "id": 1,
            "title": "Test Book Title",
            "description": "This is a test book description for embedding generation."
        }

        # Generate embedding
        text = f"{test_book['title']} {test_book['description']}"
        embedding = get_ollama_embedding(text)

        # Insert data
        entities = [
            {
                "id": test_book["id"],
                "title": test_book["title"],
                "description": test_book["description"],
                "embedding": embedding.tolist()
            }
        ]

        logger.info("=== Entity Debug Information ===")
        logger.info(f"Fields in entity: {list(entities[0].keys())}")
        logger.info(f"Title type: {type(entities[0]['title'])}")
        logger.info(f"Embedding type: {type(entities[0]['embedding'])}")
        logger.info(f"Embedding length: {len(entities[0]['embedding'])}")
        logger.info("=============================")

        collection.insert(entities)
        logger.info("Successfully inserted test book")

        # Flush to ensure data is persisted
        collection.flush()
        logger.info("Flushed data to storage")

        # Get collection information
        try:
            num_entities = collection.num_entities
            logger.info(f"Collection size: {num_entities} entities")
            
            # Get collection description
            desc = collection.description
            logger.info(f"Collection description: {desc}")
            
            # Get collection schema
            schema = collection.schema
            logger.info(f"Collection schema: {schema}")
        except Exception as e:
            logger.warning(f"Could not get complete collection information: {e}")

    except Exception as e:
        logger.error(f"Error during insertion: {e}")
        raise

def main():
    """Main execution function."""
    try:
        # Create collection
        collection = create_minimal_collection()

        # Load collection (required before operations)
        collection.load()
        logger.info("Collection loaded into memory")

        # Insert test data
        insert_minimal_data(collection)

        # Test search
        logger.info("Testing search functionality...")
        test_query = "Test Book Title"
        query_embedding = get_ollama_embedding(test_query)
        
        search_params = {
            "data": [query_embedding.tolist()],
            "anns_field": "embedding",
            "param": {"nprobe": 10},
            "limit": 1,
            "output_fields": ["title", "description"]
        }
        
        results = collection.search(**search_params)
        
        logger.info("=== Search Results ===")
        for hits in results:
            for hit in hits:
                logger.info(f"Found book: {hit}")
        logger.info("=====================")

        # Get final collection size
        final_size = collection.num_entities
        logger.info(f"Final collection size: {final_size} entities")

        # Release collection from memory
        collection.release()
        logger.info("Collection released from memory")

        # Close connection
        connections.disconnect("default")
        logger.info("Disconnected from Milvus")

        logger.info("Minimal Milvus test v2 completed successfully")

    except Exception as e:
        logger.error(f"Minimal Milvus test v2 failed: {e}")
        raise
    finally:
        # Ensure connection is closed
        try:
            connections.disconnect("default")
        except:
            pass

if __name__ == "__main__":
    main() 