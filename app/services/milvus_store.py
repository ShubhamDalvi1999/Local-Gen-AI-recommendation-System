"""
Vector store implementation using Milvus for book recommendations
"""

import os
import json
import logging
import numpy as np
import time
import requests
from typing import List, Tuple, Dict, Optional, Any
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility, MilvusException, IndexType
from dotenv import load_dotenv
from app.core.config import MilvusSettings, MILVUS_SETTINGS, OLLAMA, MILVUS_CONNECT_CONFIG
from app.models.book import Book

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class MilvusVectorStore:
    """Milvus vector store for book embeddings with enhanced metadata support"""
    
    def __init__(self, milvus_settings: MilvusSettings = MILVUS_SETTINGS):
        """Initialize Milvus vector store with enhanced schema"""
        self.settings = milvus_settings
        self.collection_name = self.settings.collection_name
        self.dimension = self.settings.dimension
        self.connection_alias = self.settings.connection_alias
        self.collection: Collection = None  # type: ignore
        self.milvus_connection = None
        self._connect_to_milvus()
        self._init_collection()
    
    def _connect_to_milvus(self):
        """Connect to Milvus server with retries and explicit alias"""
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempting to connect to Milvus (attempt {attempt + 1}/{max_retries}) with alias '{self.connection_alias}'")
                
                # Ensure URI has proper protocol
                uri = self.settings.connection_uri
                if not uri.startswith(('http://', 'https://')):
                    uri = f"https://{uri}"
                host = uri.replace('https://', '').replace('http://', '')
                logger.info(f"Connecting to Milvus host: {host}")
                
                self.milvus_connection = connections.connect(
                    alias=self.connection_alias,
                    host=host,
                    port=MILVUS_CONNECT_CONFIG['port'],
                    token=MILVUS_CONNECT_CONFIG['token'],
                    secure=MILVUS_CONNECT_CONFIG['secure'],
                    timeout=MILVUS_CONNECT_CONFIG['timeout']
                )
                logger.info(f"Successfully connected to Milvus at {uri} with alias '{self.connection_alias}'")
                time.sleep(2)  # Add delay after connection
                logger.info("Delay of 2 seconds after connection established.")
                return
                
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                    time.sleep(retry_delay * (attempt + 1))
                    try:
                        connections.disconnect(self.connection_alias)
                    except:
                        pass
                else:
                    logger.error(f"Failed to connect to Milvus after {max_retries} attempts: {e}")
                    raise
    
    def _init_collection(self):
        """Initialize Milvus collection with enhanced schema and explicit connection"""
        try:
            logger.info(f"Starting _init_collection for collection '{self.collection_name}' using connection alias '{self.connection_alias}'")
            
            # Force clean up of existing collection
            if utility.has_collection(self.collection_name, using=self.connection_alias):
                logger.info(f"Found existing collection '{self.collection_name}', dropping it... using connection alias '{self.connection_alias}'")
                utility.drop_collection(self.collection_name, using=self.connection_alias)
                logger.info(f"Dropped existing collection '{self.collection_name}' using connection alias '{self.connection_alias}'")
                time.sleep(1)  # Add delay after drop
                logger.info("Delay of 1 second after dropping collection (if it existed).")
            
            logger.info(f"Creating collection: {self.collection_name} with dimension {self.dimension} using connection alias '{self.connection_alias}'")
            
            # Define fields using FieldSchema objects
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, description="Book ID"),
                FieldSchema(name="title", dtype=DataType.VARCHAR, description="Book Title", max_length=200),
                FieldSchema(name="description", dtype=DataType.VARCHAR, description="Book Description", max_length=65535),
                FieldSchema(name="authors", dtype=DataType.VARCHAR, description="Book Authors", max_length=200),
                FieldSchema(name="categories", dtype=DataType.VARCHAR, description="Book Categories", max_length=200),
                FieldSchema(name="metadata", dtype=DataType.VARCHAR, description="Book Metadata JSON", max_length=4096),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension, description="Book Description Embedding"),
            ]
            
            # Create schema
            schema = CollectionSchema(fields=fields, description="Book embeddings collection")
            logger.info("Schema created successfully. Now creating Collection object...")
            
            time.sleep(1)  # Add delay before Collection create
            logger.info("Delay of 1 second before creating Collection object.")
            
            # Create collection with explicit connection alias
            self.collection = Collection(
                name=self.collection_name,
                schema=schema,
                using=self.connection_alias
            )
            logger.info(f"Collection '{self.collection_name}' object created successfully using connection alias '{self.connection_alias}'")
            
            # Create index on embedding field
            index_params = {
                "index_type": "IVF_FLAT",  # Use string format instead of enum
                "metric_type": "COSINE",
                "params": {
                    "nlist": 1024
                }
            }
            self.collection.create_index(
                field_name="embedding",
                index_params=index_params
            )
            logger.info("Index created successfully on 'embedding' field.")
            
            # Load collection into memory
            self.collection.load(using=self.connection_alias)
            logger.info(f"Collection '{self.collection_name}' loaded into memory.")
            time.sleep(1)  # Add delay after collection load
            logger.info("Delay of 1 second after loading collection.")
            
        except Exception as e:
            logger.error(f"Failed to initialize Milvus collection: {e}")
            logger.error(f"Exception details: {e.__class__.__name__}, {e}")
            raise
        finally:
            logger.info(f"Finished _init_collection for collection '{self.collection_name}' using connection alias '{self.connection_alias}'")
    
    def add_books(self, books: List[Book], embeddings: List[List[float]]):
        """
        Adds book data and their embeddings to Milvus.
        """
        num_entities = len(books)
        logger.info(f"Starting add_books with {len(books)} books and {len(embeddings)} embeddings")
        
        if len(books) != len(embeddings):
            raise ValueError(f"Number of books ({len(books)}) does not match number of embeddings ({len(embeddings)})")

        batch_size = 5  # Reduced batch size for better debugging
        for i in range(0, num_entities, batch_size):
            batch_books = books[i:min(i + batch_size, num_entities)]
            batch_embeddings = embeddings[i:min(i + batch_size, num_entities)]
            
            logger.info(f"Processing batch {i//batch_size + 1} of {(num_entities + batch_size - 1)//batch_size}")
            
            batch_dict: Dict[str, list] = {
                "id": [],
                "title": [],
                "description": [],
                "authors": [],
                "categories": [],
                "metadata": [],
                "embedding": []
            }

            # Process each book in the batch
            for j, book in enumerate(batch_books):
                try:
                    logger.debug(f"Processing book {i + j + 1}/{num_entities}")
                    
                    # Extract and validate book data
                    book_id = int(book['id'])
                    title = str(book['title'] or '')
                    description = str(book['description'] or '')
                    authors = str(book['authors'] or '')
                    categories = str(book['categories'] or '')
                    
                    # Create metadata dictionary
                    metadata = {
                        'published_year': book.get('published_year'),
                        'average_rating': book.get('average_rating'),
                        'num_pages': book.get('num_pages'),
                        'ratings_count': book.get('ratings_count')
                    }
                    metadata_json = json.dumps(metadata)
                    
                    # Get corresponding embedding
                    embedding_list = batch_embeddings[j]
                    
                    # Validate embedding
                    if not isinstance(embedding_list, list) or len(embedding_list) != self.dimension:
                        raise ValueError(f"Invalid embedding format or dimension. Expected list of length {self.dimension}, got {type(embedding_list)} of length {len(embedding_list) if isinstance(embedding_list, list) else 'N/A'}")
                    
                    # Add to batch dictionary
                    batch_dict["id"].append(book_id)
                    batch_dict["title"].append(title)
                    batch_dict["description"].append(description)
                    batch_dict["authors"].append(authors)
                    batch_dict["categories"].append(categories)
                    batch_dict["metadata"].append(metadata_json)
                    batch_dict["embedding"].append(embedding_list)
                    
                except Exception as e:
                    logger.error(f"Error processing book {i + j + 1}: {e}")
                    raise

            # Convert to numpy arrays and insert
            try:
                logger.info(f"Converting batch {i//batch_size + 1} to NumPy arrays...")
                
                # Convert and validate each field
                insert_data = []
                for field_name, dtype in [
                    ("id", np.int64),
                    ("title", object),
                    ("description", object),
                    ("authors", object),
                    ("categories", object),
                    ("metadata", object),
                    ("embedding", np.float32)
                ]:
                    arr = np.array(batch_dict[field_name], dtype=dtype)
                    logger.debug(f"{field_name} array shape: {arr.shape}, dtype: {arr.dtype}")
                    insert_data.append(arr)
                
                logger.info(f"Attempting to insert batch {i//batch_size + 1}...")
                
                # Insert with timeout handling
                try:
                    self.collection.insert(insert_data)
                    logger.info(f"Successfully inserted batch {i//batch_size + 1}")
                    
                    # Verify insertion
                    time.sleep(1)  # Brief pause to allow for data persistence
                    try:
                        num_entities = self.collection.num_entities
                        logger.info(f"Current collection size: {num_entities} entities")
                    except Exception as e:
                        logger.warning(f"Could not verify collection size: {e}")
                    
                except MilvusException as e:
                    logger.error(f"Milvus insertion error for batch {i//batch_size + 1}: {e}")
                    raise
                
            except Exception as e:
                logger.error(f"Error during batch {i//batch_size + 1} insertion: {e}")
                logger.error("Batch data causing error:")
                for field, values in batch_dict.items():
                    if field != "embedding":  # Don't log large embedding vectors
                        logger.error(f"{field}: {values}")
                raise

        # Final verification and cleanup
        try:
            logger.info("Finalizing insertion...")
            self.collection.flush()
            time.sleep(3)  # Wait for flush to complete
            
            final_count = self.collection.num_entities
            logger.info(f"Final collection size: {final_count} entities")
            
            if final_count < num_entities:
                logger.warning(f"Warning: Expected {num_entities} entities but found {final_count}")
            
            logger.info("Insertion completed successfully")
        except Exception as e:
            logger.error(f"Error during final verification and cleanup: {e}")
            raise
    
    def find_similar_books(
        self,
        query: str,
        limit: int = 5,
        min_similarity: float = 0.6
    ) -> List[Dict[str, Any]]:
        """Find similar books using semantic search"""
        try:
            # Generate query embedding
            query_embedding = self.get_ollama_embedding(query)
            
            # Search parameters
            search_params = {
                "data": [query_embedding.tolist()],
                "anns_field": "embedding",
                "param": {"nprobe": 10},
                "limit": limit * 2,  # Get more results for filtering
                "output_fields": ["id", "title", "description", "authors", "categories", "metadata"],
                "consistency_level": "strong"  # Ensure strong consistency for search
            }
            
            # Perform search with explicit connection alias
            results = self.collection.search(**search_params, using=self.connection_alias)
            
            similar_books = []
            for hits in results:
                for hit in hits:
                    similarity = 1 - hit.distance  # For cosine similarity
                    
                    if similarity >= min_similarity:
                        try:
                            metadata = json.loads(hit.get('metadata', '{}'))
                        except json.JSONDecodeError:
                            metadata = {}
                        
                        book_data = {
                            'id': hit.get('id'),
                            'title': hit.get('title', ''),
                            'description': hit.get('description', ''),
                            'authors': hit.get('authors', ''),
                            'categories': hit.get('categories', ''),
                            'similarity': similarity,
                            'metadata': metadata
                        }
                        similar_books.append(book_data)
            
            # Sort by similarity and limit results
            similar_books.sort(key=lambda x: x['similarity'], reverse=True)
            return similar_books[:limit]
            
        except Exception as e:
            logger.error(f"Error finding similar books: {e}")
            raise
    
    def get_ollama_embedding(self, text: str) -> np.ndarray:
        """Generate embedding using Ollama API"""
        try:
            response = requests.post(
                f'http://{OLLAMA["HOST"]}:{OLLAMA["PORT"]}/api/embeddings',
                json={'model': OLLAMA['MODEL'], 'prompt': text},
                timeout=30
            )
            if response.status_code == 200:
                embedding = np.array(response.json()['embedding'], dtype=np.float32)  # Use full embedding
                # Normalize for cosine similarity
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                return embedding
            raise ValueError(f"Failed to get embedding: {response.text}")
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            raise
    
    def __del__(self):
        """Cleanup when object is destroyed and disconnect using alias"""
        try:
            if self.collection:
                self.collection.release()
                logger.info(f"Released collection '{self.collection_name}' in __del__")
            connections.disconnect(self.connection_alias)
            logger.info(f"Disconnected Milvus connection with alias '{self.connection_alias}' in __del__")
        except Exception as e:
            logger.warning(f"Error during Milvus cleanup in __del__: {e}") 