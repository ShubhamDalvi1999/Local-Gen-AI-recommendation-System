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
from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema
from dotenv import load_dotenv
from ..core.config import OLLAMA, MILVUS_CONFIG, OLLAMA_HOST, OLLAMA_PORT

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class MilvusVectorStore:
    """Milvus vector store for book embeddings with enhanced metadata support"""
    
    def __init__(self, collection_name: str = "book_embeddings", dimension: int = None):
        """Initialize Milvus vector store with enhanced schema"""
        self.collection_name = collection_name
        self.dimension = dimension or OLLAMA['DIMENSION']
        self.client = self._connect_to_milvus()
        self._init_collection()
    
    def _connect_to_milvus(self):
        """Connect to Milvus server with retries"""
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempting to connect to Milvus (attempt {attempt + 1}/{max_retries})")
                
                # Ensure URI has proper protocol
                uri = MILVUS_CONFIG['uri']
                if not uri.startswith(('http://', 'https://')):
                    uri = f"https://{uri}"
                
                logger.info(f"Connecting to Milvus host: {uri}")
                
                # Create client with validated configuration
                client = MilvusClient(
                    uri=uri,
                    token=MILVUS_CONFIG['token'],
                    timeout=MILVUS_CONFIG['timeout']
                )
                
                # Test connection with a simple operation
                collections = client.list_collections()
                logger.info(f"Successfully connected to Milvus at {uri}")
                return client
                
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                    time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                else:
                    logger.error(f"Failed to connect to Milvus after {max_retries} attempts: {e}")
                    raise
    
    def _init_collection(self):
        """Initialize Milvus collection with enhanced schema"""
        try:
            # Force clean up of existing collection
            collections = self.client.list_collections()
            if self.collection_name in collections:
                logger.info(f"Found existing collection '{self.collection_name}', dropping it...")
                self.client.drop_collection(self.collection_name)
                logger.info(f"Dropped existing collection '{self.collection_name}'")
            
            logger.info(f"Creating collection: {self.collection_name} with dimension {self.dimension}")
            
            # Create collection with fields
            self.client.create_collection(
                collection_name=self.collection_name,
                dimension=self.dimension,
                fields=[
                    {
                        "name": "id",
                        "dtype": "Int64",
                        "description": "Unique ID for each book",
                        "is_primary": True,
                        "auto_id": False
                    },
                    {
                        "name": "title",
                        "dtype": "VarChar",
                        "max_length": 512,
                        "description": "Book title"
                    },
                    {
                        "name": "description",
                        "dtype": "VarChar",
                        "max_length": 65535,
                        "description": "Book description"
                    },
                    {
                        "name": "authors",
                        "dtype": "VarChar",
                        "max_length": 512,
                        "description": "Book authors"
                    },
                    {
                        "name": "categories",
                        "dtype": "VarChar",
                        "max_length": 512,
                        "description": "Book categories"
                    },
                    {
                        "name": "metadata",
                        "dtype": "JSON",
                        "description": "Additional book metadata"
                    }
                ]
            )
            
            # Create index on embedding field
            self.client.create_index(
                collection_name=self.collection_name,
                field_name="embedding",
                index_params={
                    "metric_type": "COSINE",
                    "index_type": "HNSW",
                    "params": {
                        "M": 16,
                        "efConstruction": 200
                    }
                }
            )
            
            logger.info("Collection and index created successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Milvus collection: {e}")
            raise

    def get_ollama_embedding(self, text: str) -> np.ndarray:
        """Generate embedding using Ollama API"""
        try:
            response = requests.post(
                f'http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/embeddings',
                json={'model': OLLAMA['MODEL'], 'prompt': text},
                timeout=30
            )
            if response.status_code == 200:
                embedding = np.array(response.json()['embedding'][:self.dimension], dtype=np.float32)
                # Normalize for cosine similarity
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                return embedding
            raise ValueError(f"Failed to get embedding: {response.text}")
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            raise

    def add_books(self, books: List[Dict[str, Any]], embeddings: Optional[List[np.ndarray]] = None):
        """Add books with their embeddings to Milvus"""
        try:
            if not books:
                raise ValueError("No books provided")
                
            if embeddings is not None and len(books) != len(embeddings):
                raise ValueError(f"Number of books ({len(books)}) doesn't match number of embeddings ({len(embeddings)})")
            
            # Generate embeddings if not provided
            if embeddings is None:
                embeddings = []
                for book in books:
                    text = f"{book['title']} {book.get('description', '')} {book.get('authors', '')}"
                    embedding = self.get_ollama_embedding(text)
                    embeddings.append(embedding)
            
            # Prepare entities for insertion
            entities = []
            for book, embedding in zip(books, embeddings):
                entity = {
                    "id": book['id'],
                    "title": book['title'],
                    "description": book.get('description', ''),
                    "authors": book.get('authors', ''),
                    "categories": book.get('categories', ''),
                    "metadata": json.dumps({
                        "published_year": book.get('published_year'),
                        "average_rating": book.get('average_rating'),
                        "num_pages": book.get('num_pages'),
                        "ratings_count": book.get('ratings_count')
                    }),
                    "embedding": embedding.tolist()
                }
                entities.append(entity)
            
            # Insert in batches
            batch_size = 100
            for i in range(0, len(entities), batch_size):
                batch = entities[i:i + batch_size]
                self.client.insert(
                    collection_name=self.collection_name,
                    data=batch
                )
                logger.info(f"Inserted batch of {len(batch)} books")
            
            logger.info(f"Successfully added {len(books)} books to Milvus")
            
        except Exception as e:
            logger.error(f"Failed to add books to Milvus: {e}")
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
                "limit": limit * 2,  # Get more results for filtering
                "output_fields": ["id", "title", "description", "authors", "categories", "metadata"]
            }
            
            # Perform search
            results = self.client.search(
                collection_name=self.collection_name,
                **search_params
            )
            
            similar_books = []
            for hits in results:
                for hit in hits:
                    similarity = 1 / (1 + hit.distance)  # Convert distance to similarity
                    
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

    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            # MilvusClient handles cleanup automatically
            pass
        except:
            pass 