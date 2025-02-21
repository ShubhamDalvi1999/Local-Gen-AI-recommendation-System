"""
Core configuration settings for the application.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Database settings
DATABASE = {
    'default': {
        'URL': f'sqlite:///{os.path.join(BASE_DIR, "data", "books.db")}',
        'ECHO': os.getenv('SQL_ECHO', 'False').lower() == 'true'
    }
}

# Ollama settings
OLLAMA = {
    'HOST': os.getenv('OLLAMA_HOST', 'localhost'),
    'PORT': os.getenv('OLLAMA_PORT', '11434'),
    'MODEL': os.getenv('OLLAMA_MODEL', 'llama3.2:3b'),
    'DIMENSION': 384
}

OLLAMA_HOST = OLLAMA['HOST']
OLLAMA_PORT = OLLAMA['PORT']

# Data files
DATA_FILES = {
    'BOOKS_CSV': os.path.join(BASE_DIR, 'data', 'books', 'data.csv')
}

# Logging configuration
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': os.path.join(BASE_DIR, 'logs', 'app.log'),
            'mode': 'a',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default', 'file'],
            'level': 'INFO',
            'propagate': True
        }
    }
}

# Validate and load Milvus configuration
def _validate_milvus_config():
    """Validate and prepare Milvus configuration."""
    host = os.getenv('MILVUS_HOST')
    if not host:
        raise ValueError("MILVUS_HOST environment variable is not set")
    
    # Ensure host has protocol
    if not any(host.startswith(proto) for proto in ['https://', 'http://', 'tcp://']):
        host = f'https://{host}'
    
    port = os.getenv('MILVUS_PORT', '443')  # Default to HTTPS port
    token = os.getenv('MILVUS_TOKEN')
    if not token:
        raise ValueError("MILVUS_TOKEN environment variable is not set")
    
    # Common SSL settings
    ssl_settings = {
        'verify': True,  # Enable SSL verification
        'secure': True,
        'timeout': 30
    }
    
    # Configuration for MilvusClient
    client_config = {
        'uri': host,
        'token': str(token),
        **ssl_settings
    }
    
    # Configuration for connections.connect()
    connect_config = {
        'host': host,
        'port': str(port),
        'token': str(token),
        'user': '',  # Empty string for cloud deployments
        'password': '',  # Empty string for cloud deployments
        **ssl_settings
    }
    
    return client_config, connect_config

try:
    MILVUS_CLIENT_CONFIG, MILVUS_CONNECT_CONFIG = _validate_milvus_config()
    logger.info(f"Milvus configuration loaded successfully (Host: {MILVUS_CLIENT_CONFIG['uri']})")
except Exception as e:
    logger.error(f"Failed to load Milvus configuration: {e}")
    raise

# For backward compatibility
MILVUS_HOST = MILVUS_CLIENT_CONFIG['uri']
MILVUS_PORT = MILVUS_CONNECT_CONFIG['port']
MILVUS_TOKEN = MILVUS_CLIENT_CONFIG['token']
MILVUS_CONFIG = MILVUS_CLIENT_CONFIG  # Default to client config for existing code

# RAG configuration
RAG_CONFIG = {
    'model_name': 'llama3.2:3b',
    'temperature': 0.7,
    'max_tokens': 512,
    'chunk_size': 512,
    'chunk_overlap': 50,
    'cache_ttl': 3600,  # Cache time-to-live in seconds
    'retrieval': {
        'n_docs': 3,
        'min_similarity': 0.7,
        'rerank_top_k': 5,
        'max_tokens_per_chunk': 500
    },
    'prompt_templates': {
        'qa': """You are a knowledgeable AI book assistant. Use the provided book-related context to answer the question accurately and helpfully.
        If the answer isn't found in the context, acknowledge this and provide general book-related guidance if possible.
        
        Guidelines:
        - Focus on information present in the context
        - Include relevant book titles, authors, and publication details when available
        - If recommending books, explain why they're relevant
        - If discussing book content, avoid spoilers unless specifically asked
        
        Context:
        {context}
        
        Question: {question}
        
        Answer: Let me help you understand this better.""",
        
        'book_recommendation': """You are a personalized book recommendation assistant. Based on the provided context about books and the user's query,
        suggest relevant books that match their interests.
        
        Guidelines:
        - Consider genre preferences, themes, and reading level
        - Explain why each recommended book might interest them
        - Include author names and brief, spoiler-free descriptions
        - Mention similar books when relevant
        
        Available Book Context:
        {context}
        
        User Request: {question}
        
        Recommendations:""",
        
        'book_analysis': """You are a literary analysis expert. Analyze the book-related information from the context to provide
        insightful commentary about themes, writing style, and literary significance.
        
        Guidelines:
        - Focus on literary elements mentioned in the context
        - Compare with similar works when relevant
        - Discuss themes and motifs
        - Avoid personal opinions not supported by the context
        
        Book Context:
        {context}
        
        Analysis Request: {question}
        
        Literary Analysis:""",
        
        'summarize': """As a book expert, provide a clear and engaging summary of the book-related information.
        Focus on key points while maintaining accuracy and avoiding spoilers.
        
        Guidelines:
        - Highlight main themes and concepts
        - Include relevant publication details
        - Maintain objective tone
        - Preserve important context
        
        Book Context:
        {context}
        
        Summary:"""
    }
} 