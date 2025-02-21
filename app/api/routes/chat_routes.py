"""
Flask routes for the chatbot interface using llama3.2:3b model
"""

from flask import Blueprint, request, jsonify, render_template, current_app
from app.services.rag_service import RAGService
from app.services.milvus_store import MilvusVectorStore
from app.core.config import OLLAMA_HOST, OLLAMA_PORT
import logging
from functools import wraps
import time
from typing import Dict, Callable
import re
import numpy as np
import requests

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

chat_bp = Blueprint('chat', __name__)

# Rate limiting
class RateLimiter:
    def __init__(self, requests_per_minute: int = 30):
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, list] = {}
        
    def is_rate_limited(self, ip: str) -> bool:
        current_time = time.time()
        if ip not in self.requests:
            self.requests[ip] = []
            
        # Clean old requests
        self.requests[ip] = [t for t in self.requests[ip] if current_time - t < 60]
        
        if len(self.requests[ip]) >= self.requests_per_minute:
            return True
            
        self.requests[ip].append(current_time)
        return False

rate_limiter = RateLimiter()

def rate_limit(f: Callable) -> Callable:
    @wraps(f)
    def decorated_function(*args, **kwargs):
        ip = request.remote_addr
        if rate_limiter.is_rate_limited(ip):
            return jsonify({
                'error': 'Rate limit exceeded. Please try again later.'
            }), 429
        return f(*args, **kwargs)
    return decorated_function

def validate_input(user_input: str) -> bool:
    """Validate user input for safety and quality"""
    if not user_input or not isinstance(user_input, str):
        return False
        
    # Remove whitespace
    user_input = user_input.strip()
    if not user_input:
        return False
        
    # Check length
    if len(user_input) > 500:  # Maximum 500 characters
        return False
        
    # Check for basic SQL injection patterns
    sql_patterns = r'(\bSELECT\b|\bUNION\b|\bDROP\b|\bDELETE\b|\bINSERT\b)'
    if re.search(sql_patterns, user_input, re.IGNORECASE):
        return False
        
    return True

# Initialize services with retry mechanism
def get_rag_service():
    if not hasattr(current_app, 'rag_service'):
        try:
            vector_store = MilvusVectorStore()
            current_app.rag_service = RAGService(vector_store=vector_store)
            logger.info("RAG service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG service: {e}")
            return None
    return current_app.rag_service

async def generate_query_embedding(text: str) -> np.ndarray:
    """Generate embedding for query text using Ollama."""
    try:
        response = requests.post(
            f'http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/embeddings',
            json={'model': 'llama3.2:3b', 'prompt': text},
            timeout=30
        )
        if response.status_code == 200:
            embedding = np.array(response.json()['embedding'], dtype=np.float32)
            # Normalize for cosine similarity
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding
        raise ValueError(f"Failed to get embedding: {response.text}")
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise

@chat_bp.route('/chat', methods=['GET'])
def chat_interface():
    """Render chat interface"""
    return render_template('chat.html')

@chat_bp.route('/api/chat', methods=['POST'])
@rate_limit
async def chat():
    """Handle chat API endpoint with RAG integration"""
    rag_service = get_rag_service()
    if not rag_service:
        logger.error("RAG service not initialized")
        return jsonify({
            'error': 'Chat service is currently unavailable. Please try again later.'
        }), 503

    try:
        # Validate request
        if not request.is_json:
            logger.error("Invalid content type")
            return jsonify({
                'error': 'Invalid content type. Please send JSON data.'
            }), 400

        data = request.get_json()
        if not data or 'message' not in data:
            logger.error("Missing message in request")
            return jsonify({
                'error': 'Missing message in request.'
            }), 400
            
        user_input = data['message']
        
        # Validate user input
        if not validate_input(user_input):
            logger.warning(f"Invalid user input: {user_input}")
            return jsonify({
                'error': 'Invalid input. Please provide a valid message.'
            }), 400
            
        logger.info(f"Processing chat request with input: {user_input}")
        
        # Generate embedding for user query
        try:
            query_embedding = await generate_query_embedding(user_input)
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            return jsonify({
                'error': 'Failed to process your request. Please try again.'
            }), 500
        
        # Process user input with RAG
        try:
            result = await rag_service.generate_response(
                query=user_input,
                query_embedding=query_embedding
            )
        except Exception as e:
            logger.error(f"RAG service error: {e}")
            return jsonify({
                'error': str(e)
            }), 500
        
        if not result or not result.get('success'):
            error_msg = result.get('error') if result else 'Unknown error'
            logger.error(f"Failed to generate response: {error_msg}")
            return jsonify({
                'error': 'Failed to generate response. Please try again.'
            }), 500
        
        # Format response
        response_data = {
            'response': result['answer'],
            'recommendations': []
        }
        
        # Add recommendations if available
        if result.get('context_used'):
            for chunk in result['context_used']:
                if not isinstance(chunk, dict):
                    continue
                    
                rec = {
                    'title': chunk.get('title', ''),
                    'similarity': chunk.get('similarity', 0),
                    'description': chunk.get('chunk_text', '')
                }
                
                if rec['title'] and isinstance(rec['similarity'], (int, float)):
                    response_data['recommendations'].append(rec)
        
        logger.info(f"Successfully processed chat request with {len(response_data['recommendations'])} recommendations")
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error processing chat request: {e}", exc_info=True)
        return jsonify({
            'error': 'An unexpected error occurred. Please try again.',
            'details': str(e) if current_app.debug else None
        }), 500

@chat_bp.route('/api/health')
def health_check():
    """Check the health of the chat service"""
    try:
        rag_service = get_rag_service()
        if not rag_service:
            return jsonify({
                'status': 'unhealthy',
                'message': 'RAG service not initialized'
            }), 503
            
        # Test basic functionality
        test_response = rag_service.generate_response("test health check")
        if not test_response or not test_response.get('success'):
            raise Exception("RAG service failed to generate test response")
            
        return jsonify({
            'status': 'healthy',
            'message': 'Chat service is operational'
        })
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'message': str(e)
        }), 503 