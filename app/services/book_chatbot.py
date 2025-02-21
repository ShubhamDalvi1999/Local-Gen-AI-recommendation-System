"""
Book recommendation chatbot using BERT embeddings and Ollama LLM
"""

import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import torch
import datetime
import os
import requests
import time
import logging
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from .vector_store import BookVectorStore

logger = logging.getLogger(__name__)

class ChatbotError(Exception):
    """Base exception class for chatbot errors"""
    pass

class ModelInitializationError(ChatbotError):
    """Raised when model initialization fails"""
    pass

class DatabaseError(ChatbotError):
    """Raised when database operations fail"""
    pass

class BookRecommendationChatbot:
    # Constants
    MAX_CACHE_SIZE = 100
    CACHE_EXPIRY_SECONDS = 3600  # 1 hour
    MIN_SIMILARITY_THRESHOLD = 0.3
    MAX_RESPONSE_LENGTH = 500
    REQUEST_TIMEOUT = 30
    MAX_RETRIES = 3
    RETRY_DELAY = 2

    def __init__(self, db_path='data/books.db', model_name='all-MiniLM-L6-v2'):
        """
        Initialize the chatbot with BERT model, Ollama, and vector store
        """
        self._initialize_with_retry()
        
    def _initialize_with_retry(self):
        """Initialize components with retry logic"""
        for attempt in range(self.MAX_RETRIES):
            try:
                self._init_bert_model()
                self._init_ollama()
                self._init_vector_store()
                self._init_cache()
                self.setup_prompt_templates()
                logger.info("Successfully initialized chatbot with all components")
                return
            except Exception as e:
                logger.error(f"Initialization attempt {attempt + 1} failed: {e}")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY)
                else:
                    raise ModelInitializationError(f"Failed to initialize after {self.MAX_RETRIES} attempts")

    def _init_bert_model(self):
        """Initialize BERT model with timeout"""
        with ThreadPoolExecutor() as executor:
            future = executor.submit(SentenceTransformer, 'all-MiniLM-L6-v2')
            try:
                self.bert_model = future.result(timeout=30)
            except TimeoutError:
                raise ModelInitializationError("BERT model initialization timed out")

    def _init_ollama(self):
        """Initialize Ollama with health check"""
        self.ollama_host = os.getenv('OLLAMA_HOST', 'localhost')
        self.ollama_port = os.getenv('OLLAMA_PORT', '11434')
        
        # Health check
        try:
            response = requests.get(
                f'http://{self.ollama_host}:{self.ollama_port}/api/health',
                timeout=5
            )
            if response.status_code != 200:
                raise ModelInitializationError("Ollama service is not healthy")
        except requests.exceptions.RequestException as e:
            raise ModelInitializationError(f"Failed to connect to Ollama: {e}")
            
        self.ollama = OllamaLLM(
            base_url=f'http://{self.ollama_host}:{self.ollama_port}',
            model="llama3.2:3b",
            temperature=0.7,
            num_ctx=2048,
            num_thread=4,
            timeout=self.REQUEST_TIMEOUT
        )

    def _init_vector_store(self):
        """Initialize vector store with connection test"""
        try:
            self.vector_store = BookVectorStore('data/books.db')
            # Test connection
            self.vector_store.test_connection()
        except Exception as e:
            raise DatabaseError(f"Failed to initialize vector store: {e}")

    def _init_cache(self):
        """Initialize response cache with timestamps"""
        self._response_cache: Dict[str, Dict] = {}
        self._cache_timestamps: Dict[str, float] = {}

    def _clean_cache(self):
        """Remove expired cache entries"""
        current_time = time.time()
        expired_keys = [
            k for k, t in self._cache_timestamps.items()
            if current_time - t > self.CACHE_EXPIRY_SECONDS
        ]
        for k in expired_keys:
            self._response_cache.pop(k, None)
            self._cache_timestamps.pop(k, None)

    def setup_prompt_templates(self):
        """Setup LangChain prompt templates for conversation"""
        self.conversation_prompt = PromptTemplate(
            input_variables=["context", "user_input"],
            template="""
            [INST] You are a book recommendation assistant. Based on these books:
            {context}

            Provide a brief response to: {user_input}
            Focus on 1-2 key matches and why they fit the request.
            Keep your response under 100 words. [/INST]
            """
        )

    def process_user_input(self, user_input: str) -> Dict:
        """Process user input and generate response"""
        if not user_input or not user_input.strip():
            raise ValueError("Empty user input")

        try:
            # Clean cache periodically
            self._clean_cache()

            # Check cache
            cache_key = user_input.lower().strip()
            if cache_key in self._response_cache:
                if time.time() - self._cache_timestamps[cache_key] <= self.CACHE_EXPIRY_SECONDS:
                    logger.info("Cache hit - returning cached response")
                    return self._response_cache[cache_key]

            # Find similar books
            similar_books = self.vector_store.find_similar_books(
                user_input,
                min_similarity=self.MIN_SIMILARITY_THRESHOLD
            )
            
            if not similar_books:
                return {
                    'response': "I couldn't find any books matching your request. Could you please try different keywords?",
                    'similar_books': []
                }

            # Prepare context
            context = self._prepare_context(similar_books)
            
            # Generate response with retry
            response = self._generate_response_with_retry(context, user_input)
            
            # Clean and validate response
            response = self._clean_response(response)
            
            result = {
                'response': response,
                'similar_books': similar_books
            }

            # Update cache
            if len(self._response_cache) >= self.MAX_CACHE_SIZE:
                oldest_key = min(self._cache_timestamps.items(), key=lambda x: x[1])[0]
                self._response_cache.pop(oldest_key)
                self._cache_timestamps.pop(oldest_key)
                
            self._response_cache[cache_key] = result
            self._cache_timestamps[cache_key] = time.time()
            
            return result

        except Exception as e:
            logger.error(f"Error processing user input: {e}", exc_info=True)
            raise ChatbotError(f"Failed to process request: {str(e)}")

    def _prepare_context(self, similar_books: List[Tuple]) -> str:
        """Prepare context for LLM from similar books"""
        return "\n".join([
            f"- {book['book']} (Score: {score:.2f}): {book.get('description', '')[:100]}"
            for book, score in similar_books[:3]  # Limit to top 3 books
        ])

    def _generate_response_with_retry(self, context: str, user_input: str) -> str:
        """Generate LLM response with retry logic"""
        for attempt in range(self.MAX_RETRIES):
            try:
                chain = LLMChain(llm=self.ollama, prompt=self.conversation_prompt)
                response = chain.run(
                    context=context,
                    user_input=user_input
                )
                
                if not response:
                    raise ValueError("Empty response from LLM")
                    
                return response
                
            except Exception as e:
                logger.warning(f"LLM attempt {attempt + 1} failed: {e}")
                if attempt == self.MAX_RETRIES - 1:
                    raise ChatbotError(f"Failed to generate response after {self.MAX_RETRIES} attempts")
                time.sleep(self.RETRY_DELAY)

    def _clean_response(self, response: str) -> str:
        """Clean and validate LLM response"""
        response = response.strip()
        
        # Remove common artifacts
        response = response.replace('[/INST]', '').strip()
        response = response.replace('[INST]', '').strip()
        
        # Truncate if too long
        if len(response) > self.MAX_RESPONSE_LENGTH:
            response = response[:self.MAX_RESPONSE_LENGTH] + "..."
            
        return response

    def save_conversation_history(self, user_input: str, response: str, file_path='conversation_history.json'):
        """Save conversation history with error handling"""
        try:
            history = {
                'user_input': user_input,
                'response': response,
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'a') as f:
                json.dump(history, f)
                f.write('\n')
                
        except Exception as e:
            logger.error(f"Failed to save conversation history: {e}")
            # Don't raise - this is non-critical functionality

    def get_device(self) -> torch.device:
        """Get the appropriate device (CPU/GPU) for BERT model"""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu') 