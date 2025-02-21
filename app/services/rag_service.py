"""
RAG (Retrieval Augmented Generation) service implementation.
Orchestrates the process of retrieving relevant context and generating responses using LLM.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from functools import lru_cache
import time
import re
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory

from .milvus_store import MilvusVectorStore
from ..core.config import OLLAMA_HOST, OLLAMA_PORT, RAG_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGService:
    """Service for handling RAG operations with proper context management."""
    
    def __init__(
        self,
        vector_store: MilvusVectorStore,
        model_name: str = RAG_CONFIG['model_name'],
        temperature: float = RAG_CONFIG['temperature'],
        max_tokens: int = RAG_CONFIG['max_tokens']
    ):
        """Initialize the RAG service."""
        self.vector_store = vector_store
        self.model_name = model_name
        self.config = RAG_CONFIG
        
        # Initialize LLM
        self.llm = Ollama(
            base_url=f"http://{OLLAMA_HOST}:{OLLAMA_PORT}",
            model=model_name,
            temperature=temperature,
            callbacks=[CallbackManager([StreamingStdOutCallbackHandler()])],
            max_tokens=max_tokens
        )
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Initialize prompt templates
        self.prompt_templates = {
            name: PromptTemplate(
                input_variables=["context", "question"],
                template=template
            )
            for name, template in RAG_CONFIG['prompt_templates'].items()
        }
        
        # Initialize chains
        self.chains = {
            name: LLMChain(
                llm=self.llm,
                prompt=template,
                verbose=True,
                memory=self.memory
            )
            for name, template in self.prompt_templates.items()
        }
        
        # Initialize cache
        self._init_cache()
    
    def _init_cache(self):
        """Initialize the response cache with TTL."""
        self.cache_ttl = self.config['cache_ttl']
        self.cache = {}
        self.cache_timestamps = {}
    
    def _get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached response if valid."""
        if cache_key in self.cache:
            timestamp = self.cache_timestamps[cache_key]
            if time.time() - timestamp <= self.cache_ttl:
                return self.cache[cache_key]
            else:
                # Clean up expired cache entry
                del self.cache[cache_key]
                del self.cache_timestamps[cache_key]
        return None
    
    def _cache_response(self, cache_key: str, response: Dict[str, Any]):
        """Cache response with timestamp."""
        self.cache[cache_key] = response
        self.cache_timestamps[cache_key] = time.time()
    
    def _format_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks into a single context string."""
        formatted_chunks = []
        for chunk in retrieved_chunks:
            text = chunk.get('chunk_text', '')
            metadata = chunk.get('metadata', {})
            # Enhanced metadata formatting
            source_info = []
            if metadata.get('title'):
                source_info.append(f"Title: {metadata['title']}")
            if metadata.get('authors'):
                source_info.append(f"Author(s): {metadata['authors']}")
            if metadata.get('published_year'):
                source_info.append(f"Year: {metadata['published_year']}")
            
            source = f"[Source: {' | '.join(source_info)}]"
            formatted_chunks.append(f"{text}\n{source}")
        
        return "\n\n".join(formatted_chunks)
    
    def _rerank_chunks(
        self,
        chunks: List[Dict[str, Any]],
        query: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Rerank chunks based on additional criteria."""
        # Simple reranking based on multiple factors
        for chunk in chunks:
            score = chunk.get('similarity', 0)
            metadata = chunk.get('metadata', {})
            
            # Boost factors
            if query.lower() in chunk.get('chunk_text', '').lower():
                score *= 1.2  # Direct match boost
            if metadata.get('published_year', 0) > 2010:
                score *= 1.1  # Recency boost
            
            chunk['rerank_score'] = score
        
        # Sort by rerank score
        reranked = sorted(chunks, key=lambda x: x.get('rerank_score', 0), reverse=True)
        return reranked[:top_k]
    
    def _get_relevant_context(
        self,
        query: str,
        query_embedding: np.ndarray,
        n_docs: int = None,
        min_similarity: float = None
    ) -> List[Dict[str, Any]]:
        """Retrieve and rerank relevant context."""
        n_docs = n_docs or self.config['retrieval']['n_docs']
        min_similarity = min_similarity or self.config['retrieval']['min_similarity']
        
        try:
            # Get similar chunks
            chunks = self.vector_store.find_similar_chunks(
                query_embedding=query_embedding,
                k=self.config['retrieval']['rerank_top_k']  # Get more for reranking
            )
            
            # Rerank chunks
            reranked_chunks = self._rerank_chunks(
                chunks=chunks,
                query=query,
                top_k=n_docs
            )
            
            # Filter by similarity
            filtered_chunks = [
                chunk for chunk in reranked_chunks
                if chunk.get('similarity', 0) >= min_similarity
            ]
            
            if not filtered_chunks:
                logger.warning(f"No chunks found above similarity threshold {min_similarity}")
                filtered_chunks = reranked_chunks[:2] if reranked_chunks else []
            
            return filtered_chunks
            
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return []
    
    async def generate_response(
        self,
        query: str,
        query_embedding: np.ndarray,
        prompt_type: str = None,
        n_docs: int = None,
        min_similarity: float = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """Generate a response using RAG with the specified prompt type."""
        try:
            # Process and clean query
            cleaned_query, detected_type = self._process_query(query)
            prompt_type = prompt_type or detected_type
            
            # Validate embedding quality
            if not self.validate_embedding_quality(query_embedding):
                logger.warning("Low quality query embedding detected")
                return {
                    "answer": "I apologize, but I couldn't process your question effectively. Could you rephrase it?",
                    "context_used": [],
                    "success": False,
                    "error": "Low quality embedding"
                }
            
            # Check cache if enabled
            if use_cache:
                cache_key = f"{cleaned_query}:{prompt_type}"
                cached = self._get_cached_response(cache_key)
                if cached:
                    logger.info("Using cached response")
                    return cached
            
            # Get relevant context
            relevant_chunks = self._get_relevant_context(
                query=cleaned_query,
                query_embedding=query_embedding,
                n_docs=n_docs,
                min_similarity=min_similarity
            )
            
            if not relevant_chunks:
                return {
                    "answer": "I apologize, but I couldn't find relevant information to answer your question accurately.",
                    "context_used": [],
                    "success": False,
                    "error": "No relevant context found"
                }
            
            # Format context
            context = self._format_context(relevant_chunks)
            
            # Enhance prompt with context-aware elements
            enhanced_prompt = self._enhance_prompt_with_context(
                prompt_type=prompt_type,
                context=context,
                query=cleaned_query
            )
            
            # Update chain with enhanced prompt
            enhanced_template = PromptTemplate(
                input_variables=["context", "question"],
                template=enhanced_prompt
            )
            
            chain = LLMChain(
                llm=self.llm,
                prompt=enhanced_template,
                verbose=True,
                memory=self.memory
            )
            
            # Generate response
            response = await chain.ainvoke({
                "context": context,
                "question": cleaned_query
            })
            
            result = {
                "answer": response["text"],
                "context_used": relevant_chunks,
                "success": True,
                "prompt_type": prompt_type,
                "metadata": {
                    "query_processed": cleaned_query,
                    "context_summary": self._extract_context_metadata(context),
                    "detected_preferences": self._extract_preferences(cleaned_query) if prompt_type == 'book_recommendation' else None,
                    "analysis_focus": self._extract_analysis_focus(cleaned_query) if prompt_type == 'book_analysis' else None
                }
            }
            
            # Cache response if enabled
            if use_cache:
                self._cache_response(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "answer": "I apologize, but I encountered an error while processing your question.",
                "context_used": [],
                "success": False,
                "error": str(e)
            }
    
    def update_prompt_template(self, prompt_type: str, new_template: str):
        """Update a specific prompt template."""
        if prompt_type not in self.prompt_templates:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
            
        self.prompt_templates[prompt_type] = PromptTemplate(
            input_variables=["context", "question"],
            template=new_template
        )
        
        self.chains[prompt_type] = LLMChain(
            llm=self.llm,
            prompt=self.prompt_templates[prompt_type],
            verbose=True,
            memory=self.memory
        )
    
    def clear_cache(self):
        """Clear the response cache."""
        self.cache = {}
        self.cache_timestamps = {}
    
    def preprocess_book_data(self, book_data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess book data to ensure consistent format."""
        try:
            # Clean and validate fields
            title = book_data.get("title", "").strip()
            description = book_data.get("description", "").strip()
            
            # Format authors
            authors = book_data.get("authors", [])
            if isinstance(authors, str):
                authors = [auth.strip() for auth in authors.split(',')]
            
            # Clean categories
            categories = book_data.get("categories", [])
            if isinstance(categories, str):
                categories = [cat.strip() for cat in categories.split(',')]
            
            # Validate year
            year = book_data.get("published_year")
            if year and isinstance(year, str):
                year = int(re.findall(r'\d{4}', year)[0]) if re.findall(r'\d{4}', year) else None
            
            return {
                "title": title,
                "description": description,
                "authors": authors,
                "metadata": {
                    "isbn": book_data.get("isbn"),
                    "categories": categories,
                    "published_year": year,
                    "rating": float(book_data.get("average_rating", 0)),
                    "page_count": int(book_data.get("num_pages", 0)),
                    "language": book_data.get("language", "en")
                }
            }
        except Exception as e:
            logger.error(f"Error preprocessing book data: {str(e)}")
            return book_data
    
    def validate_embedding_quality(self, embedding: np.ndarray) -> bool:
        """Validate embedding quality."""
        try:
            # Check for NaN values
            if np.isnan(embedding).any():
                return False
            
            # Check for zero vectors
            if np.all(embedding == 0):
                return False
            
            # Check for reasonable magnitude
            magnitude = np.linalg.norm(embedding)
            if magnitude < 0.1 or magnitude > 100:
                return False
            
            # Check for uniform distribution (potential issues)
            std = np.std(embedding)
            if std < 0.01:
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error validating embedding: {str(e)}")
            return False
    
    def _process_query(self, query: str) -> Tuple[str, str]:
        """Process query to determine type and clean format."""
        try:
            query_type = 'qa'  # default
            cleaned_query = query.strip()
            
            # Detect query type based on patterns
            query_lower = cleaned_query.lower()
            
            # Book recommendation patterns
            if any(word in query_lower for word in [
                'recommend', 'suggest', 'similar to', 'like', 'books about',
                'what should i read', 'looking for books'
            ]):
                query_type = 'book_recommendation'
            
            # Analysis patterns
            elif any(word in query_lower for word in [
                'analyze', 'analysis', 'theme', 'style', 'meaning',
                'interpret', 'explain the significance', 'literary devices',
                'symbolism', 'character development'
            ]):
                query_type = 'book_analysis'
            
            # Summary patterns
            elif any(word in query_lower for word in [
                'summarize', 'summary', 'brief', 'overview',
                'tell me about', 'what is the book about'
            ]):
                query_type = 'summarize'
            
            # Clean and normalize query
            cleaned_query = re.sub(r'\s+', ' ', cleaned_query)
            cleaned_query = cleaned_query.strip('?!.,')
            
            return cleaned_query, query_type
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return query.strip(), 'qa'
    
    def _extract_context_metadata(self, context: str) -> str:
        """Extract and summarize metadata from context."""
        try:
            # Extract book titles
            titles = re.findall(r'Title: ([^\n\[\]]+)', context)
            
            # Extract authors
            authors = re.findall(r'Author\(s\): ([^\n\[\]]+)', context)
            
            # Extract years
            years = re.findall(r'Year: (\d{4})', context)
            
            # Create summary
            summary_parts = []
            if titles:
                summary_parts.append(f"Books mentioned: {', '.join(titles)}")
            if authors:
                summary_parts.append(f"Authors referenced: {', '.join(authors)}")
            if years:
                years = sorted(map(int, years))
                summary_parts.append(f"Time period: {years[0]}-{years[-1]}")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error extracting context metadata: {str(e)}")
            return ""
    
    def _enhance_prompt_with_context(
        self,
        prompt_type: str,
        context: str,
        query: str
    ) -> str:
        """Enhance prompt based on context and query characteristics."""
        try:
            # Extract metadata summary
            metadata_summary = self._extract_context_metadata(context)
            
            # Get base prompt template
            base_prompt = self.prompt_templates[prompt_type].template
            
            # Add dynamic elements based on prompt type
            if prompt_type == 'book_recommendation':
                # Add preference hints from query
                preferences = self._extract_preferences(query)
                if preferences:
                    base_prompt += f"\n\nUser Preferences:\n{preferences}\n"
            
            elif prompt_type == 'book_analysis':
                # Add focus areas
                focus_areas = self._extract_analysis_focus(query)
                if focus_areas:
                    base_prompt += f"\n\nFocus Areas:\n{focus_areas}\n"
            
            # Add metadata summary if available
            if metadata_summary:
                base_prompt += f"\n\nContext Overview:\n{metadata_summary}\n"
            
            return base_prompt
            
        except Exception as e:
            logger.error(f"Error enhancing prompt: {str(e)}")
            return self.prompt_templates[prompt_type].template
    
    def _extract_preferences(self, query: str) -> str:
        """Extract reading preferences from query."""
        preferences = []
        
        # Genre preferences
        genres = re.findall(r'(?:like|enjoy|prefer|fan of)\s+([^,.!?]+)(?:books|novels)?', query.lower())
        if genres:
            preferences.append(f"Preferred genres: {', '.join(genres)}")
        
        # Specific authors/books mentioned
        mentions = re.findall(r'like\s+([^,.!?]+)\s+by\s+([^,.!?]+)', query)
        if mentions:
            preferences.append(f"Similar to: {', '.join([f'{book} by {author}' for book, author in mentions])}")
        
        return "\n".join(preferences) if preferences else ""
    
    def _extract_analysis_focus(self, query: str) -> str:
        """Extract analysis focus areas from query."""
        focus_areas = []
        
        # Check for specific analysis requests
        if 'theme' in query.lower():
            focus_areas.append("- Thematic analysis")
        if any(word in query.lower() for word in ['character', 'protagonist', 'antagonist']):
            focus_areas.append("- Character analysis")
        if any(word in query.lower() for word in ['style', 'writing', 'prose']):
            focus_areas.append("- Writing style analysis")
        if 'symbol' in query.lower():
            focus_areas.append("- Symbolism analysis")
        
        return "\n".join(focus_areas) if focus_areas else "" 