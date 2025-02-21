import nltk
from nltk.tokenize import sent_tokenize
from typing import List, Dict, Optional
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedTextChunker:
    """Advanced text chunking with semantic awareness and overlap."""
    
    def __init__(self, max_tokens: int = 512, overlap: float = 0.1):
        """
        Initialize the text chunker.
        
        Args:
            max_tokens (int): Maximum number of tokens per chunk
            overlap (float): Percentage of overlap between chunks (0.0 to 1.0)
        """
        self.max_tokens = max_tokens
        self.overlap = overlap
        try:
            nltk.download('punkt', quiet=True)
        except Exception as e:
            logger.warning(f"Failed to download NLTK punkt: {str(e)}")

    def _estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in text using a simple word-based approach."""
        return len(text.split())

    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Dict[str, any]]:
        """
        Create semantically coherent chunks with metadata.
        
        Args:
            text (str): Input text to chunk
            metadata (Dict, optional): Additional metadata to include with each chunk
            
        Returns:
            List[Dict]: List of chunks with their metadata
        """
        if not text.strip():
            return []

        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        overlap_size = int(self.max_tokens * self.overlap)
        
        for sentence in sentences:
            sentence_length = self._estimate_tokens(sentence)
            
            if current_length + sentence_length > self.max_tokens:
                # Create chunk with metadata
                chunk_text = ' '.join(current_chunk)
                chunk_data = {
                    'text': chunk_text,
                    'metadata': {
                        'start_idx': len(''.join(chunks)),
                        'length': len(chunk_text),
                        'sentences': len(current_chunk)
                    }
                }
                
                # Add additional metadata if provided
                if metadata:
                    chunk_data['metadata'].update(metadata)
                
                chunks.append(chunk_data)
                
                # Keep overlap for context
                if overlap_size > 0:
                    current_chunk = current_chunk[-overlap_size:]
                    current_length = self._estimate_tokens(' '.join(current_chunk))
                else:
                    current_chunk = []
                    current_length = 0
            
            current_chunk.append(sentence)
            current_length += sentence_length
            
        # Add final chunk if exists
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk_data = {
                'text': chunk_text,
                'metadata': {
                    'start_idx': len(''.join(chunks)),
                    'length': len(chunk_text),
                    'sentences': len(current_chunk)
                }
            }
            if metadata:
                chunk_data['metadata'].update(metadata)
            chunks.append(chunk_data)
            
        return chunks

    def process_batch(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> List[Dict[str, any]]:
        """
        Process a batch of texts into chunks.
        
        Args:
            texts (List[str]): List of texts to process
            metadatas (List[Dict], optional): List of metadata dicts for each text
            
        Returns:
            List[Dict]: List of all chunks with their metadata
        """
        if metadatas is None:
            metadatas = [None] * len(texts)
            
        all_chunks = []
        for text, metadata in tqdm(zip(texts, metadatas), total=len(texts), desc="Chunking texts"):
            chunks = self.chunk_text(text, metadata)
            all_chunks.extend(chunks)
            
        return all_chunks 