import numpy as np
from typing import List, Dict, Optional, Union
import logging
from tqdm import tqdm
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedEmbeddingGenerator:
    """Optimized embedding generation with batching and error handling."""
    
    def __init__(
        self,
        model_name: str = "text-embedding-ada-002",
        batch_size: int = 32,
        max_retries: int = 3,
        timeout: int = 30
    ):
        """
        Initialize the embedding generator.
        
        Args:
            model_name (str): Name of the embedding model to use
            batch_size (int): Number of texts to process in each batch
            max_retries (int): Maximum number of retries for failed requests
            timeout (int): Timeout for API requests in seconds
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.timeout = timeout
        self._executor = ThreadPoolExecutor(max_workers=4)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _generate_single_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text with retry logic.
        
        Args:
            text (str): Text to generate embedding for
            
        Returns:
            np.ndarray: Generated embedding
        """
        # Implementation depends on the model being used
        # This is a placeholder - replace with actual implementation
        try:
            # Simulate embedding generation
            # Replace this with actual API call or model inference
            time.sleep(0.1)  # Simulated API call
            return np.random.rand(1536)  # Simulated embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise

    async def generate_embeddings_batch(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> List[np.ndarray]:
        """
        Generate embeddings for a batch of texts efficiently.
        
        Args:
            texts (List[str]): List of texts to generate embeddings for
            show_progress (bool): Whether to show progress bar
            
        Returns:
            List[np.ndarray]: List of generated embeddings
        """
        embeddings = []
        failed_indices = []
        
        for i in tqdm(range(0, len(texts), self.batch_size),
                     desc="Generating embeddings",
                     disable=not show_progress):
            batch = texts[i:i + self.batch_size]
            
            # Process batch in parallel
            tasks = [self._generate_single_embedding(text) for text in batch]
            try:
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Handle results and track failures
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Failed to generate embedding for text at index {i+j}: {str(result)}")
                        failed_indices.append(i+j)
                        embeddings.append(None)
                    else:
                        embeddings.append(result)
                        
            except Exception as e:
                logger.error(f"Batch processing error: {str(e)}")
                embeddings.extend([None] * len(batch))
                failed_indices.extend(range(i, i + len(batch)))
                
        if failed_indices:
            logger.warning(f"Failed to generate embeddings for {len(failed_indices)} texts")
            
        return embeddings, failed_indices

    def process_texts(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = True
    ) -> Dict[str, Union[List[np.ndarray], List[int]]]:
        """
        Process texts and generate embeddings with batching.
        
        Args:
            texts (List[str]): Texts to process
            batch_size (int, optional): Override default batch size
            show_progress (bool): Whether to show progress bar
            
        Returns:
            Dict containing embeddings and failed indices
        """
        if batch_size is not None:
            self.batch_size = batch_size
            
        # Run async embedding generation
        loop = asyncio.get_event_loop()
        embeddings, failed_indices = loop.run_until_complete(
            self.generate_embeddings_batch(texts, show_progress)
        )
        
        return {
            'embeddings': embeddings,
            'failed_indices': failed_indices
        }

    def __del__(self):
        """Cleanup resources."""
        self._executor.shutdown(wait=False) 