"""
Embedding generator for text content using sentence-transformers.

This module handles the generation of vector embeddings from text content,
supporting both local transformer models and future integration with OpenAI API.
"""

import logging
from typing import List, Union, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generate vector embeddings for text content.
    
    Supports local sentence-transformer models with optional future support
    for cloud-based embedding services (OpenAI, Cohere, etc.).
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        normalize_embeddings: bool = True,
        batch_size: int = 32
    ):
        """
        Initialize embedding generator.
        
        Args:
            model_name: Name of the sentence-transformers model
                       - all-MiniLM-L6-v2: Fast, 384 dims (default)
                       - all-mpnet-base-v2: Better quality, 768 dims
                       - paraphrase-multilingual-MiniLM-L12-v2: Multilingual
            device: Device to run on ('cpu', 'cuda', 'mps')
            normalize_embeddings: Whether to L2-normalize embeddings
            batch_size: Batch size for encoding
        """
        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        
        logger.info(f"Initializing embedding model: {model_name}")
        logger.info(f"Device: {device}")
        
        # Auto-detect device if cuda requested but not available
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            self.device = "cpu"
        
        # Load the model
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dimension}")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get the dimensionality of embeddings."""
        return self.embedding_dimension
    
    def encode(
        self,
        texts: Union[str, List[str]],
        show_progress: bool = False,
        convert_to_numpy: bool = True
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text string or list of texts
            show_progress: Whether to show progress bar
            convert_to_numpy: Return numpy arrays
            
        Returns:
            Embedding vector(s) as numpy array(s)
        """
        # Handle single string input
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False
        
        logger.debug(f"Encoding {len(texts)} text(s)")
        
        try:
            # Generate embeddings
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=convert_to_numpy,
                normalize_embeddings=self.normalize_embeddings
            )
            
            # Return single embedding if single input
            if single_input:
                return embeddings[0]
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def encode_batch(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of text strings
            show_progress: Whether to show progress bar
            
        Returns:
            Array of embeddings, shape (len(texts), embedding_dim)
        """
        return self.encode(texts, show_progress=show_progress, convert_to_numpy=True)
    
    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        # Ensure embeddings are normalized
        if not self.normalize_embeddings:
            embedding1 = embedding1 / np.linalg.norm(embedding1)
            embedding2 = embedding2 / np.linalg.norm(embedding2)
        
        return np.dot(embedding1, embedding2)
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "device": self.device,
            "normalize_embeddings": self.normalize_embeddings,
            "batch_size": self.batch_size,
            "max_seq_length": self.model.max_seq_length
        }
    
    def __repr__(self) -> str:
        return f"EmbeddingGenerator(model={self.model_name}, dim={self.embedding_dimension}, device={self.device})"


class OpenAIEmbeddingGenerator:
    """
    Generate embeddings using OpenAI API.
    
    This is a placeholder for future OpenAI integration. 
    To be implemented when cloud-based embeddings are needed.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small"
    ):
        """
        Initialize OpenAI embedding generator.
        
        Args:
            api_key: OpenAI API key
            model: Model name (text-embedding-3-small, text-embedding-3-large)
        """
        self.api_key = api_key
        self.model = model
        
        # Set dimensions based on model
        self.embedding_dimension = 1536 if "3-small" in model else 3072
        
        logger.info(f"Initialized OpenAI embeddings: {model}")
        
        # Note: Actual implementation would use openai library
        # import openai
        # self.client = openai.OpenAI(api_key=api_key)
        
        raise NotImplementedError(
            "OpenAI embeddings not yet implemented. "
            "Use sentence-transformers for local embeddings."
        )
    
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings using OpenAI API."""
        # Placeholder for future implementation
        pass
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        return self.embedding_dimension
