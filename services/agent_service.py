"""
High-level agent service for natural language interface.
"""

import logging
from typing import Optional

from core.llm_agent import LLMAgent
from core.llm_provider import create_llm_provider
from core.rados_client import RadosClient
from core.embedding_generator import EmbeddingGenerator
from core.content_processor import ContentProcessor
from core.vector_store import VectorStore
from services.indexer import Indexer
from services.searcher import Searcher

logger = logging.getLogger(__name__)


class AgentService:
    """
    High-level service for LLM agent operations.
    
    Provides a simplified interface for creating and using the LLM agent.
    """
    
    def __init__(
        self,
        llm_config: dict,
        rados_client: RadosClient,
        embedding_generator: EmbeddingGenerator,
        content_processor: ContentProcessor,
        vector_store: VectorStore
    ):
        """
        Initialize agent service.
        
        Args:
            llm_config: LLM configuration dictionary
            rados_client: RADOS client instance
            embedding_generator: Embedding generator instance
            content_processor: Content processor instance
            vector_store: Vector store instance
        """
        # Create LLM provider
        self.llm_provider = create_llm_provider(llm_config)
        
        # Create service dependencies
        self.indexer = Indexer(
            rados_client=rados_client,
            embedding_generator=embedding_generator,
            content_processor=content_processor,
            vector_store=vector_store
        )
        
        self.searcher = Searcher(
            rados_client=rados_client,
            embedding_generator=embedding_generator,
            vector_store=vector_store
        )
        
        # Create agent
        self.agent = LLMAgent(
            llm_provider=self.llm_provider,
            rados_client=rados_client,
            indexer=self.indexer,
            searcher=self.searcher,
            vector_store=vector_store
        )
        
        logger.info("Initialized Agent Service")
    
    def execute(self, prompt: str, auto_confirm: bool = False):
        """
        Execute a natural language command.
        
        Args:
            prompt: User's natural language input
            auto_confirm: Auto-confirm destructive operations
            
        Returns:
            OperationResult
        """
        return self.agent.process_query(prompt, auto_confirm=auto_confirm)
    
    def chat(self, prompt: str):
        """
        Chat with the agent (maintains conversation context).
        
        Args:
            prompt: User's message
            
        Returns:
            OperationResult
        """
        return self.agent.process_query(prompt, auto_confirm=False)
    
    def clear_history(self):
        """Clear conversation history."""
        self.agent.clear_conversation()
