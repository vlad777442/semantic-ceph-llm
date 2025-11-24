"""
Semantic search service for querying indexed objects.

This module provides natural language search capabilities over indexed
RADOS objects using vector similarity.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from core.rados_client import RadosClient
from core.embedding_generator import EmbeddingGenerator
from core.vector_store import VectorStore
from core.metadata_schema import SearchResult, SearchQuery

logger = logging.getLogger(__name__)


class Searcher:
    """
    Service for semantic search over indexed objects.
    
    Provides natural language query capabilities using vector similarity
    search with optional metadata filtering.
    """
    
    def __init__(
        self,
        rados_client: RadosClient,
        embedding_generator: EmbeddingGenerator,
        vector_store: VectorStore
    ):
        """
        Initialize searcher.
        
        Args:
            rados_client: RADOS client instance
            embedding_generator: Embedding generator instance
            vector_store: Vector store instance
        """
        self.rados_client = rados_client
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store
        
        logger.info("Initialized Searcher service")
    
    def search(
        self,
        query_text: str,
        top_k: int = 10,
        min_score: float = 0.0,
        pool_name: Optional[str] = None,
        content_type: Optional[str] = None,
        include_content: bool = False
    ) -> List[SearchResult]:
        """
        Search for objects matching the query.
        
        Args:
            query_text: Natural language query
            top_k: Number of results to return
            min_score: Minimum relevance score (0-1)
            pool_name: Filter by pool name
            content_type: Filter by content type
            include_content: Whether to include full content
            
        Returns:
            List of SearchResult objects ordered by relevance
        """
        logger.info(f"Searching for: '{query_text}' (top_k={top_k})")
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.encode(query_text)
            
            # Prepare filters
            filter_dict = {}
            if pool_name:
                filter_dict["pool_name"] = pool_name
            if content_type:
                filter_dict["content_type"] = content_type
            
            # Perform vector search
            raw_results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                filter_dict=filter_dict if filter_dict else None
            )
            
            # Convert to SearchResult objects
            search_results = []
            
            for object_id, distance, metadata in raw_results:
                # Convert distance to similarity score (for cosine distance)
                # Cosine distance ranges from 0 (identical) to 2 (opposite)
                # Convert to similarity: 1 - (distance / 2)
                relevance_score = 1.0 - (distance / 2.0)
                
                # Skip if below threshold
                if relevance_score < min_score:
                    continue
                
                # Parse metadata
                modified_at = None
                if "modified_at" in metadata:
                    try:
                        modified_at = datetime.fromisoformat(metadata["modified_at"])
                    except:
                        pass
                
                indexed_at = None
                if "indexed_at" in metadata:
                    try:
                        indexed_at = datetime.fromisoformat(metadata["indexed_at"])
                    except:
                        pass
                
                keywords = []
                if "keywords" in metadata and metadata["keywords"]:
                    keywords = metadata["keywords"].split(",")
                
                tags = []
                if "tags" in metadata and metadata["tags"]:
                    tags = metadata["tags"].split(",")
                
                # Create result
                result = SearchResult(
                    object_id=object_id,
                    object_name=metadata.get("object_name", ""),
                    pool_name=metadata.get("pool_name", ""),
                    relevance_score=relevance_score,
                    distance=distance,
                    content_preview=metadata.get("content_preview", ""),
                    summary=metadata.get("summary"),
                    keywords=keywords,
                    tags=tags,
                    content_type=metadata.get("content_type", ""),
                    size_bytes=metadata.get("size_bytes", 0),
                    modified_at=modified_at,
                    indexed_at=indexed_at
                )
                
                # Optionally fetch full content
                if include_content:
                    try:
                        self.rados_client.ensure_connected()
                        data = self.rados_client.read_object(result.object_name)
                        result.full_content = data.decode('utf-8', errors='ignore')
                    except Exception as e:
                        logger.warning(f"Could not fetch content for {result.object_name}: {e}")
                
                search_results.append(result)
            
            logger.info(f"Found {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    def search_by_query(self, query: SearchQuery) -> List[SearchResult]:
        """
        Search using a SearchQuery object.
        
        Args:
            query: SearchQuery object with all parameters
            
        Returns:
            List of SearchResult objects
        """
        return self.search(
            query_text=query.query_text,
            top_k=query.top_k,
            min_score=query.min_score,
            pool_name=query.pool_name,
            content_type=query.content_type,
            include_content=query.include_content
        )
    
    def find_similar(
        self,
        object_name: str,
        top_k: int = 10,
        exclude_self: bool = True
    ) -> List[SearchResult]:
        """
        Find objects similar to a given object.
        
        Args:
            object_name: Name of the reference object
            top_k: Number of similar objects to return
            exclude_self: Whether to exclude the reference object
            
        Returns:
            List of similar objects
        """
        logger.info(f"Finding similar objects to: {object_name}")
        
        try:
            # Get object ID
            object_id = self.rados_client.generate_object_id(object_name)
            
            # Get object from vector store
            obj_data = self.vector_store.get(object_id)
            if not obj_data:
                logger.error(f"Object {object_name} not found in index")
                raise ValueError(f"Object {object_name} not indexed")
            
            # Get embedding
            embedding = obj_data['embedding']
            
            # Search with the embedding
            raw_results = self.vector_store.search(
                query_embedding=embedding,
                top_k=top_k + 1 if exclude_self else top_k
            )
            
            # Convert results
            search_results = []
            
            for oid, distance, metadata in raw_results:
                # Skip self if requested
                if exclude_self and oid == object_id:
                    continue
                
                relevance_score = 1.0 - (distance / 2.0)
                
                result = SearchResult(
                    object_id=oid,
                    object_name=metadata.get("object_name", ""),
                    pool_name=metadata.get("pool_name", ""),
                    relevance_score=relevance_score,
                    distance=distance,
                    content_preview=metadata.get("content_preview", ""),
                    summary=metadata.get("summary"),
                    keywords=metadata.get("keywords", "").split(",") if metadata.get("keywords") else [],
                    tags=metadata.get("tags", "").split(",") if metadata.get("tags") else [],
                    content_type=metadata.get("content_type", ""),
                    size_bytes=metadata.get("size_bytes", 0)
                )
                
                search_results.append(result)
                
                if len(search_results) >= top_k:
                    break
            
            logger.info(f"Found {len(search_results)} similar objects")
            return search_results
            
        except Exception as e:
            logger.error(f"Similar search failed: {e}")
            raise
    
    def search_by_keywords(
        self,
        keywords: List[str],
        top_k: int = 10,
        match_all: bool = False
    ) -> List[SearchResult]:
        """
        Search by keywords (metadata-based search).
        
        Args:
            keywords: List of keywords to search for
            top_k: Number of results to return
            match_all: Whether all keywords must match (AND) or any (OR)
            
        Returns:
            List of matching objects
        """
        logger.info(f"Searching by keywords: {keywords}")
        
        # For now, convert to natural language query
        # In future, could implement direct metadata filtering
        query_text = " ".join(keywords)
        
        return self.search(
            query_text=query_text,
            top_k=top_k
        )
    
    def get_object_details(self, object_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about an indexed object.
        
        Args:
            object_name: Name of the object
            
        Returns:
            Dictionary with object details or None if not found
        """
        try:
            object_id = self.rados_client.generate_object_id(object_name)
            
            obj_data = self.vector_store.get(object_id)
            if not obj_data:
                return None
            
            return {
                "object_id": object_id,
                "object_name": object_name,
                "metadata": obj_data['metadata'],
                "has_embedding": obj_data['embedding'] is not None,
                "embedding_dimension": len(obj_data['embedding']) if obj_data['embedding'] else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get object details: {e}")
            return None
