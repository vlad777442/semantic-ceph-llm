"""
Metadata schema definitions for semantic object storage.

This module defines the data models used throughout the system for storing
and retrieving object metadata, embeddings, and search results.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
import json


class ObjectMetadata(BaseModel):
    """
    Complete metadata for a RADOS object in the semantic storage system.
    
    This schema stores all information needed for semantic search and retrieval,
    including embeddings, content previews, and derived metadata.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # Core identifiers
    object_id: str = Field(..., description="Unique identifier (hash of pool+object_name)")
    object_name: str = Field(..., description="RADOS object key/name")
    pool_name: str = Field(..., description="Ceph pool name")
    
    # Content metadata
    content_type: str = Field(default="text/plain", description="MIME type or file extension")
    size_bytes: int = Field(..., description="Size of the object in bytes")
    encoding: Optional[str] = Field(default="utf-8", description="Text encoding")
    
    # Timestamps
    created_at: Optional[datetime] = Field(default=None, description="Object creation time in RADOS")
    modified_at: Optional[datetime] = Field(default=None, description="Last modification time in RADOS")
    indexed_at: datetime = Field(default_factory=datetime.now, description="Time when object was indexed")
    
    # Content and embeddings
    content_preview: str = Field(default="", description="First N characters of content")
    full_text: Optional[str] = Field(default=None, description="Full text content (for small files)")
    embedding_model: str = Field(..., description="Model used to generate embeddings")
    embedding_dimensions: int = Field(..., description="Dimensionality of embedding vector")
    
    # Derived metadata (from LLM processing)
    summary: Optional[str] = Field(default=None, description="LLM-generated summary")
    keywords: List[str] = Field(default_factory=list, description="Extracted or generated keywords")
    tags: List[str] = Field(default_factory=list, description="User or auto-generated tags")
    
    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata fields")
    
    # Chunking information (for large files)
    is_chunked: bool = Field(default=False, description="Whether object was split into chunks")
    chunk_index: Optional[int] = Field(default=None, description="Chunk index if chunked")
    total_chunks: Optional[int] = Field(default=None, description="Total number of chunks")
    parent_object_id: Optional[str] = Field(default=None, description="Parent object if this is a chunk")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return self.model_dump(mode='json')
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ObjectMetadata":
        """Create from dictionary."""
        return cls(**data)


class SearchResult(BaseModel):
    """
    Result from a semantic search query.
    
    Contains the matched object metadata, relevance score, and optional
    retrieved content.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # Matching object
    object_id: str
    object_name: str
    pool_name: str
    
    # Relevance metrics
    relevance_score: float = Field(..., description="Similarity score (0-1, higher is better)")
    distance: float = Field(..., description="Vector distance metric")
    
    # Content
    content_preview: str = Field(default="")
    summary: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    
    # Metadata
    content_type: str
    size_bytes: int
    modified_at: Optional[datetime] = None
    indexed_at: Optional[datetime] = None
    
    # Optional: full content if requested
    full_content: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump(mode='json')


class IndexingStats(BaseModel):
    """Statistics from an indexing operation."""
    
    total_objects: int = 0
    successfully_indexed: int = 0
    failed: int = 0
    skipped: int = 0
    total_size_bytes: int = 0
    duration_seconds: float = 0.0
    errors: List[str] = Field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()


class SearchQuery(BaseModel):
    """Parameters for a semantic search query."""
    
    query_text: str = Field(..., description="Natural language query")
    top_k: int = Field(default=10, description="Number of results to return")
    min_score: float = Field(default=0.0, description="Minimum relevance score")
    
    # Filters
    pool_name: Optional[str] = None
    content_type: Optional[str] = None
    tags: Optional[List[str]] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    
    # Options
    include_content: bool = Field(default=False, description="Include full content in results")
    rerank: bool = Field(default=False, description="Apply re-ranking with LLM")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump(mode='json')


class SystemStats(BaseModel):
    """System-wide statistics."""
    
    total_indexed_objects: int = 0
    total_embeddings: int = 0
    total_storage_bytes: int = 0
    pools_indexed: List[str] = Field(default_factory=list)
    embedding_model: str = ""
    collection_name: str = ""
    last_index_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump(mode='json')
