#!/usr/bin/env python3
"""
Basic usage example for Ceph Semantic Storage.

This script demonstrates the core functionality:
1. Connect to Ceph
2. Index some objects
3. Perform semantic search
4. Find similar objects
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from core.rados_client import RadosClient
from core.embedding_generator import EmbeddingGenerator
from core.content_processor import ContentProcessor
from core.vector_store import VectorStore
from services.indexer import Indexer
from services.searcher import Searcher

def load_config():
    """Load configuration"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    print("=" * 60)
    print("Ceph Semantic Storage - Basic Usage Example")
    print("=" * 60)
    
    # Load configuration
    print("\n1. Loading configuration...")
    config = load_config()
    
    # Initialize components
    print("2. Initializing components...")
    
    # RADOS Client
    rados_client = RadosClient(
        config_file=config['ceph']['config_file'],
        client_name=config['ceph']['client_name'],
        pool_name=config['ceph']['pool_name']
    )
    
    # Embedding Generator
    embedding_gen = EmbeddingGenerator(
        model_name=config['embedding']['model'],
        device=config['embedding']['device']
    )
    
    # Content Processor
    content_processor = ContentProcessor(
        max_file_size_mb=config['indexing']['max_file_size_mb'],
        supported_extensions=config['indexing']['supported_extensions']
    )
    
    # Vector Store
    vector_store = VectorStore(
        persist_directory=config['vectordb']['persist_directory'],
        collection_name=config['vectordb']['collection_name']
    )
    
    print(f"   ✓ Embedding model: {embedding_gen.model_name}")
    print(f"   ✓ Vector store: {vector_store.collection_name}")
    print(f"   ✓ Pool: {rados_client.pool_name}")
    
    # Connect to Ceph
    print("\n3. Connecting to Ceph cluster...")
    rados_client.connect()
    print(f"   ✓ Connected to cluster: {rados_client.cluster.get_fsid()}")
    
    # Create indexer
    indexer = Indexer(
        rados_client=rados_client,
        embedding_generator=embedding_gen,
        content_processor=content_processor,
        vector_store=vector_store
    )
    
    # Index some objects
    print("\n4. Indexing objects...")
    stats = indexer.index_pool(limit=10, show_progress=True)
    print(f"   ✓ Indexed: {stats.successfully_indexed}")
    print(f"   ✓ Skipped: {stats.skipped}")
    print(f"   ✓ Duration: {stats.duration_seconds:.2f}s")
    
    # Create searcher
    searcher = Searcher(
        rados_client=rados_client,
        embedding_generator=embedding_gen,
        vector_store=vector_store
    )
    
    # Example 1: Semantic search
    print("\n5. Example 1: Semantic Search")
    print("   Query: 'configuration settings'")
    results = searcher.search("configuration settings", top_k=3)
    
    for i, result in enumerate(results, 1):
        print(f"\n   Result {i}:")
        print(f"   - Name: {result.object_name}")
        print(f"   - Score: {result.relevance_score:.3f}")
        print(f"   - Preview: {result.content_preview[:100]}...")
    
    # Example 2: Find similar objects
    if results:
        print("\n6. Example 2: Find Similar Objects")
        first_object = results[0].object_name
        print(f"   Finding objects similar to: {first_object}")
        
        similar = searcher.find_similar(first_object, top_k=3)
        
        for i, result in enumerate(similar, 1):
            print(f"\n   Similar {i}:")
            print(f"   - Name: {result.object_name}")
            print(f"   - Similarity: {result.relevance_score:.3f}")
    
    # Get statistics
    print("\n7. System Statistics")
    status = indexer.get_indexing_status()
    print(f"   - Total objects in pool: {status['total_objects_in_pool']}")
    print(f"   - Indexed objects: {status['indexed_objects']}")
    print(f"   - Coverage: {status['index_coverage']}")
    
    # Cleanup
    print("\n8. Disconnecting...")
    rados_client.disconnect()
    
    print("\n" + "=" * 60)
    print("✅ Example completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
