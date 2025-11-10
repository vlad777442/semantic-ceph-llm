#!/usr/bin/env python3
"""
Batch indexing example with progress tracking and error handling.

This example shows how to:
1. Index large numbers of objects efficiently
2. Handle errors gracefully
3. Track progress and performance
4. Generate detailed reports
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import time
import json
from datetime import datetime
from pathlib import Path

from core.rados_client import RadosClient
from core.embedding_generator import EmbeddingGenerator
from core.content_processor import ContentProcessor
from core.vector_store import VectorStore
from services.indexer import Indexer

def load_config():
    """Load configuration"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_report(stats, filename="indexing_report.json"):
    """Save indexing report"""
    report = {
        "timestamp": datetime.now().isoformat(),
        "statistics": stats.to_dict(),
        "performance": {
            "objects_per_second": stats.successfully_indexed / stats.duration_seconds if stats.duration_seconds > 0 else 0,
            "bytes_per_second": stats.total_size_bytes / stats.duration_seconds if stats.duration_seconds > 0 else 0,
            "success_rate": (stats.successfully_indexed / stats.total_objects * 100) if stats.total_objects > 0 else 0
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    return report

def main():
    print("=" * 70)
    print("Batch Indexing Example - Ceph Semantic Storage")
    print("=" * 70)
    
    # Configuration
    config = load_config()
    
    print("\nConfiguration:")
    print(f"  Pool: {config['ceph']['pool_name']}")
    print(f"  Embedding Model: {config['embedding']['model']}")
    print(f"  Batch Size: {config['indexing']['batch_size']}")
    print(f"  Max File Size: {config['indexing']['max_file_size_mb']} MB")
    
    # Initialize components
    print("\n📡 Initializing components...")
    
    rados_client = RadosClient(
        config_file=config['ceph']['config_file'],
        client_name=config['ceph']['client_name'],
        pool_name=config['ceph']['pool_name']
    )
    
    embedding_gen = EmbeddingGenerator(
        model_name=config['embedding']['model'],
        device=config['embedding']['device'],
        batch_size=config['embedding']['batch_size']
    )
    
    content_processor = ContentProcessor(
        max_file_size_mb=config['indexing']['max_file_size_mb'],
        supported_extensions=config['indexing']['supported_extensions']
    )
    
    vector_store = VectorStore(
        persist_directory=config['vectordb']['persist_directory'],
        collection_name=config['vectordb']['collection_name']
    )
    
    # Connect
    print("🔗 Connecting to Ceph...")
    rados_client.connect()
    print(f"✅ Connected to cluster")
    
    # Get initial stats
    cluster_stats = rados_client.get_cluster_stats()
    print(f"\n📊 Cluster Statistics:")
    print(f"  Total Objects: {cluster_stats['num_objects']}")
    print(f"  Used Space: {cluster_stats['kb_used'] / (1024**2):.2f} GB")
    
    # Create indexer
    indexer = Indexer(
        rados_client=rados_client,
        embedding_generator=embedding_gen,
        content_processor=content_processor,
        vector_store=vector_store,
        batch_size=config['indexing']['batch_size']
    )
    
    # User input for batch size
    print("\n" + "=" * 70)
    try:
        limit = input("Enter number of objects to index (press Enter for all): ")
        limit = int(limit) if limit.strip() else None
    except:
        limit = None
    
    force_reindex = input("Force reindex existing objects? (y/n): ").lower() == 'y'
    
    # Start indexing
    print("\n🚀 Starting batch indexing...")
    print("=" * 70)
    
    start_time = time.time()
    
    stats = indexer.index_pool(
        limit=limit,
        force_reindex=force_reindex,
        show_progress=True
    )
    
    end_time = time.time()
    
    # Display results
    print("\n" + "=" * 70)
    print("✅ Indexing Complete!")
    print("=" * 70)
    
    print(f"\n📈 Results:")
    print(f"  Total Objects:        {stats.total_objects}")
    print(f"  Successfully Indexed: {stats.successfully_indexed}")
    print(f"  Skipped:             {stats.skipped}")
    print(f"  Failed:              {stats.failed}")
    print(f"  Total Size:          {stats.total_size_bytes / (1024**2):.2f} MB")
    print(f"  Duration:            {stats.duration_seconds:.2f}s")
    
    # Performance metrics
    if stats.duration_seconds > 0:
        print(f"\n⚡ Performance:")
        print(f"  Objects/second:      {stats.successfully_indexed / stats.duration_seconds:.2f}")
        print(f"  MB/second:           {(stats.total_size_bytes / (1024**2)) / stats.duration_seconds:.2f}")
        
        if stats.total_objects > 0:
            success_rate = (stats.successfully_indexed / stats.total_objects) * 100
            print(f"  Success Rate:        {success_rate:.1f}%")
    
    # Errors
    if stats.errors:
        print(f"\n⚠️  Errors ({len(stats.errors)}):")
        for i, error in enumerate(stats.errors[:10], 1):  # Show first 10
            print(f"  {i}. {error}")
        
        if len(stats.errors) > 10:
            print(f"  ... and {len(stats.errors) - 10} more errors")
    
    # Save report
    print("\n💾 Saving report...")
    report = save_report(stats)
    print(f"  Report saved to: indexing_report.json")
    
    # Vector store stats
    vec_stats = vector_store.get_stats()
    print(f"\n🗄️  Vector Store:")
    print(f"  Total Indexed: {vec_stats['total_objects']}")
    print(f"  Collection:    {vec_stats['collection_name']}")
    
    # Cleanup
    rados_client.disconnect()
    
    print("\n" + "=" * 70)
    print("🎉 Batch indexing completed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
