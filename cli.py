#!/usr/bin/env python3
"""
Command-line interface for Ceph Semantic Storage System.

This CLI provides commands for indexing, searching, and managing
semantic storage of RADOS objects.
"""

import click
import yaml
import logging
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
from rich import print as rprint
from datetime import datetime

from core.rados_client import RadosClient
from core.embedding_generator import EmbeddingGenerator
from core.content_processor import ContentProcessor
from core.vector_store import VectorStore
from services.indexer import Indexer
from services.searcher import Searcher
from services.watcher import Watcher

# Initialize console for rich output
console = Console()

# Default config path
DEFAULT_CONFIG = "config.yaml"


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        console.print(f"[red]Error loading config: {e}[/red]")
        sys.exit(1)


def setup_logging(config: dict) -> None:
    """Setup logging configuration."""
    log_config = config.get('logging', {})
    level = getattr(logging, log_config.get('level', 'INFO'))
    log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_config.get('file', 'semantic_storage.log'))
        ]
    )


def create_components(config: dict):
    """Create and initialize all system components."""
    # RADOS Client
    rados_config = config['ceph']
    rados_client = RadosClient(
        config_file=rados_config['config_file'],
        client_name=rados_config['client_name'],
        cluster_name=rados_config['cluster_name'],
        pool_name=rados_config['pool_name']
    )
    
    # Embedding Generator
    emb_config = config['embedding']
    embedding_gen = EmbeddingGenerator(
        model_name=emb_config['model'],
        device=emb_config['device'],
        normalize_embeddings=emb_config['normalize_embeddings'],
        batch_size=emb_config['batch_size']
    )
    
    # Content Processor
    idx_config = config['indexing']
    content_processor = ContentProcessor(
        max_file_size_mb=idx_config['max_file_size_mb'],
        encoding_detection=idx_config['encoding_detection'],
        fallback_encoding=idx_config['fallback_encoding'],
        supported_extensions=idx_config['supported_extensions']
    )
    
    # Vector Store
    vec_config = config['vectordb']
    vector_store = VectorStore(
        persist_directory=vec_config['persist_directory'],
        collection_name=vec_config['collection_name'],
        distance_metric=vec_config['distance_metric']
    )
    
    return rados_client, embedding_gen, content_processor, vector_store


@click.group()
@click.option('--config', default=DEFAULT_CONFIG, help='Path to configuration file')
@click.pass_context
def cli(ctx, config):
    """Ceph Semantic Storage - Semantic search for RADOS objects."""
    ctx.ensure_object(dict)
    ctx.obj['config_path'] = config
    ctx.obj['config'] = load_config(config)
    setup_logging(ctx.obj['config'])


@cli.command()
@click.option('--prefix', help='Only index objects with this prefix')
@click.option('--limit', type=int, help='Maximum number of objects to index')
@click.option('--force', is_flag=True, help='Force reindex of existing objects')
@click.pass_context
def index(ctx, prefix, limit, force):
    """Index objects from the Ceph pool."""
    config = ctx.obj['config']
    
    console.print("\n[bold cyan]🔍 Starting Indexing Process[/bold cyan]\n")
    
    try:
        # Create components
        rados_client, embedding_gen, content_processor, vector_store = create_components(config)
        
        # Connect to Ceph
        console.print("📡 Connecting to Ceph cluster...")
        rados_client.connect()
        console.print(f"✅ Connected to pool: [green]{rados_client.pool_name}[/green]\n")
        
        # Create indexer
        indexer = Indexer(
            rados_client=rados_client,
            embedding_generator=embedding_gen,
            content_processor=content_processor,
            vector_store=vector_store,
            batch_size=config['indexing']['batch_size']
        )
        
        # Run indexing
        stats = indexer.index_pool(
            prefix=prefix,
            limit=limit,
            force_reindex=force,
            show_progress=True
        )
        
        # Display results
        console.print("\n[bold green]✅ Indexing Complete![/bold green]\n")
        
        table = Table(title="Indexing Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Objects", str(stats.total_objects))
        table.add_row("Successfully Indexed", str(stats.successfully_indexed))
        table.add_row("Skipped", str(stats.skipped))
        table.add_row("Failed", str(stats.failed))
        table.add_row("Total Size", f"{stats.total_size_bytes / (1024**2):.2f} MB")
        table.add_row("Duration", f"{stats.duration_seconds:.2f}s")
        
        console.print(table)
        
        if stats.errors:
            console.print(f"\n[yellow]⚠ {len(stats.errors)} errors occurred[/yellow]")
            for error in stats.errors[:5]:  # Show first 5 errors
                console.print(f"  • {error}")
        
        rados_client.disconnect()
        
    except Exception as e:
        console.print(f"\n[red]❌ Error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument('query')
@click.option('--top-k', default=10, help='Number of results to return')
@click.option('--min-score', default=0.0, help='Minimum relevance score (0-1)')
@click.option('--pool', help='Filter by pool name')
@click.option('--type', 'content_type', help='Filter by content type')
@click.option('--content', is_flag=True, help='Include full content in results')
@click.pass_context
def search(ctx, query, top_k, min_score, pool, content_type, content):
    """Search for objects using natural language query."""
    config = ctx.obj['config']
    
    console.print(f"\n[bold cyan]🔎 Searching for:[/bold cyan] '{query}'\n")
    
    try:
        # Create components
        rados_client, embedding_gen, content_processor, vector_store = create_components(config)
        
        # Connect to Ceph
        rados_client.connect()
        
        # Create searcher
        searcher = Searcher(
            rados_client=rados_client,
            embedding_generator=embedding_gen,
            vector_store=vector_store
        )
        
        # Perform search
        results = searcher.search(
            query_text=query,
            top_k=top_k,
            min_score=min_score,
            pool_name=pool,
            content_type=content_type,
            include_content=content
        )
        
        # Display results
        if not results:
            console.print("[yellow]No results found.[/yellow]")
            return
        
        console.print(f"[green]Found {len(results)} results:[/green]\n")
        
        for i, result in enumerate(results, 1):
            console.print(f"[bold cyan]{i}. {result.object_name}[/bold cyan]")
            console.print(f"   Score: [green]{result.relevance_score:.3f}[/green]")
            console.print(f"   Pool: {result.pool_name}")
            console.print(f"   Type: {result.content_type}")
            console.print(f"   Size: {result.size_bytes / 1024:.1f} KB")
            
            if result.keywords:
                console.print(f"   Keywords: {', '.join(result.keywords[:5])}")
            
            console.print(f"   Preview: {result.content_preview[:200]}...")
            
            if content and result.full_content:
                console.print(f"\n   [dim]Full Content:[/dim]\n{result.full_content[:500]}...\n")
            
            console.print()
        
        rados_client.disconnect()
        
    except Exception as e:
        console.print(f"\n[red]❌ Error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option('--duration', type=int, help='Watch duration in seconds (default: infinite)')
@click.option('--daemon', is_flag=True, help='Run as daemon process')
@click.pass_context
def watch(ctx, duration, daemon):
    """Watch pool for changes and auto-index new objects."""
    config = ctx.obj['config']
    
    console.print("\n[bold cyan]👁 Starting Watcher Service[/bold cyan]\n")
    
    try:
        # Create components
        rados_client, embedding_gen, content_processor, vector_store = create_components(config)
        
        # Connect to Ceph
        console.print("📡 Connecting to Ceph cluster...")
        rados_client.connect()
        console.print(f"✅ Watching pool: [green]{rados_client.pool_name}[/green]\n")
        
        # Create indexer and watcher
        indexer = Indexer(
            rados_client=rados_client,
            embedding_generator=embedding_gen,
            content_processor=content_processor,
            vector_store=vector_store
        )
        
        watcher_config = config.get('watcher', {})
        watcher = Watcher(
            rados_client=rados_client,
            indexer=indexer,
            poll_interval=watcher_config.get('poll_interval_seconds', 60)
        )
        
        console.print(f"⏱ Poll interval: {watcher.poll_interval} seconds")
        console.print("Press Ctrl+C to stop\n")
        
        # Run watcher
        if daemon:
            log_file = watcher_config.get('log_file', './watcher.log')
            watcher.watch_daemon(log_file=log_file)
        else:
            watcher.watch(duration=duration)
        
        rados_client.disconnect()
        
    except KeyboardInterrupt:
        console.print("\n[yellow]⏹ Watcher stopped by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]❌ Error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.pass_context
def stats(ctx):
    """Display system statistics."""
    config = ctx.obj['config']
    
    console.print("\n[bold cyan]📊 System Statistics[/bold cyan]\n")
    
    try:
        # Create components
        rados_client, embedding_gen, content_processor, vector_store = create_components(config)
        
        # Connect to Ceph
        rados_client.connect()
        
        # Get cluster stats
        cluster_stats = rados_client.get_cluster_stats()
        pools = rados_client.list_pools()
        
        # Get vector store stats
        vec_stats = vector_store.get_stats()
        
        # Display Ceph stats
        table = Table(title="Ceph Cluster")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Space", f"{cluster_stats['kb'] / (1024**2):.2f} GB")
        table.add_row("Used Space", f"{cluster_stats['kb_used'] / (1024**2):.2f} GB")
        table.add_row("Available Space", f"{cluster_stats['kb_avail'] / (1024**2):.2f} GB")
        table.add_row("Total Objects", str(cluster_stats['num_objects']))
        table.add_row("Pools", ", ".join(pools))
        
        console.print(table)
        console.print()
        
        # Display vector store stats
        table2 = Table(title="Vector Store")
        table2.add_column("Metric", style="cyan")
        table2.add_column("Value", style="green")
        
        table2.add_row("Collection", vec_stats['collection_name'])
        table2.add_row("Indexed Objects", str(vec_stats['total_objects']))
        table2.add_row("Distance Metric", vec_stats['distance_metric'])
        table2.add_row("Indexed Pools", ", ".join(vec_stats.get('pools', [])))
        table2.add_row("Embedding Models", ", ".join(vec_stats.get('embedding_models', [])))
        
        console.print(table2)
        console.print()
        
        # Display model info
        model_info = embedding_gen.get_model_info()
        table3 = Table(title="Embedding Model")
        table3.add_column("Property", style="cyan")
        table3.add_column("Value", style="green")
        
        for key, value in model_info.items():
            table3.add_row(key.replace('_', ' ').title(), str(value))
        
        console.print(table3)
        
        rados_client.disconnect()
        
    except Exception as e:
        console.print(f"\n[red]❌ Error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument('object_name')
@click.option('--top-k', default=5, help='Number of similar objects to find')
@click.pass_context
def similar(ctx, object_name, top_k):
    """Find objects similar to a given object."""
    config = ctx.obj['config']
    
    console.print(f"\n[bold cyan]🔍 Finding similar objects to:[/bold cyan] {object_name}\n")
    
    try:
        # Create components
        rados_client, embedding_gen, content_processor, vector_store = create_components(config)
        
        # Connect to Ceph
        rados_client.connect()
        
        # Create searcher
        searcher = Searcher(
            rados_client=rados_client,
            embedding_generator=embedding_gen,
            vector_store=vector_store
        )
        
        # Find similar objects
        results = searcher.find_similar(object_name, top_k=top_k)
        
        if not results:
            console.print("[yellow]No similar objects found.[/yellow]")
            return
        
        console.print(f"[green]Found {len(results)} similar objects:[/green]\n")
        
        for i, result in enumerate(results, 1):
            console.print(f"[bold cyan]{i}. {result.object_name}[/bold cyan]")
            console.print(f"   Similarity: [green]{result.relevance_score:.3f}[/green]")
            console.print(f"   Preview: {result.content_preview[:150]}...")
            console.print()
        
        rados_client.disconnect()
        
    except Exception as e:
        console.print(f"\n[red]❌ Error: {e}[/red]")
        sys.exit(1)


if __name__ == '__main__':
    cli(obj={})
