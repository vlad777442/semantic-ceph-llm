# Ceph Semantic Storage

A semantic file system interface for Ceph/RADOS that enables natural language search over stored objects using vector embeddings and semantic indexing.

## 🎯 Overview

This project implements a **semantic object storage layer** on top of Ceph RADOS, inspired by LSFS (LLM-based Semantic File System). It provides:

- **🤖 Natural Language Interface**: Control your storage using plain English (NEW!)
- **Semantic Indexing**: Automatically extract content and generate embeddings for RADOS objects
- **Natural Language Search**: Query objects using natural language instead of exact filenames
- **Vector Similarity**: Find semantically similar documents
- **Automatic Monitoring**: Watch for new/modified objects and auto-index them
- **Metadata Extraction**: Generate summaries, keywords, and tags (extensible to LLM-based)
- **CRUD Operations**: Create, read, update, delete via natural language or CLI

## 🆕 NEW: LLM Agent

Interact with your Ceph storage using natural language!

```bash
# Interactive chat mode
./run.sh chat

You: search for files about greetings
You: create a file called welcome.txt with "Hello World"
You: show me storage statistics

# One-shot commands
./run.sh execute "find all Python files"
./run.sh execute "delete test.txt"
```

**See [AGENT_GUIDE.md](AGENT_GUIDE.md) for complete documentation.**


## 🏗️ Architecture

```
┌─────────────────┐
│   CLI / API     │  ← User Interface
└────────┬────────┘
         │
┌────────┴────────┐
│   Services      │
│  - Indexer      │  ← Orchestration Layer
│  - Searcher     │
│  - Watcher      │
└────────┬────────┘
         │
┌────────┴────────┐
│   Core          │
│  - RADOS Client │  ← Component Layer
│  - Embeddings   │
│  - Vector Store │
│  - Content Proc │
└────────┬────────┘
         │
┌────────┴────────┐
│  Storage Layer  │
│  - Ceph/RADOS   │  ← Backend
│  - ChromaDB     │
└─────────────────┘
```

### Components

**Core Layer:**
- `rados_client.py`: Interface to Ceph RADOS for object operations
- `embedding_generator.py`: Generate vector embeddings using sentence-transformers or OpenAI
- `content_processor.py`: Extract and preprocess text from various file types
- `vector_store.py`: ChromaDB interface for storing and querying embeddings
- `metadata_schema.py`: Pydantic models for type-safe metadata handling

**Services Layer:**
- `indexer.py`: Scan pool, extract content, generate embeddings, store in vector DB
- `searcher.py`: Natural language search and similarity queries
- `watcher.py`: Monitor pool for changes and auto-index

**Interface Layer:**
- `cli.py`: Command-line interface for all operations

## 📋 Data Schema

```python
ObjectMetadata:
  - object_id: str              # SHA256(pool:object_name)
  - object_name: str            # RADOS object key
  - pool_name: str              # Ceph pool
  - content_type: str           # MIME type
  - size_bytes: int
  - encoding: str
  - created_at: datetime
  - modified_at: datetime
  - indexed_at: datetime
  - content_preview: str        # First 500 chars
  - full_text: str (optional)   # For small files
  - embedding_model: str        # Model used
  - embedding_dimensions: int
  - summary: str (optional)     # LLM-generated
  - keywords: List[str]
  - tags: List[str]
  - is_chunked: bool           # For large files
  - metadata: Dict             # Custom fields
```

## 🚀 Installation

### Prerequisites

1. **Ceph Cluster**: A running Ceph cluster with RADOS access
2. **Python**: Python 3.8+
3. **System Dependencies**:
   ```bash
   sudo apt-get update
   sudo apt-get install -y python3-rados python3-dev libmagic1
   ```

### Setup

1. **Clone and navigate**:
   ```bash
   cd /path/to/semantic-ceph-llm
   ```

2. **Install system dependencies**:
   ```bash
   sudo apt-get update
   sudo apt-get install -y python3-rados python3-venv python3-dev libmagic1 ceph-common
   ```

3. **Create virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. **Install Python dependencies**:
   ```bash
   # Create a requirements file without rados (it's system-wide)
   grep -v "^rados" requirements.txt > requirements_venv.txt
   pip install -r requirements_venv.txt
   
   # Link the system rados module to the virtual environment
   ln -s /usr/lib/python3/dist-packages/rados*.so venv/lib/python3.*/site-packages/
   ```

5. **Configure**:
   Edit `config.yaml` to match your setup:
   ```yaml
   ceph:
     config_file: /etc/ceph/ceph.conf
     pool_name: cephfs.cephfs.data  # Your pool name
   ```

6. **Verify Ceph access**:
   ```bash
   sudo ceph -s
   sudo ceph osd pool ls
   ```

## 📖 Usage

### Running Commands

Since Ceph keyring files require root permissions, you need to run commands with sudo while using the virtual environment's Python:

```bash
# Option 1: Use the convenience script
./run.sh index

# Option 2: Manually activate and run with sudo
source venv/bin/activate
sudo venv/bin/python cli.py index
```

### Index Objects

Index all objects in your Ceph pool:

```bash
./run.sh index
```

Options:
- `--prefix PREFIX`: Only index objects with specific prefix
- `--limit N`: Limit to N objects
- `--force`: Reindex existing objects

Examples:
```bash
# Index only Python files
./run.sh index --prefix "python/"

# Index first 100 objects
./run.sh index --limit 100

# Force reindex all
./run.sh index --force
```

### Search Objects

Search using natural language:

```bash
./run.sh search "machine learning algorithms"
```

Options:
- `--top-k N`: Number of results (default: 10)
- `--min-score SCORE`: Minimum relevance score 0-1 (default: 0.0)
- `--pool POOL`: Filter by pool
- `--type TYPE`: Filter by content type
- `--content`: Include full content

Examples:
```bash
# Find configuration files
./run.sh search "yaml configuration files" --top-k 5

# Search with threshold
sudo python3 cli.py search "database schema" --min-score 0.7

# Get full content
sudo python3 cli.py search "error handling" --content
```

### Find Similar Objects

Find objects similar to a known object:

```bash
sudo python3 cli.py similar example.py --top-k 5
```

### Watch for Changes

Monitor pool and auto-index new/modified objects:

```bash
sudo python3 cli.py watch
```

Options:
- `--duration SECONDS`: Watch for specific duration
- `--daemon`: Run as background daemon

Examples:
```bash
# Watch for 1 hour
sudo python3 cli.py watch --duration 3600

# Run as daemon
sudo python3 cli.py watch --daemon
```

### View Statistics

Display system statistics:

```bash
sudo python3 cli.py stats
```

Shows:
- Ceph cluster statistics
- Indexed objects count
- Vector store information
- Embedding model details

## 🔬 Academic Research Usage

This system is designed for research on semantic file systems and can be used to:

### Research Questions

1. **Semantic Search Effectiveness**:
   - Compare semantic search vs traditional metadata search
   - Measure precision/recall across different domains
   - Analyze query-document relevance

2. **Embedding Model Comparison**:
   - Test different embedding models (MiniLM, MPNet, multilingual)
   - Compare local vs cloud embeddings (OpenAI)
   - Measure trade-offs: speed vs accuracy

3. **Storage System Integration**:
   - Performance impact of semantic indexing
   - Scalability analysis (objects, pool size)
   - Integration patterns with existing systems

4. **LLM-Enhanced Metadata**:
   - Auto-summarization quality
   - Keyword extraction effectiveness
   - Tag generation accuracy

### Data Collection

The system logs comprehensive metrics for analysis:

```python
# Indexing metrics
- Objects indexed per second
- Embedding generation time
- Storage overhead
- Error rates

# Search metrics  
- Query latency
- Relevance scores
- Result ranking quality
- Cache hit rates
```

### Extensibility

Easy extension points for research:

1. **Custom Embedding Models**:
   ```python
   # In embedding_generator.py
   class CustomEmbeddingGenerator(EmbeddingGenerator):
       def encode(self, texts):
           # Your custom model
           pass
   ```

2. **LLM Integration**:
   ```python
   # Add to indexer.py
   def generate_summary(text):
       # Call GPT-4, Llama, etc.
       return summary
   ```

3. **Re-ranking Algorithms**:
   ```python
   # In searcher.py
   def rerank_results(query, results):
       # Custom re-ranking logic
       return reranked_results
   ```

## 🔧 Configuration

### Embedding Models

**Local Models** (sentence-transformers):
- `all-MiniLM-L6-v2`: Fast, 384 dims (default)
- `all-mpnet-base-v2`: Better quality, 768 dims
- `paraphrase-multilingual-MiniLM-L12-v2`: Multilingual support

**Cloud Models** (future):
- OpenAI `text-embedding-3-small`: 1536 dims
- OpenAI `text-embedding-3-large`: 3072 dims

Edit `config.yaml`:
```yaml
embedding:
  provider: sentence-transformers
  model: all-MiniLM-L6-v2
  device: cpu  # or cuda for GPU
```

### Performance Tuning

For large-scale indexing:

```yaml
indexing:
  batch_size: 32              # Process N objects at once
  max_file_size_mb: 100       # Skip files larger than this
  chunk_size: 1000            # Characters per chunk
  parallel_processing: true   # Enable parallelization

embedding:
  batch_size: 64              # Encode N texts at once
  device: cuda                # Use GPU if available

cache:
  enable_embedding_cache: true
  max_cache_size_mb: 1000
```

## 📊 Performance

Typical performance on commodity hardware (4-core CPU, 16GB RAM):

- **Indexing**: ~50-100 objects/minute
- **Search**: <100ms per query
- **Embedding generation**: ~20ms per object (CPU), ~5ms (GPU)
- **Storage overhead**: ~1-2KB per object (metadata + embedding)

## 🔐 Security

**Important**: This system requires root/sudo access to read Ceph keyrings. For production:

1. Create a dedicated Ceph client with limited permissions:
   ```bash
   ceph auth get-or-create client.semantic mon 'allow r' osd 'allow r pool=your-pool'
   ```

2. Update `config.yaml`:
   ```yaml
   ceph:
     client_name: client.semantic
   ```

3. Set appropriate permissions:
   ```bash
   sudo chown semantic-user:semantic-user /etc/ceph/ceph.client.semantic.keyring
   sudo chmod 600 /etc/ceph/ceph.client.semantic.keyring
   ```

## 🤝 Contributing

Research contributions welcome! Areas of interest:

- Novel embedding models for file content
- LLM-based metadata extraction
- Advanced re-ranking algorithms
- Multi-modal embeddings (code + docs)
- Distributed indexing for large clusters

## 📝 Citation

If you use this work in academic research, please cite:

```bibtex
@software{ceph_semantic_storage,
  title = {Ceph Semantic Storage: Semantic Search for RADOS Objects},
  author = {Your Name},
  year = {2025},
  note = {Research prototype for semantic file systems}
}
```

## 📚 References

- **LSFS**: LLM-based Semantic File System concepts
- **Ceph**: https://docs.ceph.com/
- **Sentence Transformers**: https://www.sbert.net/
- **ChromaDB**: https://docs.trychroma.com/

## 🐛 Troubleshooting

### Common Issues

**1. Permission Denied (RADOS)**:
```bash
# Run with sudo
sudo python3 cli.py command

# Or fix keyring permissions
sudo chown $USER /etc/ceph/ceph.client.admin.keyring
```

**2. Module Not Found**:
```bash
# Install dependencies
sudo pip3 install -r requirements.txt
```

**3. CUDA Out of Memory**:
```yaml
# In config.yaml, switch to CPU
embedding:
  device: cpu
```

**4. Slow Indexing**:
```yaml
# Reduce file size limit
indexing:
  max_file_size_mb: 10
  
# Use GPU if available
embedding:
  device: cuda
  batch_size: 64
```

## 📄 License

MIT License - See LICENSE file

## 🎓 Academic Context

This project serves as a research platform for exploring:
- Semantic file system architectures
- Vector database integration with object storage
- Natural language interfaces for data retrieval
- LLM applications in storage systems
- Scalable indexing strategies

Perfect for:
- Master's/PhD research projects
- Systems architecture papers
- HCI studies on search interfaces
- Performance analysis papers
- Comparative studies of embedding models

---

**Note**: This is a research prototype. For production use, additional hardening, security audits, and performance optimization are recommended.
