# Quick Start Guide

## Installation Complete! ✅

Your semantic-ceph-llm environment is now fully configured.

## Running Commands

Because Ceph requires root permissions to read the keyring file (`/etc/ceph/ceph.client.admin.keyring`), 
you need to run commands with sudo while using the virtual environment.

### Method 1: Use the convenience script (Recommended)

```bash
cd /users/vlad777/research/semantic-ceph-llm
./run.sh <command> [options]
```

Examples:
```bash
./run.sh index --limit 10
./run.sh search "your query here"
./run.sh stats
```

### Method 2: Manual execution

```bash
cd /users/vlad777/research/semantic-ceph-llm
source venv/bin/activate
sudo venv/bin/python cli.py <command> [options]
```

## Available Commands

- `index` - Index objects from the Ceph pool
  - `--prefix TEXT` - Only index objects with this prefix
  - `--limit INTEGER` - Maximum number of objects to index
  - `--force` - Force reindex of existing objects

- `search` - Search for objects using natural language
  - `--query TEXT` - Your search query
  - `--top-k INTEGER` - Number of results to return
  - `--min-score FLOAT` - Minimum similarity score (0-1)

- `similar` - Find objects similar to a given object
  - `--object-name TEXT` - Name of the reference object
  - `--top-k INTEGER` - Number of similar objects to find

- `stats` - Display system statistics

- `watch` - Watch pool for changes and auto-index

## Important Notes

1. **Don't use `sudo python3`** - This uses system Python, not your virtual environment
2. **Use `sudo venv/bin/python`** - This uses the venv Python with sudo permissions
3. Your virtual environment is at: `/users/vlad777/research/semantic-ceph-llm/venv`
4. The embedding model downloads automatically on first run
5. Vector database is stored in: `./chroma_data`

## Testing Your Setup

Try indexing a small number of objects:

```bash
cd /users/vlad777/research/semantic-ceph-llm
./run.sh index --limit 5
```

If the pool is empty (0 objects found), you can:
1. Create test data in your Ceph pool
2. Point to a different pool in `config.yaml`
3. Use the test data creation script:
   ```bash
   source venv/bin/activate
   sudo venv/bin/python create_test_data.py
   ```

## Troubleshooting

**Error: "RADOS permission denied"**
- Solution: Make sure you're using sudo with the venv Python

**Error: "No module named 'numpy'"**
- Solution: Don't use `sudo python3`, use `sudo venv/bin/python` instead

**Error: "ImportError: attempted relative import"**
- Solution: This has been fixed in the code. Make sure you're on the latest version.

## Next Steps

1. Configure your pool in `config.yaml`
2. Index your objects: `./run.sh index`
3. Search: `./run.sh search "your query"`
4. Check stats: `./run.sh stats`

## 🆕 Try the NEW LLM Agent!

Interact with your storage using natural language:

```bash
# Setup (one-time)
./setup_llm.sh

# Interactive chat
./run.sh chat
You: search for files about greetings
You: create welcome.txt with "Hello World"
You: show me storage stats

# One-shot commands
./run.sh execute "find all text files"
```

**Complete guide**: See [AGENT_GUIDE.md](AGENT_GUIDE.md)

For more details, see README.md
