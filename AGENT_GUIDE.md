# LLM Agent - Natural Language Interface for Ceph Storage

## 🎯 Overview

The LLM Agent provides a natural language interface to interact with your Ceph storage system. Instead of remembering specific commands, you can simply describe what you want to do in plain English.

## ✨ Features

- **Natural Language Commands**: "search for files about greetings" instead of complex CLI syntax
- **Intelligent Intent Recognition**: Understands what you want to do and extracts parameters
- **Interactive Chat Mode**: Have a conversation with your storage system
- **CRUD Operations**: Create, read, update, delete objects using natural language
- **Semantic Search**: Find files by meaning, not just keywords
- **Safe Operations**: Asks for confirmation on destructive operations

## 🏗️ Architecture

```
User Input (Natural Language)
         ↓
   LLM Provider (Ollama/OpenAI)
         ↓
   Intent Classifier
         ↓
   Operation Executor
         ↓
   Ceph/Vector Store
         ↓
   Natural Language Response
```

## 📋 Prerequisites

1. **Ollama** (for local LLM) or **OpenAI API** key
2. Existing Ceph semantic storage setup
3. Python 3.8+

## 🚀 Installation

### Option 1: Ollama (Local, Private, Free)

1. **Install Ollama**:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```

2. **Run setup script**:
   ```bash
   cd /users/vlad777/research/semantic-ceph-llm
   source venv/bin/activate
   ./setup_llm.sh
   ```

   This will:
   - Install Python dependencies
   - Check Ollama installation
   - Pull the recommended model (llama3.2)
   - Test the setup

### Option 2: OpenAI API

1. **Get API key** from https://platform.openai.com/

2. **Update config.yaml**:
   ```yaml
   llm:
     provider: openai
     model: gpt-4-turbo-preview
     api_key: your-api-key-here  # or use environment variable
   ```

3. **Install dependencies**:
   ```bash
   source venv/bin/activate
   pip install openai
   ```

## 📖 Usage

### Interactive Chat Mode

The most intuitive way to interact with your storage:

```bash
./run.sh chat
```

Example conversation:
```
You: search for files about greetings
Assistant: Found 2 objects matching 'greetings':
1. 10000000000.00000000 (score: 0.92)
2. 10000000001.00000000 (score: 0.88)

You: show me the content of the first one
Assistant: Content of '10000000000.00000000':
Hello World

You: create a file called welcome.txt with "Welcome to Ceph!"
Assistant: Created object 'welcome.txt' (19 bytes) and indexed for search

You: how many files are in the pool?
Assistant: Pool 'cephfs.cephfs.data' contains 3 objects (3 indexed)

You: delete welcome.txt
⚠️ This operation requires confirmation
Proceed? (yes/no): yes
Assistant: Deleted object 'welcome.txt'
```

### One-Shot Commands

Execute a single command without entering chat mode:

```bash
./run.sh execute "search for configuration files"
./run.sh execute "create test.txt with hello world"
./run.sh execute "show me storage statistics"
./run.sh execute "list all objects"
```

### Examples of Natural Language Commands

**Search Operations**:
- "find files about machine learning"
- "search for python scripts"
- "show me documents containing greetings"

**Read Operations**:
- "read the file test.txt"
- "show me the content of welcome.txt"
- "what's in document xyz?"
- "list all objects"
- "show me storage stats"

**Create Operations**:
- "create a file called notes.txt with my meeting notes"
- "write hello world to test.txt"
- "make a new document called readme.md"

**Update Operations**:
- "update test.txt with new content"
- "append more text to notes.txt"

**Delete Operations** (requires confirmation):
- "delete the file test.txt"
- "remove welcome.txt"

**Index Operations**:
- "index all new files"
- "reindex everything"
- "make test.txt searchable"

**Analysis Operations**:
- "find files similar to test.txt"
- "show me metadata for document xyz"
- "what files are related to test.txt?"

## ⚙️ Configuration

Edit [`config.yaml`](config.yaml ):

```yaml
llm:
  # Enable/disable agent
  agent_enabled: true
  
  # Provider: ollama or openai
  provider: ollama
  
  # Model selection
  model: llama3.2  # Ollama models: llama3.2, llama3.1, mistral, qwen2.5
                   # OpenAI models: gpt-4-turbo-preview, gpt-3.5-turbo
  
  # LLM parameters
  temperature: 0.1  # Lower = more focused (0.0-1.0)
  max_tokens: 2000
  
  # Ollama configuration
  ollama_host: http://localhost:11434
  
  # OpenAI configuration
  # openai_api_key: ${OPENAI_API_KEY}
```

## 🤖 Supported Models

### Ollama (Local)

| Model | Size | Best For | Speed |
|-------|------|----------|-------|
| llama3.2 | 2GB | Balanced performance | Fast |
| llama3.1 | 4.7GB | Better accuracy | Medium |
| mistral | 4.1GB | Coding tasks | Fast |
| qwen2.5 | 4.7GB | Multilingual | Medium |
| mixtral | 26GB | Complex reasoning | Slow |

Pull a model:
```bash
ollama pull llama3.2
ollama pull mistral
```

List installed models:
```bash
ollama list
```

### OpenAI (API)

| Model | Cost/1M tokens | Best For |
|-------|----------------|----------|
| gpt-4-turbo | $10/$30 | Complex tasks |
| gpt-3.5-turbo | $0.50/$1.50 | Fast, cheap |

## 🔧 Advanced Usage

### Custom Operations

The agent supports these operation types:

- **SEMANTIC_SEARCH**: Natural language search
- **FIND_SIMILAR**: Find related documents
- **READ_OBJECT**: Read file content
- **LIST_OBJECTS**: List all objects
- **CREATE_OBJECT**: Create new file
- **UPDATE_OBJECT**: Modify existing file
- **DELETE_OBJECT**: Delete file
- **INDEX_OBJECT**: Index for search
- **BATCH_INDEX**: Index multiple files
- **GET_STATS**: Storage statistics
- **GET_METADATA**: File information

### Conversation Context

The agent remembers your recent conversation (last 10 messages by default):

```bash
./run.sh chat --history-size 20  # Keep 20 messages
```

Clear history during chat:
```
You: clear
```

### Auto-Confirmation

Skip confirmation prompts (use carefully!):

```bash
./run.sh execute "delete test.txt" --auto-confirm
```

## 🔒 Security Considerations

1. **Destructive Operations**: Always require confirmation (delete, bulk operations)
2. **Input Validation**: All inputs are sanitized before execution
3. **Local LLM**: With Ollama, your data never leaves your infrastructure
4. **API Keys**: Use environment variables for OpenAI keys

## 📊 Performance

**Ollama (Local)**:
- First query: 2-5 seconds (model loading)
- Subsequent queries: 0.5-2 seconds
- No cost, unlimited usage
- Requires: 4-8GB RAM, optional GPU

**OpenAI API**:
- Query: 0.5-3 seconds
- Cost: ~$0.01-0.03 per query
- No local resources needed

## 🐛 Troubleshooting

### "Ollama not found"
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Verify installation
ollama --version
```

### "Connection refused"
```bash
# Start Ollama service
ollama serve

# Or run in background
nohup ollama serve > /dev/null 2>&1 &
```

### "Model not found"
```bash
# Pull the model
ollama pull llama3.2

# List available models
ollama list
```

### "OpenAI API error"
```bash
# Check API key
echo $OPENAI_API_KEY

# Or set in config.yaml
llm:
  openai_api_key: sk-...
```

### Poor response quality
```bash
# Try a better model
ollama pull llama3.1

# Or increase temperature for creativity
llm:
  temperature: 0.3  # Higher = more creative
```

## 📚 Examples

### Complete Workflow

```bash
# Start chat
./run.sh chat

# Search for files
You: find all text files

# Read a specific file
You: show me the first one

# Create a new file
You: create summary.txt with "Project summary: ..."

# Find related documents
You: what files are similar to summary.txt?

# Get statistics
You: how many files do we have indexed?

# Index new files
You: index all new files

# Clean up
You: delete old test files
```

### Batch Operations

```bash
# Index all Python files
./run.sh execute "index all files with .py extension"

# List files by prefix
./run.sh execute "list all files starting with test"

# Get storage overview
./run.sh execute "show me storage statistics and indexed objects count"
```

## 🔄 Switching Between Ollama and OpenAI

You can easily switch providers:

1. **Edit config.yaml**:
   ```yaml
   llm:
     provider: openai  # Change from ollama
     model: gpt-4-turbo-preview
     openai_api_key: ${OPENAI_API_KEY}
   ```

2. **No code changes needed** - the agent automatically uses the configured provider

## 📖 Next Steps

1. Try the interactive chat mode: `./run.sh chat`
2. Experiment with different models
3. Create custom workflows
4. Integrate into your scripts using execute mode

## 🆘 Getting Help

In chat mode, you can ask:
- "what can you do?"
- "help me search for files"
- "how do I create a file?"

For technical help:
- Check logs: `tail -f semantic_storage.log`
- Enable debug mode in config.yaml: `logging.level: DEBUG`
- Review this guide: `AGENT_GUIDE.md`

## 🎓 Tips

1. **Be specific**: "create welcome.txt with hello" works better than "make a file"
2. **Use natural language**: The agent understands context and synonyms
3. **Confirm deletions**: Always review before confirming destructive operations
4. **Start with search**: Search is the most useful feature for discovery
5. **Try different phrasings**: If one doesn't work, rephrase your request
