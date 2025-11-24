# 🚀 Quick Reference - LLM Agent Commands

## Setup (One-Time)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Setup LLM Agent
cd /users/vlad777/research/semantic-ceph-llm
source venv/bin/activate
./setup_llm.sh
```

## Running Commands

```bash
# Interactive Chat Mode (Recommended)
./run.sh chat

# One-Shot Command
./run.sh execute "your command here"

# With auto-confirmation (skip prompts)
./run.sh execute "delete test.txt" --auto-confirm
```

## Common Commands

### 🔍 Search & Discovery
```
search for files about greetings
find all Python files
show me documents containing configuration
what files do we have?
```

### 📄 Read & Inspect
```
read test.txt
show me the content of welcome.txt
what's in document xyz?
show me metadata for test.txt
```

### ✏️ Create & Modify
```
create test.txt with "Hello World"
make a file called notes.txt with my meeting notes
update test.txt with new content
append more text to notes.txt
```

### 🗑️ Delete
```
delete test.txt
remove welcome.txt
```
*(Will ask for confirmation)*

### 📊 Statistics & Analysis
```
show me storage statistics
how many files are indexed?
find files similar to test.txt
list all objects
```

### 🔄 Indexing
```
index all new files
make test.txt searchable
reindex everything
```

## Chat Mode Tips

```bash
./run.sh chat

# Clear conversation history
You: clear

# Exit
You: exit
# or: quit, q, or Ctrl+C
```

## Configuration Quick Edit

```bash
# Edit config
nano config.yaml

# Key settings:
llm:
  agent_enabled: true          # Enable/disable agent
  provider: ollama             # ollama or openai
  model: llama3.2              # Model to use
  temperature: 0.1             # 0=focused, 1=creative
```

## Model Management (Ollama)

```bash
# List installed models
ollama list

# Pull a new model
ollama pull llama3.1
ollama pull mistral
ollama pull qwen2.5

# Remove a model
ollama rm modelname

# Check Ollama status
curl http://localhost:11434/api/tags
```

## Troubleshooting

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve

# View logs
tail -f semantic_storage.log

# Test basic functionality
./run.sh execute "show stats"
```

## Example Workflow

```bash
# 1. Start chat
./run.sh chat

# 2. Explore
You: what files do we have?
You: search for configuration files

# 3. Read
You: show me the first one

# 4. Create
You: create summary.txt with "Project summary goes here"

# 5. Find related
You: what files are similar to summary.txt?

# 6. Stats
You: show me storage statistics

# 7. Exit
You: exit
```

## File Locations

```
Core Components:
  core/llm_agent.py          - Main agent logic
  core/llm_provider.py       - Ollama/OpenAI providers
  core/intent_schema.py      - Operation definitions
  core/tool_registry.py      - Available functions

Configuration:
  config.yaml                - Main configuration
  
Documentation:
  AGENT_GUIDE.md             - Complete guide
  IMPLEMENTATION_SUMMARY.md  - Technical details
  QUICK_REFERENCE.md         - This file
```

## Support

- **Full Guide**: See [AGENT_GUIDE.md](AGENT_GUIDE.md)
- **Logs**: `tail -f semantic_storage.log`
- **Debug Mode**: Set `logging.level: DEBUG` in config.yaml

---

**Remember**: The agent understands natural language - just describe what you want to do!
