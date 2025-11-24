#!/bin/bash
# Setup script for LLM Agent with Ollama

set -e

echo "🚀 Setting up LLM Agent for Semantic Ceph Storage"
echo "=================================================="
echo

# Check if running in virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Warning: Not in a virtual environment"
    echo "Please run: source venv/bin/activate"
    exit 1
fi

# Install new Python dependencies
echo "📦 Installing Python dependencies..."
grep -v "^rados" requirements.txt > /tmp/requirements_venv.txt
pip install -q -r /tmp/requirements_venv.txt
echo "✅ Python dependencies installed"
echo

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama is not installed"
    echo
    echo "To install Ollama:"
    echo "  curl -fsSL https://ollama.com/install.sh | sh"
    echo
    echo "Or visit: https://ollama.com/download"
    exit 1
fi

echo "✅ Ollama is installed"
echo

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags &> /dev/null; then
    echo "⚠️  Ollama service is not running"
    echo "Starting Ollama in the background..."
    ollama serve &
    sleep 3
fi

echo "✅ Ollama service is running"
echo

# Pull recommended model
MODEL="llama3.2"
echo "📥 Pulling recommended model: $MODEL"
echo "This may take a few minutes on first run..."
ollama pull $MODEL

echo
echo "✅ Model $MODEL is ready"
echo

# Test the setup
echo "🧪 Testing the setup..."
python3 -c "
from core.llm_provider import OllamaProvider
provider = OllamaProvider(model='$MODEL')
response = provider.complete('Say hello in one word')
print(f'✅ Test successful! Model response: {response}')
"

echo
echo "=================================================="
echo "🎉 Setup Complete!"
echo
echo "You can now use the natural language interface:"
echo "  ./run.sh chat              # Interactive chat mode"
echo "  ./run.sh execute \"search for files about greetings\""
echo
echo "Available models:"
ollama list
echo
echo "To change models, edit config.yaml:"
echo "  llm:"
echo "    model: llama3.2  # or llama3.1, mistral, qwen2.5, etc."
echo
