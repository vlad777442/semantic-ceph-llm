#!/bin/bash
# Helper script to run CLI commands with proper permissions and virtual environment

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment and run with sudo using venv's Python
source venv/bin/activate
sudo venv/bin/python cli.py "$@"
