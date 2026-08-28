#!/bin/bash
# Run SATELLITE AI Backend
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "  SATELLITE AI — Starting Backend"
echo "========================================"

# Create required directories
mkdir -p data/demo data/input data/output models

# Run the server
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload --log-level info
