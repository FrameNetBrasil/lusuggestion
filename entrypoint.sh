#!/bin/bash
# FrameNet LU Generator API Entrypoint Script
# Sets up environment and initializes the container

set -e

# Default environment variables
export MODEL_PATH=${MODEL_PATH:-"/srv/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"}
export PORT=${PORT:-5000}
export HOST=${HOST:-"0.0.0.0"}
export PYTHONPATH=${PYTHONPATH:-"/opt/app"}

# Create models directory if it doesn't exist
mkdir -p "$(dirname "$MODEL_PATH")"

echo "=================================="
echo "FrameNet LU Generator API Starting"
echo "=================================="
echo "Model Path: $MODEL_PATH"
echo "Port: $PORT"
echo "Host: $HOST"
echo "Python Path: $PYTHONPATH"
echo "=================================="

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Warning: Model file not found at $MODEL_PATH"
    echo "The application will attempt to download it on first request if --download is used"
fi

# Check GPU availability
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
    echo "=================================="
else
    echo "No GPU detected, running on CPU"
    echo "=================================="
fi

# Verify Python dependencies
echo "Verifying Python dependencies..."
python -c "import flask; print(f'Flask: {flask.__version__}')"
python -c "import llama_cpp; print(f'llama-cpp-python: {llama_cpp.__version__}')"
python -c "import huggingface_hub; print(f'huggingface_hub: {huggingface_hub.__version__}')"
echo "Dependencies verified successfully"
echo "=================================="

# Execute the command passed to the container
exec "$@"