# FrameNet LU Generator API Dockerfile
# Based on Python 3.9 with CUDA support for GPU acceleration

FROM python:3.9-slim

# Set working directory
WORKDIR /opt/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    curl \
    wget \
    git \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Install CUDA-compatible PyTorch and llama-cpp-python
# Note: This installs CPU version by default. For GPU support in container,
# you'll need nvidia/cuda base image or proper CUDA toolkit installation
RUN pip install --upgrade pip

# Install llama-cpp-python (will auto-detect CUDA if nvidia-docker is used)
RUN pip install llama-cpp-python --no-cache-dir

# Install other dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Create directory for models
RUN mkdir -p /srv/models

# Copy application code
COPY framenet_generator.py .
COPY app.py .
COPY setup.py .

# Install the package
RUN pip install -e .

# Copy entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Environment variables
ENV MODEL_PATH=/srv/models/models--bartowski--Meta-Llama-3.1-8B-Instruct-GGUF/snapshots/bf5b95e96dac0462e2a09145ec66cae9a3f12067/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
ENV PORT=5000
ENV HOST=0.0.0.0
ENV PYTHONPATH=/opt/app

# Expose port
EXPOSE 5000

# Use entrypoint script
ENTRYPOINT ["/entrypoint.sh"]

# Default command
CMD ["python", "app.py", "--download"]