# FrameNet Lexical Unit Generator

A comprehensive tool for generating Brazilian Portuguese lexical units for FrameNet frames using large language models. This project provides both command-line scripts and a web API for generating contextually appropriate lexical units.

## 🚀 Features

- **Multiple LLM Backends**: Support for Sabiá-7B, Bode-7B, and Llama 3.1-8B-Instruct models
- **Web API**: Flask-based REST API for integration with web applications
- **Docker Support**: Containerized deployment with GPU acceleration
- **Portuguese Focus**: Specialized for Brazilian Portuguese lexical units
- **FrameNet Integration**: Follows FrameNet Brasil standards and conventions

## 📋 Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM (16GB+ recommended)
- Docker and Docker Compose (for containerized deployment)

### Python Dependencies

```bash
pip install llama-cpp-python[cublas] huggingface_hub flask numpy tqdm requests
```

## 🛠️ Installation

### Local Installation

```bash
git clone https://github.com/FrameNetBrasil/lusuggestion.git
cd lusuggestion
pip install -r requirements.txt
```

### Docker Installation

```bash
git clone https://github.com/FrameNetBrasil/lusuggestion.git
cd lusuggestion
docker-compose up --build
```

## 📖 Usage

### Command Line Interface

#### Basic Usage

```bash
# With automatic model download (recommended)
python framenet_lu_generator_llama.py parsed_prompt.txt --download

# Using specific model path (if already downloaded)
python framenet_lu_generator_llama.py parsed_prompt.txt --model-path "./models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
```

#### Advanced Options

```bash
# With custom parameters
python framenet_lu_generator_llama.py parsed_prompt.txt --download \
    --temperature 0.1 \
    --max-tokens 2048 \
    --output custom_results.json

# Different model quantization
python framenet_lu_generator_llama.py parsed_prompt.txt --download \
    --model-file Meta-Llama-3.1-8B-Instruct-Q8_0.gguf
```

#### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model-path PATH` | Path to local GGUF model file | Auto-detected |
| `--model-file FILENAME` | Specific GGUF model file | Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf |
| `--download` | Download model from HuggingFace Hub | False |
| `--output FILENAME` | Output JSON file path | Auto-generated |
| `--temperature FLOAT` | Sampling temperature (0.0-1.0) | 0.1 |
| `--max-tokens INT` | Maximum tokens to generate | 2048 |
| `--cache-dir PATH` | Model cache directory | ./models |

### Web API

#### Starting the API Server

```bash
# Local development
python app.py --model-path /path/to/model.gguf --port 5000

# With Docker
docker-compose up

# With custom configuration
python app.py --model-path /path/to/model.gguf --port 8080 --host 0.0.0.0
```

#### API Endpoints

##### `GET /` - Web Interface
Simple web interface for testing the API.

##### `GET /health` - Health Check
Returns the health status of the API and model.

**Response:**
```json
{
  "status": "healthy",
  "model_status": "loaded",
  "model_path": "/path/to/model.gguf"
}
```

##### `POST /generate` - Generate Lexical Units
Main endpoint for generating lexical units.

**Request:**
```json
{
  "frame": "Artefato",
  "frame_definition": "Um artefato feito ou modificado por uma entidade inteligente para ser destinado a um determinado tipo de Uso.",
  "target_count": 10,
  "exclusion_list": ["anilha", "arame", "aspirador"],
  "temperature": 0.1,
  "max_tokens": 2048
}
```

**Response:**
```json
{
  "frame": "Artefato",
  "total": 10,
  "items": [
    {
      "lemma": "ferramenta",
      "pos": "NOUN",
      "mwe": false,
      "gloss_pt": "objeto usado para realizar trabalho específico",
      "example_pt": "A ferramenta foi criada para facilitar o trabalho.",
      "confidence": 0.95
    }
  ]
}
```

#### API Usage Examples

##### Using curl

```bash
# Generate lexical units
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "frame": "Artefato",
    "frame_definition": "Um artefato feito ou modificado por uma entidade inteligente.",
    "target_count": 5,
    "exclusion_list": ["anilha", "arame"],
    "temperature": 0.1
  }'

# Health check
curl http://localhost:5000/health
```

##### Using Python requests

```python
import requests

# Generate lexical units
response = requests.post('http://localhost:5000/generate', json={
    "frame": "Artefato",
    "frame_definition": "Um artefato feito ou modificado por uma entidade inteligente.",
    "target_count": 10,
    "exclusion_list": ["anilha", "arame", "aspirador"],
    "temperature": 0.1
})

result = response.json()
print(f"Generated {result['total']} lexical units")
```

## 🐳 Docker Deployment

### Basic Deployment

```bash
# Build and run
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f
```

### Production Deployment

```bash
# With Nginx reverse proxy
docker-compose --profile production up -d

# Custom configuration
MODEL_PATH=/custom/path/model.gguf PORT=8080 docker-compose up
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_PATH` | Path to model file in container | /srv/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf |
| `PORT` | API server port | 5000 |
| `HOST` | API server host | 0.0.0.0 |

## 🎯 Model Options

### Supported Models

| Model | Size | Quality | Speed | VRAM | Description |
|-------|------|---------|--------|------|-------------|
| **Llama 3.1-8B-Instruct** | 4.9-8.5GB | Best | Fast-Slow | 6-10GB | Recommended, multilingual |
| **Bode-7B** | 3.8-7GB | Good | Fast | 5-8GB | Portuguese-specific |
| **Sabiá-7B** | 3.8-7GB | Fair | Fast | 5-8GB | Portuguese base model |

### Model Quantization Levels

| Quantization | File Size | Quality | Speed | Memory |
|--------------|-----------|---------|--------|--------|
| Q4_K_M | ~5GB | Good | Fast | ~6GB |
| Q5_K_M | ~6GB | Better | Medium | ~7GB |
| Q8_0 | ~8GB | Best | Slower | ~10GB |

## 📝 API Error Handling

The API returns appropriate HTTP status codes and error messages:

- `200 OK` - Successful generation
- `400 Bad Request` - Invalid input parameters
- `404 Not Found` - Endpoint not found
- `500 Internal Server Error` - Model or generation error

Example error response:
```json
{
  "error": "Temperature must be between 0.0 and 1.0"
}
```

## 🔧 Configuration

### Model Configuration

Models are automatically downloaded from HuggingFace Hub when using the `--download` flag. The default model repository is `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF`.

### Generation Parameters

- **Temperature**: Controls randomness (0.0 = deterministic, 1.0 = very random)
- **Max Tokens**: Maximum number of tokens to generate
- **Target Count**: Number of lexical units to generate
- **Exclusion List**: Existing lexical units to avoid

## 🚀 Performance Optimization

### GPU Acceleration

The application automatically uses GPU acceleration when available:
- CUDA-compatible GPU recommended
- Install `llama-cpp-python[cublas]` for GPU support
- Docker containers include GPU support via nvidia-docker

### Memory Management

- Use appropriate quantization levels based on available VRAM
- Q4_K_M recommended for most use cases (good quality/speed balance)
- Q8_0 for maximum quality with sufficient VRAM

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- FrameNet Brasil team for linguistic expertise
- Meta for Llama 3.1 model
- bartowski for GGUF model quantizations
- Hugging Face for model hosting and tools

## 📧 Contact

For questions and support:
- GitHub Issues: [lusuggestion/issues](https://github.com/FrameNetBrasil/lusuggestion/issues)
- FrameNet Brasil: framenetbr@gmail.com

---

**FrameNet Brasil** | Advancing Portuguese computational linguistics through AI-powered lexical unit generation.