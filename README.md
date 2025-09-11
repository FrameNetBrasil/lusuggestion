 Basic Usage:

  # With automatic download (recommended)
  python framenet_lu_generator_llama.py parsed_prompt.txt --download

  # Using specific model path (if already downloaded)
  python framenet_lu_generator_llama.py parsed_prompt.txt --model-path "./models/models--bartowski--Meta-Llama-3.1-8B-Instruct-GGUF/snapshots/bf5b95e96dac0462e2a09145ec66cae9a3f12067/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

  Advanced Options:

  # With custom parameters
  python framenet_lu_generator_llama.py parsed_prompt.txt --download \
      --temperature 0.1 \
      --max-tokens 2048 \
      --output custom_results.json

  # Different model quantization
  python framenet_lu_generator_llama.py parsed_prompt.txt --download \
      --model-file Meta-Llama-3.1-8B-Instruct-Q8_0.gguf

  # Custom cache directory
  python framenet_lu_generator_llama.py parsed_prompt.txt --download \
      --cache-dir ./my_models

  All Available Options:

  python framenet_lu_generator_llama.py <prompt_file> [options]

  Options:
    --model-path PATH              Path to local GGUF model file
    --model-file FILENAME          Specific GGUF model file (default: Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf)
    --download                     Download model from HuggingFace Hub if not found locally
    --output FILENAME              Output JSON file path (default: auto-generated)
    --temperature FLOAT            Sampling temperature (default: 0.1)
    --max-tokens INT               Maximum tokens to generate (default: 2048)
    --cache-dir PATH               Model cache directory (default: ./models)

  Model Quantization Options:

  | Model File                             | Size  | VRAM  | Quality | Speed  |
  |----------------------------------------|-------|-------|---------|--------|
  | Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf | 4.9GB | ~6GB  | Good    | Fast   |
  | Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf | 5.7GB | ~7GB  | Better  | Medium |
  | Meta-Llama-3.1-8B-Instruct-Q8_0.gguf   | 8.5GB | ~10GB | Best    | Slower |

  Successful Example (that just ran):

  python framenet_lu_generator_llama.py parsed_prompt.txt \
      --model-path "./models/models--bartowski--Meta-Llama-3.1-8B-Instruct-GGUF/snapshots/bf5b95e96dac0462e2a09145ec66cae9a3f12067/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" \
      --temperature 0.1

  The script successfully generated 17 Portuguese lexical units with proper JSON formatting, including nouns, adjectives, and verbs for the "Artefato" frame!
