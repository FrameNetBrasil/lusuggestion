#!/usr/bin/env python3
"""
FrameNet Lexical Unit Generator using Llama 3.1-8B-Instruct GGUF

This script generates Brazilian Portuguese lexical units for FrameNet frames
using the Llama 3.1-8B-Instruct model in GGUF format with GPU acceleration.
Llama 3.1-8B-Instruct is a powerful multilingual instruction-tuned model with 8K context.

Usage:
    python framenet_lu_generator_llama.py <prompt_file> [options]

Requirements:
    pip install llama-cpp-python[cublas] huggingface_hub

Author: Generated for FrameNet Brasil extension
"""

import argparse
import json
import sys
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import time

try:
    from llama_cpp import Llama
    from huggingface_hub import hf_hub_download
except ImportError as e:
    print(f"Error importing required libraries: {e}")
    print("Please install required packages:")
    print("pip install llama-cpp-python[cublas] huggingface_hub")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class LlamaFrameNetGenerator:
    """FrameNet Lexical Unit Generator using Llama 3.1-8B-Instruct GGUF"""
    
    def __init__(self, model_path: str, **kwargs):
        """
        Initialize the generator with Llama 3.1-8B-Instruct model
        
        Args:
            model_path: Path to the GGUF model file
            **kwargs: Additional parameters for Llama model
        """
        self.model_path = model_path
        self.model = None
        self.default_params = {
            'n_gpu_layers': -1,  # Use all GPU layers
            'n_ctx': 8192,       # Larger context window for Llama 3.1
            'n_batch': 512,      # Batch size
            'verbose': False,
            'seed': 42,          # For reproducible results
        }
        self.default_params.update(kwargs)
        
    def load_model(self):
        """Load the Llama 3.1-8B-Instruct GGUF model"""
        try:
            logger.info(f"Loading Llama 3.1-8B-Instruct model from: {self.model_path}")
            self.model = Llama(
                model_path=self.model_path,
                **self.default_params
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def generate_response(self, prompt: str, generation_params: Optional[Dict] = None) -> str:
        """
        Generate response using the loaded model
        
        Args:
            prompt: The input prompt
            generation_params: Generation parameters
            
        Returns:
            Generated text response
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        default_gen_params = {
            'max_tokens': 2048,      # Sufficient for JSON response
            'temperature': 0.1,      # Low temperature for structured output
            'top_p': 0.9,
            'top_k': 40,
            'repeat_penalty': 1.05,  # Minimal repetition penalty
            'stop': ["<|eot_id|>", "<|end_of_text|>"],  # Llama 3.1 stop tokens
        }
        
        if generation_params:
            default_gen_params.update(generation_params)
        
        try:
            logger.info("Generating response...")
            start_time = time.time()
            
            response = self.model(
                prompt,
                **default_gen_params
            )
            
            generation_time = time.time() - start_time
            logger.info(f"Generation completed in {generation_time:.2f} seconds")
            
            return response['choices'][0]['text'].strip()
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

def download_model(model_repo: str, model_filename: str, cache_dir: str = "./models") -> str:
    """
    Download Llama 3.1-8B-Instruct GGUF model from HuggingFace Hub
    
    Args:
        model_repo: Repository ID (e.g., "unsloth/Llama-3.1-8B-Instruct-GGUF")
        model_filename: Specific model file to download
        cache_dir: Local cache directory
        
    Returns:
        Path to downloaded model file
    """
    try:
        logger.info(f"Downloading {model_filename} from {model_repo}")
        model_path = hf_hub_download(
            repo_id=model_repo,
            filename=model_filename,
            cache_dir=cache_dir,
            resume_download=True
        )
        logger.info(f"Model downloaded to: {model_path}")
        return model_path
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise

def load_prompt_file(file_path: str) -> str:
    """
    Load prompt from file
    
    Args:
        file_path: Path to prompt file
        
    Returns:
        Prompt content as string
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        logger.info(f"Loaded prompt from: {file_path}")
        return content
    except Exception as e:
        logger.error(f"Failed to load prompt file: {e}")
        raise

def format_prompt_for_llama(base_prompt: str) -> str:
    """
    Format the prompt specifically for Llama 3.1-8B-Instruct model
    Uses the Llama 3 chat template format for optimal performance
    
    Args:
        base_prompt: The base FrameNet prompt
        
    Returns:
        Formatted prompt for Llama 3.1
    """
    # Llama 3.1 uses specific chat template format
    # formatted_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    formatted_prompt = f"""<|start_header_id|>system<|end_header_id|>

Você é um especialista em lexicografia computacional trabalhando no FrameNet Brasil. Sua tarefa é gerar unidades lexicais em português brasileiro seguindo rigorosamente as instruções fornecidas. Responda apenas com JSON válido, sem explicações adicionais.<|eot_id|><|start_header_id|>user<|end_header_id|>

{base_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    return formatted_prompt

def extract_json_from_response(response: str) -> Dict[str, Any]:
    """
    Extract and validate JSON from model response
    
    Args:
        response: Raw model response
        
    Returns:
        Parsed JSON object
    """
    try:
        # Try to find JSON in the response
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        
        if json_start == -1 or json_end == 0:
            raise ValueError("No JSON found in response")
        
        json_str = response[json_start:json_end]
        parsed_json = json.loads(json_str)
        
        logger.info("Successfully extracted and parsed JSON")
        return parsed_json
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}")
        logger.error(f"Response content: {response}")
        raise
    except Exception as e:
        logger.error(f"Error extracting JSON: {e}")
        raise

def save_output(output_data: Dict[str, Any], output_file: str):
    """
    Save output to JSON file
    
    Args:
        output_data: Data to save
        output_file: Output file path
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Output saved to: {output_file}")
    except Exception as e:
        logger.error(f"Failed to save output: {e}")
        raise

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Generate FrameNet lexical units using Llama 3.1-8B-Instruct GGUF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python framenet_lu_generator_llama.py prompt.txt
    python framenet_lu_generator_llama.py prompt.txt --model-file Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
    python framenet_lu_generator_llama.py prompt.txt --download --output results.json
        """
    )
    
    parser.add_argument(
        'prompt_file',
        help='Path to file containing the FrameNet prompt'
    )
    
    parser.add_argument(
        '--model-path',
        default=None,
        help='Path to local GGUF model file'
    )
    
    parser.add_argument(
        '--model-file',
        default='Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf',
        help='Specific GGUF model file to use (default: Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf)'
    )
    
    parser.add_argument(
        '--download',
        action='store_true',
        help='Download model from HuggingFace Hub if not found locally'
    )
    
    parser.add_argument(
        '--output',
        default=None,
        help='Output JSON file path (default: auto-generated based on input)'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.1,
        help='Sampling temperature (default: 0.1 for structured output)'
    )
    
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=4096,
        help='Maximum tokens to generate (default: 4096)'
    )
    
    parser.add_argument(
        '--cache-dir',
        default='./models',
        help='Model cache directory (default: ./models)'
    )
    
    args = parser.parse_args()
    
    try:
        # Load prompt
        prompt_content = load_prompt_file(args.prompt_file)
        
        # Determine model path
        if args.model_path:
            model_path = args.model_path
        else:
            model_path = os.path.join(args.cache_dir, args.model_file)
            
            # Download model if not found and download flag is set
            if not os.path.exists(model_path) and args.download:
                model_path = download_model(
                    model_repo="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
                    model_filename=args.model_file,
                    cache_dir=args.cache_dir
                )
            elif not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                logger.error("Use --download to automatically download the model")
                sys.exit(1)
        
        # Format prompt for Llama 3.1
        formatted_prompt = format_prompt_for_llama(prompt_content)
        
        # Initialize generator
        generator = LlamaFrameNetGenerator(
            model_path=model_path,
            n_ctx=8192,
            n_gpu_layers=-1
        )
        
        # Load model
        generator.load_model()
        
        # Generate response
        generation_params = {
            'temperature': args.temperature,
            'max_tokens': args.max_tokens,
        }
        
        response = generator.generate_response(formatted_prompt, generation_params)
        
        # Extract JSON from response
        try:
            output_data = extract_json_from_response(response)
        except Exception:
            # If JSON extraction fails, save raw response for debugging
            logger.warning("Failed to extract JSON, saving raw response")
            output_data = {
                "error": "Failed to parse JSON from response",
                "raw_response": response,
                "formatted_prompt": formatted_prompt
            }
        
        # Determine output file
        if args.output:
            output_file = args.output
        else:
            base_name = Path(args.prompt_file).stem
            output_file = f"{base_name}_output.json"
        
        # Save output
        save_output(output_data, output_file)
        
        # Print summary
        if "error" not in output_data:
            frame_name = output_data.get("frame", "Unknown")
            total_items = output_data.get("total", 0)
            print(f"\n=== Generation Summary ===")
            print(f"Frame: {frame_name}")
            print(f"Generated LUs: {total_items}")
            print(f"Output saved to: {output_file}")
        else:
            print(f"\n=== Error occurred ===")
            print(f"Check output file for details: {output_file}")
        
    except KeyboardInterrupt:
        logger.info("Generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
