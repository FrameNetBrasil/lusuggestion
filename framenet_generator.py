#!/usr/bin/env python3
"""
FrameNet Lexical Unit Generator Core Module

This module provides the core functionality for generating FrameNet lexical units
using LLM models. Extracted from the command-line scripts for use in web services.

Author: Generated for FrameNet Brasil extension
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

try:
    from llama_cpp import Llama
    from huggingface_hub import hf_hub_download
except ImportError as e:
    raise ImportError(f"Required libraries not found: {e}. Please install: pip install llama-cpp-python[cublas] huggingface_hub")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class FrameNetLUGenerator:
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
        if self.model is not None:
            logger.info("Model already loaded")
            return
            
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

    def format_prompt_for_llama(self, base_prompt: str) -> str:
        """
        Format the prompt specifically for Llama 3.1-8B-Instruct model
        Uses the Llama 3 chat template format for optimal performance
        
        Args:
            base_prompt: The base FrameNet prompt
            
        Returns:
            Formatted prompt for Llama 3.1
        """
        # Llama 3.1 uses specific chat template format (without <|begin_of_text|> to avoid duplication warning)
        formatted_prompt = f"""<|start_header_id|>system<|end_header_id|>

Você é um especialista em lexicografia computacional trabalhando no FrameNet Brasil. Sua tarefa é gerar unidades lexicais em português brasileiro seguindo rigorosamente as instruções fornecidas. Responda apenas com JSON válido, sem explicações adicionais.<|eot_id|><|start_header_id|>user<|end_header_id|>

{base_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        return formatted_prompt

    def extract_json_from_response(self, response: str) -> Dict[str, Any]:
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

    def generate_lexical_units(self, 
                              frame: str,
                              frame_definition: str,
                              target_count: int = 10,
                              exclusion_list: List[str] = None,
                              acceptable_pos: List[str] = None,
                              generation_params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Generate lexical units for a given frame
        
        Args:
            frame: Frame name (e.g., "Artefato")
            frame_definition: Frame definition text
            target_count: Number of lexical units to generate
            exclusion_list: List of existing lexical units to avoid
            acceptable_pos: List of acceptable POS tags (default: ["VERB", "NOUN", "ADJ"])
            generation_params: Generation parameters for the model
            
        Returns:
            Dictionary containing generated lexical units in JSON format
        """
        # Ensure model is loaded
        if self.model is None:
            self.load_model()
        
        if exclusion_list is None:
            exclusion_list = []
            
        if acceptable_pos is None:
            acceptable_pos = ["VERB", "NOUN", "ADJ"]
        
        # Create the prompt
        exclusion_text = "\n".join([f"- {item}" for item in exclusion_list]) if exclusion_list else "None specified"
        pos_list = ", ".join(acceptable_pos)
        
        prompt = f"""You are an expert lexicographer working on FrameNet Brasil (FN-Br).
Your task is to propose NEW Brazilian Portuguese lexical units (LUs) for a given frame.

FRAME: {frame}
DEFINITION: {frame_definition}

⚠️  EXISTING LEXICAL UNITS TO AVOID (CRITICAL) ⚠️
The following LUs already exist for this frame and MUST BE COMPLETELY AVOIDED:
{exclusion_text}

You must cross-reference EVERY proposed lemma against this list before including it in your response.

STRICT REQUIREMENTS:
1) Language: pt-BR lemmas only (avoid Europeanisms unless also common in Brazil).
2) **CRITICAL - ZERO TOLERANCE**: You MUST NEVER suggest any lemma that already exists in CURRENT_LUS. This is a CRITICAL ERROR that invalidates the entire response.
   - Compare case-insensitively by lemma+POS (e.g., "comprar.VERB" matches "Comprar.VERB")
   - Avoid near-duplicates and morphological variants (e.g., compra.NOUN vs compras.NOUN)
   - Before finalizing each suggestion, MANDATORY cross-check against the complete CURRENT_LUS list
   - If you cannot generate TARGET_N unique suggestions due to overlap with existing LUs, provide fewer and explain in excluded_notes
3) **CRITICAL**: Output EXACTLY {target_count} suggestions when possible. Do not output fewer unless absolutely impossible due to frame constraints OR duplicate avoidance. If you must provide fewer, explain thoroughly in excluded_notes why TARGET_N suggestions cannot be generated.
4) Each suggestion MUST plausibly evoke the TARGET_FRAME given its DEFINITION and the exemplars in CURRENT_LUS.
   - If a candidate typically evokes a DIFFERENT frame (e.g., Causation vs Commerce_buy), exclude it.
5) **POS restriction**: Only return lemmas tagged with the following POS: {pos_list}.
   - Do NOT include ADV, interjections, or function words.
   - Do NOT output MWEs (multiword expressions). Only single-word lemmas are valid.
6) Prefer contemporary, common lemmas; avoid archaic, highly regional, or named-entity items unless frame-typical.
7) Lemmatize (singular for nouns; masculine singular for adjectives; infinitive for verbs).
8) Sense is crucial: if the lemma is polysemous, ensure the proposed sense matches the TARGET_FRAME (note that in `sense_note`).
9) Provide one SHORT example sentence in pt-BR for each LU that clearly instantiates the TARGET_FRAME.
10) Format the output as a single JSON object matching the SCHEMA below. No prose outside JSON.

QUALITY CHECKS:
- Calibrate coverage: include hyponyms, near-synonyms, and derivationally related forms that evoke the same frame.
- Assign a confidence in [0,1] based on sense reliability in the target frame.
- If you must include borderline items, keep confidence ≤ 0.6 and explain briefly in `rationale_short`.

MANDATORY VALIDATION BEFORE OUTPUT:
1. Review each proposed lemma in your JSON response
2. Cross-check each lemma.POS combination against the CURRENT_LUS list (case-insensitive)
3. Remove any matches or near-duplicates immediately
4. If removal reduces suggestions below TARGET_N, explain in excluded_notes
5. Double-check that no suggested lemma appears in CURRENT_LUS

SCHEMA (JSON):
{{
  "frame": "{frame}",
  "total": {target_count},
  "items": [
    {{
      "lemma": "<pt-BR lemma in canonical form>",
      "pos": "<TARGET_POS>",
      "mwe": false,
      "gloss_pt": "<one-line sense aligned to the frame>",
      "example_pt": "<short sentence showing the frame usage>",
      "sense_note": "<disambiguate if polysemous; why this evokes the frame>",
      "rationale_short": "<why this fits; link to core participants or prototypical scenarios>",
      "confidence": <0.0-1.0>
    }}
  ],
  "excluded_notes": "<if fewer than TARGET_N, say why (e.g., frame is narrow)>"
}}

Return only this JSON.
"""

        # Format for Llama 3.1
        formatted_prompt = self.format_prompt_for_llama(prompt)
        
        # Generate response
        response = self.generate_response(formatted_prompt, generation_params)
        
        # Extract and return JSON
        try:
            return self.extract_json_from_response(response)
        except Exception as e:
            logger.warning("Failed to extract JSON, returning raw response")
            return {
                "error": "Failed to parse JSON from response",
                "raw_response": response,
                "formatted_prompt": formatted_prompt
            }


def download_model(model_repo: str, model_filename: str, cache_dir: str = "./models") -> str:
    """
    Download Llama 3.1-8B-Instruct GGUF model from HuggingFace Hub
    
    Args:
        model_repo: Repository ID (e.g., "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF")
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