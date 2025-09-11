#!/usr/bin/env python3
"""
FrameNet Lexical Unit Generator Flask API

This Flask application provides a web API for generating Brazilian Portuguese 
lexical units for FrameNet frames using Llama 3.1-8B-Instruct model.

Usage:
    python app.py --model-path /path/to/model.gguf --port 5000

API Endpoints:
    GET  /              - Web interface for testing
    POST /generate      - Generate lexical units (JSON API)
    GET  /health        - Health check endpoint

Author: Generated for FrameNet Brasil extension
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, Any

from flask import Flask, request, jsonify, render_template_string

# Import our core generator
try:
    from framenet_generator import FrameNetLUGenerator, download_model
except ImportError:
    print("Error: Could not import framenet_generator module.")
    print("Make sure framenet_generator.py is in the same directory.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Parse command line arguments
parser = argparse.ArgumentParser(description="FrameNet LU Generator Flask API")
parser.add_argument('--model-path', '-m', 
                   metavar='MODEL_PATH', 
                   default=os.environ.get('MODEL_PATH', '/srv/model/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf'),
                   help='Path to GGUF model file')
parser.add_argument('--port', '-p', 
                   type=int, 
                   default=int(os.environ.get('PORT', 5000)),
                   help='Port to run the Flask app on (default: 5000)')
parser.add_argument('--host', 
                   default=os.environ.get('HOST', '0.0.0.0'),
                   help='Host to bind the Flask app to (default: 0.0.0.0)')
parser.add_argument('--download', 
                   action='store_true',
                   help='Download model if not found locally')
parser.add_argument('--model-repo',
                   default='bartowski/Meta-Llama-3.1-8B-Instruct-GGUF',
                   help='HuggingFace model repository')
parser.add_argument('--model-file',
                   default='Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf',
                   help='Model filename to download')

args = parser.parse_args()

# Global generator instance
generator = None

# Initialize Flask app
app = Flask(__name__)

# Simple HTML template for testing
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>FrameNet LU Generator API</title>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .container { max-width: 800px; margin: 0 auto; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input, textarea, select { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
        button { background-color: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background-color: #0056b3; }
        .result { margin-top: 20px; padding: 15px; background-color: #f8f9fa; border-radius: 4px; }
        .error { background-color: #f8d7da; color: #721c24; }
        .success { background-color: #d4edda; color: #155724; }
        pre { white-space: pre-wrap; word-wrap: break-word; }
    </style>
</head>
<body>
    <div class="container">
        <h1>FrameNet LU Generator API</h1>
        <p>Generate Brazilian Portuguese lexical units for FrameNet frames using Llama 3.1-8B-Instruct.</p>
        
        <form id="generateForm">
            <div class="form-group">
                <label for="frame">Frame Name:</label>
                <input type="text" id="frame" name="frame" value="Artefato" required>
            </div>
            
            <div class="form-group">
                <label for="frame_definition">Frame Definition:</label>
                <textarea id="frame_definition" name="frame_definition" rows="3" required>Um artefato feito ou modificado por uma entidade inteligente para ser destinado a um determinado tipo de Uso. O artefato pode ou não ser uma entidade física.</textarea>
            </div>
            
            <div class="form-group">
                <label for="target_count">Number of Lexical Units:</label>
                <input type="number" id="target_count" name="target_count" value="10" min="1" max="50" required>
            </div>
            
            <div class="form-group">
                <label for="exclusion_list">Exclusion List (one per line):</label>
                <textarea id="exclusion_list" name="exclusion_list" rows="5" placeholder="anilha&#10;arame&#10;aspirador"></textarea>
            </div>
            
            <div class="form-group">
                <label for="acceptable_pos">Acceptable POS Tags (one per line):</label>
                <textarea id="acceptable_pos" name="acceptable_pos" rows="3" placeholder="VERB&#10;NOUN&#10;ADJ">VERB&#10;NOUN&#10;ADJ</textarea>
            </div>
            
            <div class="form-group">
                <label for="temperature">Temperature (0.0-1.0):</label>
                <input type="number" id="temperature" name="temperature" value="0.1" min="0" max="1" step="0.01">
            </div>
            
            <button type="submit">Generate Lexical Units</button>
        </form>
        
        <div id="result" class="result" style="display: none;"></div>
    </div>

    <script>
        document.getElementById('generateForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(e.target);
            const data = {
                frame: formData.get('frame'),
                frame_definition: formData.get('frame_definition'),
                target_count: parseInt(formData.get('target_count')),
                temperature: parseFloat(formData.get('temperature')),
                exclusion_list: formData.get('exclusion_list').split('\\n').filter(item => item.trim() !== ''),
                acceptable_pos: formData.get('acceptable_pos').split('\\n').filter(item => item.trim() !== '')
            };
            
            const resultDiv = document.getElementById('result');
            resultDiv.style.display = 'block';
            resultDiv.innerHTML = '<p>Generating lexical units... This may take a minute.</p>';
            resultDiv.className = 'result';
            
            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    resultDiv.className = 'result success';
                    resultDiv.innerHTML = '<h3>Generated Lexical Units:</h3><pre>' + JSON.stringify(result, null, 2) + '</pre>';
                } else {
                    resultDiv.className = 'result error';
                    resultDiv.innerHTML = '<h3>Error:</h3><pre>' + JSON.stringify(result, null, 2) + '</pre>';
                }
            } catch (error) {
                resultDiv.className = 'result error';
                resultDiv.innerHTML = '<h3>Error:</h3><p>' + error.message + '</p>';
            }
        });
    </script>
</body>
</html>
"""

def initialize_model():
    """Initialize the global model instance"""
    global generator
    
    if generator is not None:
        logger.info("Model already initialized")
        return
    
    try:
        # Check if model file exists
        if not os.path.exists(args.model_path):
            if args.download:
                logger.info(f"Model not found at {args.model_path}, downloading...")
                model_path = download_model(
                    model_repo=args.model_repo,
                    model_filename=args.model_file,
                    cache_dir=os.path.dirname(args.model_path) or "./models"
                )
                args.model_path = model_path
            else:
                raise FileNotFoundError(f"Model file not found: {args.model_path}")
        
        # Initialize generator
        logger.info("Initializing FrameNet LU Generator...")
        generator = FrameNetLUGenerator(model_path=args.model_path)
        generator.load_model()
        logger.info("Model initialization completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise

@app.route('/')
def index():
    """Web interface for testing the API"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/health')
def health():
    """Health check endpoint"""
    try:
        model_status = "loaded" if generator and generator.model else "not_loaded"
        return jsonify({
            "status": "healthy",
            "model_status": model_status,
            "model_path": args.model_path
        }), 200
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

@app.route('/generate', methods=['POST'])
def generate():
    """Generate lexical units for a given frame"""
    try:
        # Validate request
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['frame', 'frame_definition']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Extract parameters with defaults
        frame = data['frame']
        frame_definition = data['frame_definition']
        target_count = data.get('target_count', 10)
        exclusion_list = data.get('exclusion_list', [])
        acceptable_pos = data.get('acceptable_pos', ['VERB', 'NOUN', 'ADJ'])
        temperature = data.get('temperature', 0.1)
        max_tokens = data.get('max_tokens', 2048)
        
        # Validate parameters
        if not isinstance(frame, str) or not frame.strip():
            return jsonify({'error': 'Frame must be a non-empty string'}), 400
        
        if not isinstance(frame_definition, str) or not frame_definition.strip():
            return jsonify({'error': 'Frame definition must be a non-empty string'}), 400
        
        if not isinstance(target_count, int) or target_count < 1 or target_count > 50:
            return jsonify({'error': 'Target count must be between 1 and 50'}), 400
        
        if not isinstance(exclusion_list, list):
            return jsonify({'error': 'Exclusion list must be an array'}), 400
        
        if not isinstance(acceptable_pos, list) or not acceptable_pos:
            return jsonify({'error': 'Acceptable POS must be a non-empty array'}), 400
            
        # Validate POS tags
        valid_pos_tags = {'VERB', 'NOUN', 'ADJ', 'ADV', 'PRON', 'DET', 'ADP', 'CONJ', 'NUM', 'PUNCT', 'SYM', 'X'}
        for pos in acceptable_pos:
            if not isinstance(pos, str) or pos not in valid_pos_tags:
                return jsonify({'error': f'Invalid POS tag: {pos}. Valid tags: {", ".join(sorted(valid_pos_tags))}'}), 400
        
        if not isinstance(temperature, (int, float)) or temperature < 0 or temperature > 1:
            return jsonify({'error': 'Temperature must be between 0.0 and 1.0'}), 400
        
        # Ensure model is initialized
        if generator is None:
            return jsonify({'error': 'Model not initialized'}), 500
        
        # Generation parameters
        generation_params = {
            'temperature': temperature,
            'max_tokens': max_tokens,
        }
        
        # Generate lexical units
        logger.info(f"Generating {target_count} lexical units for frame: {frame} with POS: {acceptable_pos}")
        result = generator.generate_lexical_units(
            frame=frame,
            frame_definition=frame_definition,
            target_count=target_count,
            exclusion_list=exclusion_list,
            acceptable_pos=acceptable_pos,
            generation_params=generation_params
        )
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    try:
        # Initialize model on startup
        initialize_model()
        
        # Run Flask app
        logger.info(f"Starting FrameNet LU Generator API on {args.host}:{args.port}")
        app.run(host=args.host, port=args.port, debug=False)
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        sys.exit(1)