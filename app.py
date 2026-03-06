"""
Simple Flask server for testing the image processor
Run with: python app.py
"""
from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS
import json
import asyncio
from concept_extraction_agent import ConceptExtractor
from image_highlighter import highlight_image_with_concepts
from speech_generator import SpeechGenerator
from dotenv import load_dotenv

load_dotenv(override=True)

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:8000", "http://localhost:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Store the event loop for async operations
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)


@app.route('/', methods=['GET'])
def index():
    """Serve the frontend index.html"""
    return send_from_directory('.', 'index.html')


@app.route('/<path:path>', methods=['GET'])
def serve_static(path):
    """Serve other static files, but do not intercept API routes"""
    if path.startswith('api/'):
        abort(404)
    try:
        return send_from_directory('.', path)
    except Exception:
        abort(404)


@app.route('/api/health', methods=['GET', 'OPTIONS'])
def health():
    """Health check endpoint"""
    if request.method == 'OPTIONS':
        return '', 204
    return jsonify({
        "status": "healthy",
        "service": "Interactive Note Processor"
    })


@app.route('/api/upload_notes', methods=['POST', 'OPTIONS'])
def upload_notes():
    """
    Process student notes image
    
    Request body:
    {
        "image_base64": "..."
    }
    """
    if request.method == 'OPTIONS':
        return '', 204
    try:
        req_body = request.get_json()
        image_base64 = req_body.get('image_base64')
        
        if not image_base64:
            return jsonify({
                "success": False,
                "error": "Missing image_base64 in request"
            }), 400
        
        # Extract concepts using agent
        extraction_result = ConceptExtractor.extract_concepts_from_highlighted_region(image_base64)
        if not extraction_result:
            return jsonify({
                "success": False,
                "error": "Failed to extract concepts"
            }), 500
        # If the result is a dict with 'concepts' key, extract it, else treat as concepts list/string
        if isinstance(extraction_result, dict) and 'concepts' in extraction_result:
            concepts = extraction_result['concepts']
        else:
            concepts = extraction_result

        # Highlight image with concepts
        highlight_result = highlight_image_with_concepts(image_base64, concepts)

        if not highlight_result.get('success'):
            return jsonify({
                "success": False,
                "error": highlight_result.get('error', 'Failed to highlight image')
            }), 500

        # Merge summary field from original concepts into regions
        regions = highlight_result['regions']
        concept_summary_map = {c['id']: c.get('summary', '') for c in concepts}
        for region in regions:
            region['summary'] = concept_summary_map.get(region['id'], '')

            # Clean up (no extractor instance to close)

        # Prepare response
        response_data = {
            "success": True,
            "highlighted_image": highlight_result['highlighted_image'],
            "concepts": regions,
            "image_dimensions": highlight_result['image_dimensions'],
            "message": f"Successfully extracted {len(concepts)} concepts"
        }

        return jsonify(response_data), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Internal server error: {str(e)}"
        }), 500


@app.route('/api/generate_speech', methods=['POST', 'OPTIONS'])
def generate_speech():
    """
    Generate audio explanation for a concept
    
    Request body:
    {
        "concept_name": "string",
        "concept_description": "string",
        "language": "en-US"  (optional)
    }
    """
    if request.method == 'OPTIONS':
        return '', 204
    try:
        req_body = request.get_json()
        concept_name = req_body.get('concept_name')
        concept_description = req_body.get('concept_description')
        language = req_body.get('language', 'en-US')
        
        if not concept_name or not concept_description:
            return jsonify({
                "success": False,
                "error": "Missing concept_name or concept_description"
            }), 400
        
        # Generate speech
        speech_gen = SpeechGenerator()
        result = loop.run_until_complete(speech_gen.generate_speech(
            concept_name,
            concept_description,
            language
        ))
        
        if not result.get('success'):
            return jsonify(result), 500
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Internal server error: {str(e)}"
        }), 500


@app.route('/api/supported_languages', methods=['GET', 'OPTIONS'])
def get_supported_languages():
    """Get list of supported languages for speech synthesis"""
    if request.method == 'OPTIONS':
        return '', 204
    try:
        speech_gen = SpeechGenerator()
        languages = speech_gen.get_supported_languages()
        
        return jsonify({
            "success": True,
            "supported_languages": languages
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == '__main__':
    print("🚀 Starting Interactive Note Processor server...")
    print("📍 API running at: http://localhost:3000/api")
    print("🌐 Open index.html in your browser")
    print("\n✓ Image processing: READY")
    print("⏳ Speech generation: Configure SPEECH_KEY when ready")
    print("\nPress Ctrl+C to stop the server\n")
    
    app.run(debug=True, host='localhost', port=3000)
