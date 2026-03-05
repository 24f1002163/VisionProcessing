"""
Azure Functions HTTP endpoint for the Interactive Note Processor
Orchestrates concept extraction, highlighting, and speech generation
"""
import azure.functions as func
import json
import base64
import asyncio
from concept_extraction_agent import ConceptExtractor
from image_highlighter import highlight_image_with_concepts
from speech_generator import SpeechGenerator
from dotenv import load_dotenv

load_dotenv(override=True)

# Create the blueprint for HTTP trigger
bp = func.Blueprint()


@bp.route(route="upload_notes", methods=["POST"])
async def upload_notes(req: func.HttpRequest) -> func.HttpResponse:
    """
    Process student notes image
    
    Request body:
    {
        "image_base64": "...",
        "image_format": "png"  (optional)
    }
    
    Response:
    {
        "success": bool,
        "highlighted_image": "base64_string",
        "concepts": [
            {
                "id": "concept_1",
                "name": "Concept Name",
                "description": "Description",
                "color": {"r": 255, "g": 200, "b": 0},
                "region": {"x1": 10, "y1": 20, "x2": 100, "y2": 120}
            }
        ],
        "message": "string"
    }
    """
    try:
        # Parse request
        req_body = req.get_json()
        image_base64 = req_body.get('image_base64')
        
        if not image_base64:
            return func.HttpResponse(
                json.dumps({"success": False, "error": "Missing image_base64 in request"}),
                status_code=400,
                mimetype="application/json"
            )
        
        # Extract concepts using agent
        extractor = ConceptExtractor()
        extraction_result = await extractor.extract_concepts(image_base64)
        
        if not extraction_result.get('success'):
            await extractor.close()
            return func.HttpResponse(
                json.dumps({
                    "success": False,
                    "error": extraction_result.get('error', 'Failed to extract concepts')
                }),
                status_code=500,
                mimetype="application/json"
            )
        
        concepts = extraction_result.get('concepts', [])
        
        # Highlight image with concepts
        highlight_result = highlight_image_with_concepts(image_base64, concepts)
        
        if not highlight_result.get('success'):
            await extractor.close()
            return func.HttpResponse(
                json.dumps({
                    "success": False,
                    "error": highlight_result.get('error', 'Failed to highlight image')
                }),
                status_code=500,
                mimetype="application/json"
            )
        
        # Clean up
        await extractor.close()
        
        # Prepare response
        response_data = {
            "success": True,
            "highlighted_image": highlight_result['highlighted_image'],
            "concepts": highlight_result['regions'],
            "image_dimensions": highlight_result['image_dimensions'],
            "message": f"Successfully extracted {len(concepts)} concepts"
        }
        
        return func.HttpResponse(
            json.dumps(response_data),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        return func.HttpResponse(
            json.dumps({
                "success": False,
                "error": f"Internal server error: {str(e)}"
            }),
            status_code=500,
            mimetype="application/json"
        )


@bp.route(route="generate_speech", methods=["POST"])
async def generate_speech(req: func.HttpRequest) -> func.HttpResponse:
    """
    Generate audio explanation for a concept
    
    Request body:
    {
        "concept_name": "string",
        "concept_description": "string",
        "language": "en-US"  (optional, default: en-US)
    }
    
    Response:
    {
        "success": bool,
        "audio_base64": "string",
        "audio_format": "wav",
        "language": "string",
        "voice": "string",
        "duration_ms": int
    }
    """
    try:
        # Parse request
        req_body = req.get_json()
        concept_name = req_body.get('concept_name')
        concept_description = req_body.get('concept_description')
        language = req_body.get('language', 'en-US')
        
        if not concept_name or not concept_description:
            return func.HttpResponse(
                json.dumps({
                    "success": False,
                    "error": "Missing concept_name or concept_description"
                }),
                status_code=400,
                mimetype="application/json"
            )
        
        # Generate speech
        speech_gen = SpeechGenerator()
        result = await speech_gen.generate_speech(
            concept_name,
            concept_description,
            language
        )
        
        if not result.get('success'):
            return func.HttpResponse(
                json.dumps(result),
                status_code=500,
                mimetype="application/json"
            )
        
        return func.HttpResponse(
            json.dumps(result),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        return func.HttpResponse(
            json.dumps({
                "success": False,
                "error": f"Internal server error: {str(e)}"
            }),
            status_code=500,
            mimetype="application/json"
        )


@bp.route(route="supported_languages", methods=["GET"])
def get_supported_languages(req: func.HttpRequest) -> func.HttpResponse:
    """
    Get list of supported languages for speech synthesis
    
    Response:
    {
        "supported_languages": {
            "en-US": {...},
            "hi-IN": {...},
            ...
        }
    }
    """
    try:
        speech_gen = SpeechGenerator()
        languages = speech_gen.get_supported_languages()
        
        return func.HttpResponse(
            json.dumps({
                "success": True,
                "supported_languages": languages
            }),
            status_code=200,
            mimetype="application/json"
        )
    except Exception as e:
        return func.HttpResponse(
            json.dumps({
                "success": False,
                "error": str(e)
            }),
            status_code=500,
            mimetype="application/json"
        )


@bp.route(route="health", methods=["GET"])
def health_check(req: func.HttpRequest) -> func.HttpResponse:
    """Health check endpoint"""
    return func.HttpResponse(
        json.dumps({
            "status": "healthy",
            "service": "Interactive Note Processor"
        }),
        status_code=200,
        mimetype="application/json"
    )
