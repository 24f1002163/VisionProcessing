"""
Flask server for the Interactive Note Processor
Run with: python app.py
"""
from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS
import asyncio

from concept_extraction_agent import ConceptExtractor
from image_highlighter import highlight_image_with_concepts
from speech_generator import SpeechGenerator
from quiz_generator import QuizGenerator
from speech_to_text import SpeechToText
from dotenv import load_dotenv

load_dotenv(override=True)

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:8000", "http://localhost:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
    }
})

# Shared async event loop (used for SpeechGenerator which is async)
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)


# ──────────────────────────────────────────────────────────────────────────────
# Static file serving
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return send_from_directory(".", "index.html")


@app.route("/<path:path>", methods=["GET"])
def serve_static(path):
    if path.startswith("api/"):
        abort(404)
    try:
        return send_from_directory(".", path)
    except Exception:
        abort(404)


# ──────────────────────────────────────────────────────────────────────────────
# Health / info
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/api", methods=["GET"])
def api_root():
    return jsonify({
        "status": "healthy",
        "service": "Interactive Note Processor",
        "endpoints": [
            "/api/health",
            "/api/upload_notes",
            "/api/generate_speech",
            "/api/supported_languages",
            "/api/quiz/generate_question",
            "/api/quiz/speech_to_text",
            "/api/quiz/evaluate_answer",
        ],
    }), 200


@app.route("/api/health", methods=["GET", "OPTIONS"])
def health():
    if request.method == "OPTIONS":
        return "", 204
    return jsonify({"status": "healthy", "service": "Interactive Note Processor"})


# ──────────────────────────────────────────────────────────────────────────────
# Notes processing
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/api/upload_notes", methods=["POST", "OPTIONS"])
def upload_notes():
    if request.method == "OPTIONS":
        return "", 204
    try:
        body = request.get_json()
        image_base64 = body.get("image_base64")
        if not image_base64:
            return jsonify({"success": False, "error": "Missing image_base64"}), 400

        extraction_result = ConceptExtractor.extract_concepts_from_highlighted_region(image_base64)
        if not extraction_result:
            return jsonify({"success": False, "error": "Failed to extract concepts"}), 500

        concepts = (
            extraction_result["concepts"]
            if isinstance(extraction_result, dict) and "concepts" in extraction_result
            else extraction_result
        )

        highlight_result = highlight_image_with_concepts(image_base64, concepts)
        if not highlight_result.get("success"):
            return jsonify({
                "success": False,
                "error": highlight_result.get("error", "Failed to highlight image"),
            }), 500

        regions = highlight_result["regions"]
        concept_summary_map = {c["id"]: c.get("summary", "") for c in concepts}
        for region in regions:
            region["summary"] = concept_summary_map.get(region["id"], "")

        return jsonify({
            "success": True,
            "highlighted_image": highlight_result["highlighted_image"],
            "concepts": regions,
            "image_dimensions": highlight_result["image_dimensions"],
            "message": f"Successfully extracted {len(concepts)} concepts",
        }), 200

    except Exception as exc:
        return jsonify({"success": False, "error": f"Internal server error: {exc}"}), 500


# ──────────────────────────────────────────────────────────────────────────────
# Speech synthesis (TTS)
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/api/generate_speech", methods=["POST", "OPTIONS"])
def generate_speech():
    if request.method == "OPTIONS":
        return "", 204
    try:
        body = request.get_json()
        concept_name = body.get("concept_name")
        concept_description = body.get("concept_description")
        language = body.get("language", "en-US")

        if not concept_name or not concept_description:
            return jsonify({
                "success": False,
                "error": "Missing concept_name or concept_description",
            }), 400

        speech_gen = SpeechGenerator()
        result = loop.run_until_complete(
            speech_gen.generate_speech(concept_name, concept_description, language)
        )

        if not result.get("success"):
            return jsonify(result), 500

        result["explanation_text"] = concept_description
        return jsonify(result), 200

    except Exception as exc:
        return jsonify({"success": False, "error": f"Internal server error: {exc}"}), 500


@app.route("/api/supported_languages", methods=["GET", "OPTIONS"])
def get_supported_languages():
    if request.method == "OPTIONS":
        return "", 204
    try:
        speech_gen = SpeechGenerator()
        return jsonify({
            "success": True,
            "supported_languages": speech_gen.get_supported_languages(),
        }), 200
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


# ──────────────────────────────────────────────────────────────────────────────
# Quiz endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/api/quiz/generate_question", methods=["POST", "OPTIONS"])
def quiz_generate_question():
    """
    Request body:
        {
            "concept_name":        "string",
            "concept_description": "string",
            "difficulty":          "easy" | "medium" | "hard",   (optional)
            "language":            "en-US"                        (optional)
        }

    Response:
        {
            "success":      true,
            "question":     "string",
            "audio_base64": "string",
            "language":     "string",
            "voice":        "string"
        }
    """
    if request.method == "OPTIONS":
        return "", 204
    try:
        body = request.get_json()
        concept_name        = body.get("concept_name")
        concept_description = body.get("concept_description")
        difficulty          = body.get("difficulty", "medium")
        language            = body.get("language", "en-US")

        if not concept_name or not concept_description:
            return jsonify({
                "success": False,
                "error": "Missing concept_name or concept_description",
            }), 400

        # 1 — Generate question text via Claude
        quiz_gen = QuizGenerator()
        q_result = quiz_gen.generate_question(concept_name, concept_description, difficulty)
        if not q_result.get("success"):
            return jsonify(q_result), 500

        question_text = q_result["question"]

        # 2 — Synthesise question to speech
        speech_gen = SpeechGenerator()
        speech_result = loop.run_until_complete(
            speech_gen.generate_speech(
                f"Quiz question about {concept_name}",
                question_text,
                language,
            )
        )

        if not speech_result.get("success"):
            return jsonify({
                "success": True,
                "question": question_text,
                "audio_base64": None,
                "language": language,
                "voice": None,
                "tts_error": speech_result.get("error"),
            }), 200

        return jsonify({
            "success": True,
            "question": question_text,
            "audio_base64": speech_result.get("audio_base64"),
            "language": language,
            "voice": speech_result.get("voice"),
        }), 200

    except Exception as exc:
        return jsonify({"success": False, "error": f"Internal server error: {exc}"}), 500


@app.route("/api/quiz/speech_to_text", methods=["POST", "OPTIONS"])
def quiz_speech_to_text():
    """
    Request body:
        {
            "audio_base64": "string",
            "language":     "en-US"   (optional)
        }

    Response:
        { "success": true, "text": "string", "language": "string" }
    """
    if request.method == "OPTIONS":
        return "", 204
    try:
        body = request.get_json()
        audio_base64 = body.get("audio_base64")
        language     = body.get("language", "en-US")

        if not audio_base64:
            return jsonify({"success": False, "error": "Missing audio_base64"}), 400

        stt = SpeechToText()
        result = stt.transcribe_audio(audio_base64, language)
        status = 200 if result.get("success") else 500
        return jsonify(result), status

    except Exception as exc:
        return jsonify({"success": False, "error": f"Internal server error: {exc}"}), 500


@app.route("/api/quiz/evaluate_answer", methods=["POST", "OPTIONS"])
def quiz_evaluate_answer():
    """
    Request body:
        {
            "concept_name":        "string",
            "concept_description": "string",
            "question":            "string",
            "student_answer":      "string",
            "language":            "en-US"   (optional)
        }

    Response:
        {
            "success":      true,
            "rating":       "correct" | "partially_correct" | "incorrect",
            "feedback":     "string",
            "audio_base64": "string",
            "language":     "string",
            "voice":        "string"
        }
    """
    if request.method == "OPTIONS":
        return "", 204
    try:
        body = request.get_json()
        concept_name        = body.get("concept_name")
        concept_description = body.get("concept_description")
        question            = body.get("question")
        student_answer      = body.get("student_answer")
        language            = body.get("language", "en-US")

        if not all([concept_name, concept_description, question, student_answer]):
            return jsonify({
                "success": False,
                "error": "Missing one or more required fields: concept_name, concept_description, question, student_answer",
            }), 400

        # 1 — Evaluate via Claude
        quiz_gen = QuizGenerator()
        eval_result = quiz_gen.evaluate_answer(
            concept_name, concept_description, question, student_answer
        )
        if not eval_result.get("success"):
            return jsonify(eval_result), 500

        feedback_text = eval_result["feedback"]
        rating        = eval_result["rating"]

        # 2 — Synthesise feedback to speech
        speech_gen = SpeechGenerator()
        speech_result = loop.run_until_complete(
            speech_gen.generate_speech(
                f"Feedback on {concept_name}",
                feedback_text,
                language,
            )
        )

        response = {
            "success":  True,
            "rating":   rating,
            "feedback": feedback_text,
            "language": language,
        }

        if speech_result.get("success"):
            response["audio_base64"] = speech_result.get("audio_base64")
            response["voice"]        = speech_result.get("voice")
        else:
            response["audio_base64"] = None
            response["tts_error"]    = speech_result.get("error")

        return jsonify(response), 200

    except Exception as exc:
        return jsonify({"success": False, "error": f"Internal server error: {exc}"}), 500


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os as _os

    def _check(label, val, secret=True):
        if val:
            display = (val[:8] + "…") if secret else val
            print(f"  ✓ {label}: {display}")
        else:
            print(f"  ✗ {label}: NOT SET ← fix this in .env")

    print("🚀 Starting Interactive Note Processor server...")
    print("📍 API: http://localhost:3000\n")
    print("── Environment check ──────────────────────────")
    _check("OPENAI_KEY",                   _os.getenv("OPENAI_KEY"))
    _check("SPEECH_KEY",                   _os.getenv("SPEECH_KEY"))
    _check("SPEECH_REGION",                _os.getenv("SPEECH_REGION", "eastus"), secret=False)
    _check("AZURE_OPENAI_ENDPOINT",        _os.getenv("AZURE_OPENAI_ENDPOINT"), secret=False)
    _check("AZURE_OPENAI_API_KEY",         _os.getenv("AZURE_OPENAI_API_KEY"))
    _check("AZURE_OPENAI_DEPLOYMENT_NAME", _os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"), secret=False)
    print("───────────────────────────────────────────────\n")
    print("Press Ctrl+C to stop\n")

    app.run(debug=True, host="localhost", port=3000)