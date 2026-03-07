"""
SpeechToText — wraps Azure Cognitive Services Speech SDK for
speech recognition from a base64-encoded WAV audio clip.
"""
import base64
import os
import tempfile

import azure.cognitiveservices.speech as speechsdk


class SpeechToText:
    # Maps the language codes used in the UI to Azure locale strings
    LANGUAGE_MAP: dict[str, str] = {
        "en-US": "en-US",
        "en-IN": "en-IN",
        "hi-IN": "hi-IN",
        "ta-IN": "ta-IN",
        "te-IN": "te-IN",
        "mr-IN": "mr-IN",
    }

    def __init__(self):
        self.speech_key = os.getenv("SPEECH_KEY")
        self.speech_region = os.getenv("SPEECH_REGION", "eastus")
        if not self.speech_key:
            raise EnvironmentError(
                "SPEECH_KEY environment variable is not set. "
                "Please set it to your Azure Speech Services key."
            )

    def transcribe_audio(self, audio_base64: str, language: str = "en-US") -> dict:
        """
        Transcribe a base64-encoded WAV file using Azure Speech-to-Text.

        Args:
            audio_base64: Base64-encoded WAV audio data (from MediaRecorder in browser).
            language:     BCP-47 locale string — must be a key in LANGUAGE_MAP.

        Returns:
            {
                "success": bool,
                "text":     str,   # recognised transcript (on success)
                "language": str,
                "error":    str    # on failure
            }
        """
        locale = self.LANGUAGE_MAP.get(language, "en-US")

        # Decode audio into a temporary WAV file
        try:
            audio_bytes = base64.b64decode(audio_base64)
        except Exception as exc:
            return {"success": False, "error": f"Failed to decode audio: {exc}"}

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            # Build Azure Speech config
            speech_config = speechsdk.SpeechConfig(
                subscription=self.speech_key,
                region=self.speech_region,
            )
            speech_config.speech_recognition_language = locale

            audio_config = speechsdk.audio.AudioConfig(filename=tmp_path)
            recognizer = speechsdk.SpeechRecognizer(
                speech_config=speech_config,
                audio_config=audio_config,
            )

            result = recognizer.recognize_once_async().get()

        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        # Interpret result
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return {
                "success": True,
                "text": result.text,
                "language": language,
            }
        elif result.reason == speechsdk.ResultReason.NoMatch:
            return {
                "success": False,
                "error": (
                    "No speech was recognised. "
                    "Please speak clearly and try again."
                ),
            }
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation = result.cancellation_details
            return {
                "success": False,
                "error": (
                    f"Recognition cancelled: {cancellation.reason}. "
                    f"Details: {cancellation.error_details}"
                ),
            }
        else:
            return {
                "success": False,
                "error": f"Unexpected recognition result: {result.reason}",
            }