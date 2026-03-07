"""
SpeechToText — wraps Azure Cognitive Services Speech SDK.
Converts browser MediaRecorder output (WebM/Opus) to WAV using
PyAV — no system ffmpeg required, codecs are bundled in the package.

Install: pip install av
"""
import array
import base64
import io
import os
import tempfile
import wave

import azure.cognitiveservices.speech as speechsdk


class SpeechToText:
    LANGUAGE_MAP: dict[str, str] = {
        "en-US": "en-US",
        "en-IN": "en-IN",
        "hi-IN": "hi-IN",
        "ta-IN": "ta-IN",
        "te-IN": "te-IN",
        "mr-IN": "mr-IN",
    }

    TARGET_SAMPLE_RATE  = 16000
    TARGET_CHANNELS     = 1
    TARGET_SAMPLE_WIDTH = 2  # 16-bit

    def __init__(self):
        self.speech_key    = os.getenv("SPEECH_KEY")
        self.speech_region = os.getenv("SPEECH_REGION", "eastus")
        if not self.speech_key:
            raise EnvironmentError("SPEECH_KEY environment variable is not set.")

    # ------------------------------------------------------------------ #
    #  Audio conversion                                                    #
    # ------------------------------------------------------------------ #

    def _to_wav_bytes(self, audio_bytes: bytes) -> bytes:
        """
        Convert any browser audio (WebM/Opus/WAV/OGG) →
        16 kHz 16-bit mono WAV using PyAV (bundled codecs, no system ffmpeg).
        """
        try:
            import av
        except ImportError:
            raise RuntimeError(
                "PyAV is not installed. Run: pip install av"
            )

        import numpy as np

        pcm_chunks = []

        try:
            container = av.open(io.BytesIO(audio_bytes))
            resampler = av.AudioResampler(
                format="s16",
                layout="mono",
                rate=self.TARGET_SAMPLE_RATE,
            )

            for frame in container.decode(audio=0):
                for resampled in resampler.resample(frame):
                    # to_ndarray gives shape (channels, samples)
                    pcm_chunks.append(resampled.to_ndarray().tobytes())

            # Flush resampler
            for resampled in resampler.resample(None):
                pcm_chunks.append(resampled.to_ndarray().tobytes())

            container.close()

        except Exception as exc:
            raise RuntimeError(f"PyAV failed to decode audio: {exc}")

        if not pcm_chunks:
            raise RuntimeError("No audio data decoded from the input stream.")

        pcm_data = b"".join(pcm_chunks)

        out = io.BytesIO()
        with wave.open(out, "wb") as wf:
            wf.setnchannels(self.TARGET_CHANNELS)
            wf.setsampwidth(self.TARGET_SAMPLE_WIDTH)
            wf.setframerate(self.TARGET_SAMPLE_RATE)
            wf.writeframes(pcm_data)
        return out.getvalue()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def transcribe_audio(self, audio_base64: str, language: str = "en-US") -> dict:
        locale = self.LANGUAGE_MAP.get(language, "en-US")

        try:
            audio_bytes = base64.b64decode(audio_base64)
        except Exception as exc:
            return {"success": False, "error": f"Failed to decode audio: {exc}"}

        try:
            wav_bytes = self._to_wav_bytes(audio_bytes)
        except Exception as exc:
            return {"success": False, "error": f"Audio conversion failed: {exc}"}

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(wav_bytes)
                tmp_path = tmp.name

            speech_config = speechsdk.SpeechConfig(
                subscription=self.speech_key,
                region=self.speech_region,
            )
            speech_config.speech_recognition_language = locale

            audio_config = speechsdk.audio.AudioConfig(filename=tmp_path)
            recognizer   = speechsdk.SpeechRecognizer(
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

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return {"success": True, "text": result.text, "language": language}

        elif result.reason == speechsdk.ResultReason.NoMatch:
            return {
                "success": False,
                "error": "No speech was recognised. Please speak clearly and try again.",
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