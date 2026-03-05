"""
Concept Extraction using Azure OpenAI GPT-4o Vision
Extracts important concepts from student notes images

DEMO MODE: If Azure OpenAI is not configured, this uses mock data
"""
import json
import base64
import os
from io import BytesIO
from PIL import Image
import openai
from dotenv import load_dotenv

load_dotenv(override=True)

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

class ConceptExtractor:

        @staticmethod
        def _get_demo_concepts():
            """
            Returns demo/mock concepts for testing when Azure OpenAI is not configured.
            """
            return {
                "concepts": [
                    {"id": 1, "name": "Photosynthesis", "summary": "Process by which green plants convert sunlight into energy."},
                    {"id": 2, "name": "Chlorophyll", "summary": "Green pigment in plants that absorbs light for photosynthesis."},
                    {"id": 3, "name": "Glucose", "summary": "A simple sugar produced during photosynthesis."}
                ],
                "source": "demo"
            }
        """Concept extractor using GPT-4o Vision (with demo mode fallback)"""

        @staticmethod
        def extract_concepts_from_highlighted_region(image_data, highlight_box=None, use_demo=False):
            """
            Extracts concepts from the highlighted portion of an uploaded image using Azure OpenAI Vision.
            highlight_box: (left, upper, right, lower) tuple in pixel coordinates.
            If use_demo is True or credentials are missing, returns demo data.
            """
            if use_demo or not (AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY and AZURE_OPENAI_DEPLOYMENT_NAME):
                # Demo fallback
                return ConceptExtractor._get_demo_concepts()

            try:
                # Decode base64 image
                image = Image.open(BytesIO(base64.b64decode(image_data)))

                # If a highlight box is provided, crop to that region
                if highlight_box:
                    cropped = image.crop(highlight_box)
                else:
                    cropped = image

                # Convert cropped image to bytes for OpenAI API
                buffered = BytesIO()
                cropped.save(buffered, format="PNG")
                img_bytes = buffered.getvalue()

                # Call Azure OpenAI Vision (GPT-4o)
                client = openai.AzureOpenAI(
                    api_key=AZURE_OPENAI_API_KEY,
                    api_version="2024-02-15-preview",
                    azure_endpoint=AZURE_OPENAI_ENDPOINT
                )
                response = client.chat.completions.create(
                    model=AZURE_OPENAI_DEPLOYMENT_NAME,
                    messages=[
                        {"role": "system", "content": "You are an expert at extracting key concepts from handwritten notes. For each concept, provide a concise summary."},
                        {"role": "user", "content": [
                            {"type": "text", "text": "Extract the main concepts and a 1-sentence summary for each from this highlighted portion of handwritten notes."},
                            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(img_bytes).decode()}}
                        ]}
                    ],
                    max_tokens=512,
                    temperature=0.2
                )
                concepts = response.choices[0].message.content
                return {"concepts": concepts, "source": "azure"}
            except Exception as e:
                print(f"[Concept Extraction] Error: {e}")
                # Fallback to demo data
                return ConceptExtractor._get_demo_concepts()


