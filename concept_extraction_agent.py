"""
Concept Extraction using OpenAI GPT-4o Vision (or Azure OpenAI)
Extracts important concepts from student notes images

Priority order for credentials:
1. OPENAI_KEY -> uses standard OpenAI API directly
2. AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY + AZURE_OPENAI_DEPLOYMENT_NAME -> uses Azure OpenAI
3. Fallback -> demo mode
"""
import json
import base64
import os
from io import BytesIO
from PIL import Image
import openai
from dotenv import load_dotenv

load_dotenv(override=True)

OPENAI_KEY = os.getenv("OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")


class ConceptExtractor:
    """Concept extractor using GPT-4o Vision (with demo mode fallback)"""

    @staticmethod
    def _get_demo_concepts():
        """
        Returns demo/mock concepts for testing when no API credentials are set.
        Includes normalized region coordinates (0.0-1.0) so the image highlighter works.
        """
        return {
            "concepts": [
                {
                    "id": "concept_1",
                    "name": "Photosynthesis",
                    "summary": "Process by which green plants convert sunlight into energy using chlorophyll.",
                    "category": "Biology",
                    "region": {"x1": 0.05, "y1": 0.05, "x2": 0.45, "y2": 0.25}
                },
                {
                    "id": "concept_2",
                    "name": "Chlorophyll",
                    "summary": "Green pigment in plant cells that absorbs sunlight for photosynthesis.",
                    "category": "Biology",
                    "region": {"x1": 0.05, "y1": 0.30, "x2": 0.45, "y2": 0.50}
                },
                {
                    "id": "concept_3",
                    "name": "Glucose",
                    "summary": "A simple sugar produced as the output of photosynthesis, used for plant energy.",
                    "category": "Chemistry",
                    "region": {"x1": 0.05, "y1": 0.55, "x2": 0.45, "y2": 0.75}
                }
            ],
            "source": "demo"
        }

    @staticmethod
    def _get_openai_client():
        """
        Returns the appropriate OpenAI client based on available credentials.
        Returns (client, model_name) tuple.
        """
        if OPENAI_KEY:
            print("[Concept Extraction] Using standard OpenAI API key")
            client = openai.OpenAI(api_key=OPENAI_KEY)
            return client, "gpt-4o"
        elif AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY:
            print("[Concept Extraction] Using Azure OpenAI")
            client = openai.AzureOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                api_version="2024-02-15-preview",
                azure_endpoint=AZURE_OPENAI_ENDPOINT
            )
            return client, AZURE_OPENAI_DEPLOYMENT_NAME
        return None, None

    @staticmethod
    def extract_concepts_from_highlighted_region(image_data, highlight_box=None, use_demo=False):
        """
        Extracts concepts from the image using GPT-4o Vision.
        Returns a dict: {"concepts": [...], "source": "openai"|"azure"|"demo"}
        Each concept has: id, name, summary, category, region (normalized x1/y1/x2/y2)
        """
        client, model_name = ConceptExtractor._get_openai_client()

        if use_demo or client is None:
            print("[Concept Extraction] Using demo mode (no API credentials configured)")
            return ConceptExtractor._get_demo_concepts()

        try:
            # Decode base64 image
            image = Image.open(BytesIO(base64.b64decode(image_data)))

            # If a highlight box is provided, crop to that region
            if highlight_box:
                cropped = image.crop(highlight_box)
            else:
                cropped = image

            # Convert image to PNG bytes for the API
            buffered = BytesIO()
            cropped.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()

            system_prompt = """You are an expert at extracting key concepts from student notes images.
Analyze the image and identify the main concepts/topics visible in it.
You MUST respond with ONLY a valid JSON array — no markdown, no explanation, just the raw JSON.

Each object in the array must have:
- "id": unique string like "concept_1", "concept_2", etc.
- "name": short concept name (2-5 words)
- "summary": one clear sentence explaining the concept
- "category": subject area (e.g. "Biology", "Math", "Physics", "Chemistry", "History", "General")
- "region": normalized bounding box (values 0.0 to 1.0) where concept appears in the image:
  { "x1": float, "y1": float, "x2": float, "y2": float }
  (x1,y1 = top-left, x2,y2 = bottom-right)

Example:
[
  {
    "id": "concept_1",
    "name": "Newton's Second Law",
    "summary": "Force equals mass times acceleration, describing how objects respond to forces.",
    "category": "Physics",
    "region": {"x1": 0.05, "y1": 0.08, "x2": 0.55, "y2": 0.28}
  }
]
"""

            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {
                            "type": "text",
                            "text": "Extract the main concepts from this student notes image and return them as a JSON array."
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                        }
                    ]}
                ],
                max_tokens=1024,
                temperature=0.2
            )

            raw_content = response.choices[0].message.content.strip()
            print(f"[Concept Extraction] Response received ({len(raw_content)} chars)")

            # Strip markdown code fences if GPT wraps the JSON in ```
            if raw_content.startswith("```"):
                lines = raw_content.split("\n")
                # Remove first line (```json or ```) and last line (```)
                raw_content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

            concepts_list = json.loads(raw_content)

            # Ensure all concepts have required fields with fallback defaults
            for i, c in enumerate(concepts_list):
                if "id" not in c:
                    c["id"] = f"concept_{i+1}"
                if "region" not in c:
                    step = 1.0 / max(len(concepts_list), 1)
                    c["region"] = {"x1": 0.0, "y1": i * step, "x2": 1.0, "y2": (i + 1) * step}
                if "category" not in c:
                    c["category"] = "General"
                if "summary" not in c:
                    c["summary"] = c.get("description", "")

            source = "openai" if OPENAI_KEY else "azure"
            print(f"[Concept Extraction] Extracted {len(concepts_list)} concepts via {source}")
            return {"concepts": concepts_list, "source": source}

        except json.JSONDecodeError as e:
            print(f"[Concept Extraction] JSON parse error: {e}")
            return ConceptExtractor._get_demo_concepts()
        except Exception as e:
            print(f"[Concept Extraction] Error: {e}")
            return ConceptExtractor._get_demo_concepts()
