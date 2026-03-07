"""
QuizGenerator — uses OpenAI (gpt-4o) to:
  1. Generate a spoken quiz question about a concept
  2. Evaluate a student's transcribed answer and produce spoken feedback

Required .env variables:
    OPENAI_KEY   your OpenAI API key
"""

import os
import requests


class QuizGenerator:

    API_URL = "https://api.openai.com/v1/chat/completions"
    MODEL   = "gpt-4o"

    def __init__(self):
        self.api_key = os.getenv("OPENAI_KEY", "")
        if not self.api_key:
            raise EnvironmentError("OPENAI_KEY is not set in .env")

    # ------------------------------------------------------------------ #
    #  Internal helper                                                     #
    # ------------------------------------------------------------------ #
    def _call(self, prompt: str, max_tokens: int = 300) -> str:
        """Send a single-turn request to the OpenAI Chat Completions API."""
        response = requests.post(
            self.API_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type":  "application/json",
            },
            json={
                "model":      self.MODEL,
                "max_tokens": max_tokens,
                "messages":   [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    # ------------------------------------------------------------------ #
    #  Question generation                                                 #
    # ------------------------------------------------------------------ #
    def generate_question(
        self,
        concept_name: str,
        concept_description: str,
        difficulty: str = "medium",
    ) -> dict:
        """
        Generate one spoken quiz question for the given concept.

        Returns:
            { "success": True,  "question": str }
          | { "success": False, "error":    str }
        """
        difficulty_guidance = {
            "easy":   "Focus on basic recall and simple understanding.",
            "medium": "Test application of the concept to a real scenario.",
            "hard":   "Require analysis, synthesis, or critical evaluation.",
        }.get(difficulty, "Test application of the concept to a real scenario.")

        prompt = (
            f"You are an engaging educational quiz master creating spoken quiz questions.\n\n"
            f"Concept: {concept_name}\n"
            f"Description: {concept_description}\n"
            f"Difficulty: {difficulty} — {difficulty_guidance}\n\n"
            f"Create ONE clear, thought-provoking question that:\n"
            f"- Tests genuine understanding (not surface-level recall)\n"
            f"- Sounds natural when read aloud\n"
            f"- Is concise (1–2 sentences max)\n\n"
            f"Return ONLY the question text. No preamble, no numbering, no explanation."
        )

        try:
            question = self._call(prompt, max_tokens=150)
            return {"success": True, "question": question.strip()}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    # ------------------------------------------------------------------ #
    #  Answer evaluation                                                   #
    # ------------------------------------------------------------------ #
    def evaluate_answer(
        self,
        concept_name: str,
        concept_description: str,
        question: str,
        student_answer: str,
    ) -> dict:
        """
        Evaluate a student's answer and return conversational spoken feedback.

        Returns:
            {
                "success":  True,
                "rating":   "correct" | "partially_correct" | "incorrect",
                "feedback": str
            }
          | { "success": False, "error": str }
        """
        prompt = (
            f"You are a warm, encouraging educational tutor evaluating a student's spoken answer.\n\n"
            f"Concept: {concept_name}\n"
            f"Reference description: {concept_description}\n"
            f"Question asked: {question}\n"
            f"Student's answer: \"{student_answer}\"\n\n"
            f"Provide spoken feedback that:\n"
            f"1. Acknowledges what the student got right\n"
            f"2. Gently corrects any misconceptions or gaps\n"
            f"3. Offers the key insight they may have missed\n"
            f"4. Ends with brief encouragement\n\n"
            f"Use natural, conversational language — no bullet points. Keep it to 3–5 sentences.\n\n"
            f"Respond in exactly this format (no extra text):\n"
            f"RATING: <correct|partially_correct|incorrect>\n"
            f"FEEDBACK: <your spoken feedback>"
        )

        try:
            raw = self._call(prompt, max_tokens=450)

            rating   = "partially_correct"
            feedback = raw

            for i, line in enumerate(raw.split("\n")):
                if line.startswith("RATING:"):
                    rating = line.replace("RATING:", "").strip().lower()
                elif line.startswith("FEEDBACK:"):
                    feedback = "\n".join(raw.split("\n")[i:]).replace("FEEDBACK:", "", 1).strip()
                    break

            if rating not in ("correct", "partially_correct", "incorrect"):
                rating = "partially_correct"

            return {"success": True, "rating": rating, "feedback": feedback}

        except Exception as exc:
            return {"success": False, "error": str(exc)}