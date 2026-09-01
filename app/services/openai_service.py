import time
import logging
from openai import OpenAI
from app.core.config import get_settings
from app.core.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)
settings = get_settings()


class OpenAIService:
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_chat_model

    def ask(self, message: str) -> str:
        start = time.perf_counter()
        print("OpenAI request started...")
        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": message},
            ],
        )
        end = time.perf_counter()
        print(f"OpenAI took {end - start:.2f} seconds")
        return response.output_text.strip()

    def stream_answer(self, message: str):
        """
        Yields text chunks as they arrive, for use with StreamingResponse
        in the /chat/stream endpoint.
        """
        try:
            with self.client.responses.stream(
                model=self.model,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": message},
                ],
            ) as stream:
                for event in stream:
                    if event.type == "response.output_text.delta":
                        yield event.delta
        except Exception:
            logger.exception("OpenAI streaming failed")
            yield "Sorry, I ran into an error while responding."

    def improve_resume(self, resume_text: str, job_description: str) -> str:
        """
        Rewrites the resume to better align with the given job description.
        Used by ResumeBuilderService when a JD is provided.
        """
        prompt = (
            "You are an expert resume writer and ATS optimization specialist.\n"
            "Rewrite the resume below so it aligns strongly with the target "
            "job description. Keep it truthful — do not invent experience, "
            "companies, titles, or dates that aren't implied by the original. "
            "Improve wording, add relevant keywords from the job description "
            "where genuinely applicable, strengthen bullet points with "
            "measurable impact, and keep the overall structure resume-like "
            "(summary, experience, skills, education).\n\n"
            f"JOB DESCRIPTION:\n{job_description}\n\n"
            f"ORIGINAL RESUME:\n{resume_text}\n\n"
            "Return only the rewritten resume text, no commentary."
        )

        try:
            response = self.client.responses.create(
                model=self.model,
                input=[
                    {
                        "role": "system",
                        "content": "You rewrite resumes to align with job descriptions.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            content = response.output_text
            if not content or not content.strip():
                logger.warning("improve_resume returned empty content")
                return resume_text
            return content.strip()
        except Exception:
            logger.exception("Resume improvement failed")
            return resume_text


openai_service = OpenAIService()