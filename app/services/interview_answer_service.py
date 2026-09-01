from app.core.config import get_settings
from app.services.llm_service import LLMService


class InterviewAnswerService:
    def __init__(self):
        settings = get_settings()
        self.llm = LLMService(
            api_key=settings.openai_api_key,
            model=settings.openai_chat_model,
        )

    async def generate_answer(self, interviewer_text: str) -> str:
        if not interviewer_text.strip():
            return ""

        conversation = [
            {
                "role": "system",
                "content": (
                    "You are a technical interview answer assistant. "
                    "Given the interviewer's question, generate a concise, strong, truthful candidate answer. "
                    "Always respond in English. "
                    "Keep it practical, short, and easy to read during a live interview. "
                    "Do not use markdown or bullets unless essential."
                ),
            },
            {
                "role": "user",
                "content": interviewer_text,
            },
        ]
        return await self.llm.get_reply(conversation)