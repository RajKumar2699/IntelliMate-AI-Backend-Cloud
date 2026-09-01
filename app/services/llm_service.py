import logging
from openai import AsyncOpenAI


logger = logging.getLogger(__name__)


class LLMService:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def get_reply(
        self,
        conversation: list[dict],
        detected_language: str | None = None,
        system_instruction: str | None = None,
        temperature: float = 0.3,
        max_output_tokens: int = 300,
    ) -> str:
        try:
            input_messages = []

            if system_instruction:
                input_messages.append(
                    {
                        "role": "system",
                        "content": system_instruction,
                    }
                )

            if detected_language:
                input_messages.append(
                    {
                        "role": "system",
                        "content": (
                            f"Reply in {detected_language}. "
                            "Use the same language as the user's latest utterance. "
                            "Do not translate unless asked."
                        ),
                    }
                )

            input_messages.extend(conversation)

            response = await self.client.responses.create(
                model=self.model,
                input=input_messages,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )

            content = (getattr(response, "output_text", "") or "").strip()

            if not content:
                logger.warning("LLM returned empty content")
                return "I am not sure how to answer that yet."

            return content

        except Exception as error:
            logger.exception("OpenAI reply generation failed: %s", error)
            return "I am having trouble responding right now."