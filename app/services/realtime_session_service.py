import logging
from openai import AsyncOpenAI
from app.core.config import get_settings
from app.core.prompts import build_realtime_system_prompt

logger = logging.getLogger(__name__)


class RealtimeSessionService:
    def __init__(self):
        settings = get_settings()
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_realtime_model
        self.default_voice = settings.openai_tts_voice or "alloy"

    async def create_session(
        self,
        language_hint: str | None = None,
        voice: str | None = None,
    ) -> dict:
        instructions = build_realtime_system_prompt(language_hint=language_hint)

        session = await self.client.realtime.client_secrets.create(
            session={
                "type": "realtime",
                "model": self.model,
                "instructions": instructions,
                "audio": {
                    "output": {
                        "voice": voice or self.default_voice,
                    }
                },
            }
        )

        # The ephemeral token is a top-level field on the response, not nested
        # under session.client_secret — the "client_secret" sub-object you see
        # in the docs lives inside the *effective session config*, mirrored
        # for reference, not the field to read the token from.
        client_secret = getattr(session, "value", None)
        if not client_secret:
            raise RuntimeError("Realtime client secret was not returned")

        expires_at = getattr(session, "expires_at", None)

        # The session id lives on the nested effective session object.
        nested_session = getattr(session, "session", None)
        session_id = getattr(nested_session, "id", None)

        return {
            "client_secret": client_secret,
            "expires_at": expires_at,
            "session_id": session_id,
            "model": self.model,
            "voice": voice or self.default_voice,
        }