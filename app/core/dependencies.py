from functools import lru_cache

from app.core.config import get_settings
from app.services.llm_service import LLMService
from app.services.tts_service import TTSService
from app.services.realtime_session_service import RealtimeSessionService
from app.services.sarvam_stt_service import SarvamSTTService


@lru_cache
def get_llm_service() -> LLMService:
    settings = get_settings()
    return LLMService(
        api_key=settings.openai_api_key,
        model=settings.openai_chat_model,
    )


@lru_cache
def get_tts_service() -> TTSService:
    settings = get_settings()
    return TTSService(
        api_key=settings.openai_api_key,
        voice_id=settings.openai_tts_voice,
        model=settings.openai_tts_model,
    )


@lru_cache
def get_realtime_session_service() -> RealtimeSessionService:
    return RealtimeSessionService()


@lru_cache
def get_sarvam_stt_service() -> SarvamSTTService:
    return SarvamSTTService()