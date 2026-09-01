from app.core.config import get_settings
from app.services.llm_service import LLMService
from app.services.tts_service import TTSService
from app.services.realtime_session_service import RealtimeSessionService
from app.services.sarvam_stt_service import SarvamSTTService

settings = get_settings()

llm_service = LLMService(
    api_key=settings.openai_api_key,
    model=settings.openai_chat_model,
)

tts_service = TTSService(
    api_key=settings.openai_api_key,
    voice_id=settings.openai_tts_voice,
    model=settings.openai_tts_model,
)

realtime_session_service = RealtimeSessionService()
sarvam_stt_service = SarvamSTTService()