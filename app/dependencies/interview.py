from openai import OpenAI

from app.core.config import get_settings
from app.services.interview_answer_service import InterviewAnswerService
from app.services.interview_session_service import InterviewSessionService
from app.services.speaker_embedding_service import SpeakerEmbeddingService
from app.services.voice_profile_store import VoiceProfileStore

settings = get_settings()

profile_store = VoiceProfileStore()
embedding_service = SpeakerEmbeddingService()
session_service = InterviewSessionService(profile_store, embedding_service)
answer_service = InterviewAnswerService()
openai_client = OpenAI(api_key=settings.openai_api_key)