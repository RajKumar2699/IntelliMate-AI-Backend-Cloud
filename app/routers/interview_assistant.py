from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas.interview_audio import VoiceEnrollmentResponse
from app.services.speaker_embedding_service import SpeakerEmbeddingService
from app.services.voice_profile_store import VoiceProfileStore

router = APIRouter(prefix="/api/v1/interview", tags=["Interview Assistant"])

profile_store = VoiceProfileStore()
embedding_service = SpeakerEmbeddingService()


def detect_suffix(filename: str | None) -> str:
    if not filename:
        return ".wav"
    lowered = filename.lower()
    for ext in [".wav", ".m4a", ".mp3", ".mp4", ".mpeg", ".mpga", ".webm"]:
        if lowered.endswith(ext):
            return ext
    return ".wav"


@router.post("/enroll", response_model=VoiceEnrollmentResponse)
async def enroll_candidate(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Uploaded audio is empty")

    suffix = detect_suffix(file.filename)
    embedding = embedding_service.embedding_from_bytes(audio_bytes, suffix=suffix)
    profile_id = profile_store.create_profile("candidate", embedding)

    return VoiceEnrollmentResponse(
        profile_id=profile_id,
        message="Candidate voice enrolled successfully",
    )