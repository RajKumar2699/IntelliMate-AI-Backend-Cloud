from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.config import get_settings
from app.dependencies.interview import embedding_service, profile_store
from app.schemas.interview_audio import VoiceEnrollmentResponse

router = APIRouter(prefix="/api/v1/interview", tags=["Interview Assistant"])


def detect_suffix(filename: str | None) -> str:
    if not filename:
        return ".wav"

    lowered = filename.lower()
    for ext in [".wav", ".m4a", ".mp3", ".mp4", ".mpeg", ".mpga", ".webm", ".caf"]:
        if lowered.endswith(ext):
            return ext

    return ".wav"


@router.post("/enroll", response_model=VoiceEnrollmentResponse)
async def enroll_candidate(files: list[UploadFile] = File(...)):
    settings = get_settings()

    if len(files) < settings.interview_min_enroll_files:
        raise HTTPException(
            status_code=400,
            detail=f"Please upload at least {settings.interview_min_enroll_files} voice samples",
        )

    embeddings = []
    valid_files = 0

    for file in files:
        audio_bytes = await file.read()
        if not audio_bytes:
            continue

        suffix = detect_suffix(file.filename)

        try:
            embedding = embedding_service.embedding_from_bytes(audio_bytes, suffix=suffix)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to process audio file '{file.filename or 'unknown'}': {str(e)}",
            )

        embeddings.append(embedding)
        valid_files += 1

    if valid_files < settings.interview_min_enroll_files:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough valid enrollment samples. Valid files: {valid_files}",
        )

    centroid = embedding_service.centroid(embeddings)
    profile_id = profile_store.create_profile("candidate", centroid)

    return VoiceEnrollmentResponse(
        profile_id=profile_id,
        message=f"Candidate voice enrolled successfully with {valid_files} samples",
    )