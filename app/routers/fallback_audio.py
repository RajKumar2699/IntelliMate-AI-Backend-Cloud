from fastapi import APIRouter, Depends, HTTPException

from app.schemas.fallback_audio import (
    FallbackTranscriptionRequest,
    FallbackTranscriptionResponse,
)
from app.services.sarvam_stt_service import SarvamSTTService
from app.core.dependencies import get_sarvam_stt_service

router = APIRouter(prefix="/api/v1/fallback", tags=["fallback"])


@router.post("/transcribe", response_model=FallbackTranscriptionResponse)
async def fallback_transcribe(
    request: FallbackTranscriptionRequest,
    service: SarvamSTTService = Depends(get_sarvam_stt_service),
):
    try:
        transcript, language_code = await service.transcribe_base64_wav(
            audio_base64=request.audio_base64,
            language_code=request.language_code,
        )
        return FallbackTranscriptionResponse(
            transcript=transcript,
            language_code=language_code,
        )
    except Exception:
        # Avoid leaking internal exception details to the client.
        raise HTTPException(status_code=500, detail="Transcription failed")