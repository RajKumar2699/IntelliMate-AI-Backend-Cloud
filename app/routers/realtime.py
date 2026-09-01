import logging
from fastapi import APIRouter, Depends, HTTPException

from app.schemas.realtime import RealtimeSessionRequest, RealtimeSessionResponse
from app.services.realtime_session_service import RealtimeSessionService
from app.core.dependencies import get_realtime_session_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/realtime", tags=["realtime"])


@router.post("/session", response_model=RealtimeSessionResponse)
async def create_realtime_session(
    request: RealtimeSessionRequest,
    service: RealtimeSessionService = Depends(get_realtime_session_service),
):
    try:
        data = await service.create_session(
            language_hint=request.language_hint,
            voice=request.voice,
        )
        return RealtimeSessionResponse(**data)
    except Exception:
        logger.exception("Failed to create realtime session")
        raise HTTPException(status_code=500, detail="Failed to create realtime session")