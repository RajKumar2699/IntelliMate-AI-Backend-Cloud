from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.core.config import get_settings
from app.api.v1.chat import router as chat_router
from app.api.v1.resume_builder import router as resume_builder_router
from app.routers.realtime import router as realtime_router
from app.routers.fallback_audio import router as fallback_audio_router
from app.routers.interview_enrollment import router as interview_enroll_router
from app.routers.interview_ws import router as interview_ws_router
from app.routers import interview_enrollment, interview_ws
from app.api.v1.interview.mac_ws import router as mac_ws_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_settings()
    yield


app = FastAPI(
    title="IntelliMate AI Backend",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(chat_router, prefix="/api/v1", tags=["Chat"])
app.include_router(resume_builder_router, prefix="/api/v1")
app.include_router(realtime_router)
app.include_router(fallback_audio_router)
app.include_router(interview_enroll_router)
app.include_router(interview_ws_router)
app.include_router(mac_ws_router)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def root():
    return {"message": "Backend Running 🚀"}