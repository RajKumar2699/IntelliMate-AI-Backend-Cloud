from pydantic import BaseModel, Field


class FallbackTranscriptionRequest(BaseModel):
    audio_base64: str
    mime_type: str = Field(default="audio/wav")
    language_code: str | None = Field(default="unknown")


class FallbackTranscriptionResponse(BaseModel):
    transcript: str
    language_code: str | None = None