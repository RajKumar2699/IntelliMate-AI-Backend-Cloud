from pydantic import BaseModel, Field


class RealtimeSessionRequest(BaseModel):
    language_hint: str | None = Field(default=None)
    voice: str | None = Field(default=None)


class RealtimeSessionResponse(BaseModel):
    client_secret: str
    expires_at: int | None = None
    session_id: str | None = None
    model: str
    voice: str | None = None