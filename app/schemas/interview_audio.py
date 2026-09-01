from pydantic import BaseModel


class VoiceEnrollmentResponse(BaseModel):
    profile_id: str
    message: str