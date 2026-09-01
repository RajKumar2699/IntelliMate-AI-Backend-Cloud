from pydantic import BaseModel

class ChatResponse(BaseModel):
    success: bool
    response: str