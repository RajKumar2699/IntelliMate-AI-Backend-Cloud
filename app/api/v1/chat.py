from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.models.request import ChatRequest
from app.models.response import ChatResponse
from app.services.openai_service import openai_service

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        answer = openai_service.ask(request.message)
        return ChatResponse(
            success=True,
            response=answer
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@router.post("/chat/stream")
def chat_stream(request: ChatRequest):
    try:
        def generate():
            for chunk in openai_service.stream_answer(request.message):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream"
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))