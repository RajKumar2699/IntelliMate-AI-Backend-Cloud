from __future__ import annotations

import asyncio
import os
import tempfile
import uuid
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.config import get_settings
from app.dependencies.interview import (
    answer_service,
    embedding_service,
    openai_client,
    profile_store,
)
from app.services.utterance_detector import UtteranceDetector

router = APIRouter(prefix="/api/v1/interview", tags=["Interview Assistant"])

settings = get_settings()
DEBUG_AUDIO_DIR = "data/debug_utterances"
if settings.debug:
    os.makedirs(DEBUG_AUDIO_DIR, exist_ok=True)


@dataclass
class InterviewSession:
    session_id: str
    candidate_profile_id: str
    transcript: list[dict] = field(default_factory=list)
    answered_question_keys: set[str] = field(default_factory=set)


def pcm16_to_wav_bytes(pcm_bytes: bytes, sample_rate: int = 16000, channels: int = 1) -> bytes:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp_path = tmp.name
    try:
        with wave.open(tmp_path, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_bytes)
        return Path(tmp_path).read_bytes()
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def pcm_rms(pcm_bytes: bytes) -> float:
    if not pcm_bytes:
        return 0.0
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(samples ** 2)))


def detect_language(text: str) -> str:
    devanagari_chars = sum(1 for c in text if "\u0900" <= c <= "\u097F")
    ascii_chars = sum(1 for c in text if c.isascii())
    total_chars = len(text)
    if total_chars == 0:
        return "unknown"
    if devanagari_chars / total_chars > 0.3:
        return "hi"
    if ascii_chars / total_chars > 0.7:
        return "en"
    return "mixed"


def is_valid_transcript(text: str) -> bool:
    return bool(text and len(text.strip()) >= 2)


def transcribe_wav_bytes(wav_bytes: bytes, language: str | None = None) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(wav_bytes)
        tmp_path = tmp.name
    try:
        with open(tmp_path, "rb") as audio_file:
            params: dict[str, Any] = {
                "model": settings.openai_stt_model,
                "file": audio_file,
            }
            if language in {"en", "hi"}:
                params["language"] = language
            transcript = openai_client.audio.transcriptions.create(**params)
        text = getattr(transcript, "text", None) or (
            transcript.get("text", "") if isinstance(transcript, dict) else ""
        )
        return text.strip()
    finally:
        Path(tmp_path).unlink(missing_ok=True)


async def generate_and_send_answer(websocket: WebSocket, question: str, language: str = "en") -> None:
    try:
        answer = await answer_service.generate_answer(question, language=language)
        await websocket.send_json(
            {
                "type": "answer",
                "question": question,
                "answer": answer,
                "language": language,
            }
        )
    except Exception as e:
        await websocket.send_json({"type": "error", "message": f"Answer generation failed: {str(e)}"})


def make_question_key(text: str) -> str:
    return " ".join(text.lower().strip().split())


async def classify_and_process_utterance(
    websocket: WebSocket,
    session: InterviewSession,
    utterance_pcm: bytes,
) -> None:
    rms = pcm_rms(utterance_pcm)
    duration_s = len(utterance_pcm) / (
        settings.audio_sample_rate * settings.audio_channels * settings.audio_sample_width_bytes
    )

    if duration_s < settings.interview_min_utterance_seconds or rms < settings.interview_low_rms_threshold:
        return

    utterance_wav = pcm16_to_wav_bytes(
        utterance_pcm,
        sample_rate=settings.audio_sample_rate,
        channels=settings.audio_channels,
    )

    emb = await asyncio.to_thread(embedding_service.embedding_from_bytes, utterance_wav, ".wav")
    candidate_embedding = profile_store.load_embedding(session.candidate_profile_id)
    self_score = embedding_service.similarity(emb, candidate_embedding)

    # Three-zone gate: confidently you -> drop, confidently not-you -> process,
    # ambiguous middle band -> drop rather than guess.
    if self_score >= settings.interview_weak_accept:
        await websocket.send_json({"type": "debug_dropped", "reason": "self_voice", "score": self_score})
        return

    if self_score > settings.interview_reject_below:
        await websocket.send_json({"type": "debug_dropped", "reason": "ambiguous_speaker", "score": self_score})
        return

    if settings.debug:
        debug_path = os.path.join(DEBUG_AUDIO_DIR, f"{uuid.uuid4()}.wav")
        with open(debug_path, "wb") as f:
            f.write(utterance_wav)

    text = await asyncio.to_thread(transcribe_wav_bytes, utterance_wav)
    if not text or not is_valid_transcript(text):
        return

    qkey = make_question_key(text)
    if qkey in session.answered_question_keys:
        return
    session.answered_question_keys.add(qkey)

    detected_lang = detect_language(text)
    session.transcript.append(
        {
            "speaker": "manager",
            "text": text,
            "language": detected_lang,
            "self_score": self_score,
        }
    )

    await websocket.send_json(
        {
            "type": "transcript",
            "speaker": "manager",
            "similarity": self_score,
            "text": text,
            "language": detected_lang,
        }
    )

    asyncio.create_task(generate_and_send_answer(websocket, text, detected_lang))


@router.websocket("/ws")
async def interview_ws(websocket: WebSocket):
    await websocket.accept()

    try:
        init_payload = await websocket.receive_json()
        candidate_profile_id = init_payload.get("candidate_profile_id")

        if not candidate_profile_id:
            await websocket.send_json({"type": "error", "message": "candidate_profile_id is required"})
            await websocket.close(code=1008)
            return

        if not profile_store.exists(candidate_profile_id):
            await websocket.send_json({"type": "error", "message": "Invalid candidate_profile_id"})
            await websocket.close(code=1008)
            return

        session = InterviewSession(
            session_id=str(uuid.uuid4()),
            candidate_profile_id=candidate_profile_id,
        )

        detector = UtteranceDetector(
            silence_rms_threshold=140.0,
            silence_hangover_chunks=10,
            min_speech_chunks=3,
            max_utterance_chunks=90,
        )

        await websocket.send_json(
            {
                "type": "session_started",
                "session_id": session.session_id,
                "candidate_profile_id": candidate_profile_id,
            }
        )

        async def receive_loop():
            while True:
                chunk = await websocket.receive_bytes()
                if not chunk:
                    continue

                utterance = detector.process_chunk(chunk)
                if utterance:
                    asyncio.create_task(classify_and_process_utterance(websocket, session, utterance))

        await receive_loop()

    except WebSocketDisconnect:
        return
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
            await websocket.close(code=1011)
        except Exception:
            pass