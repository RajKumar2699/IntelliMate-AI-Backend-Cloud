from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import secrets
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.config import get_settings
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)
router = APIRouter()

settings = get_settings()

llm_service = LLMService(
    api_key=settings.openai_api_key,
    model=settings.openai_chat_model,
)

# Bound how much raw text we'll ever push into the LLM in one go.
MAX_TRANSCRIPT_CHARS = 4000
MIN_QUESTION_CHARS = 12
MIN_PARTIAL_WORD_COUNT = 5

# How long the speaker must pause before we draft an answer from a partial
# transcript. Every new partial resets this timer, so a single long question
# only produces one draft answer (after the speaker stops), instead of a new
# answer every couple of seconds while they're still talking.
PARTIAL_DEBOUNCE_SECONDS = 0.5


@dataclass
class TranscriptSession:
    session_id: str
    transcript_text: str = ""
    normalized_question: str = ""
    answer_text: str = ""
    detected_language: Optional[str] = None
    target_role: str = "ios_developer"
    last_final_question: str = ""
    last_partial_question: str = ""
    last_answer_at: float = 0.0
    # Tracks the last normalized question we actually sent to the LLM,
    # so we don't re-answer the same question twice.
    last_llm_input: str = ""
    # In-flight debounce timer for partial transcripts. Cancelled and
    # replaced every time a new partial arrives.
    pending_task: Optional[asyncio.Task] = field(default=None, repr=False, compare=False)


ALLOWED_ROLES = {
    "ios_developer",
    "android_developer",
    "python_developer",
    "mern_developer",
    "backend_developer",
    "developer",
}


ROLE_CONTEXT = {
    "ios_developer": (
        "Technical interview for an iOS developer. Prefer Swift, UIKit, SwiftUI, MVVM, "
        "Clean Architecture, Core Data, URLSession, GCD, async await, ARC, memory management, "
        "Auto Layout, WebSocket, persistence, performance, and debugging."
    ),
    "android_developer": (
        "Technical interview for an Android developer. Prefer Kotlin, Android SDK, Jetpack Compose, "
        "ViewModel, coroutines, Room, Retrofit, Hilt, WorkManager, architecture, and performance."
    ),
    "python_developer": (
        "Technical interview for a Python developer. Prefer Python, FastAPI, Django, Flask, asyncio, "
        "SQLAlchemy, APIs, testing, backend development, concurrency, and clean architecture."
    ),
    "mern_developer": (
        "Technical interview for a MERN developer. Prefer MongoDB, Express.js, React, Node.js, "
        "TypeScript, REST APIs, authentication, deployment, state management, and full-stack architecture."
    ),
    "backend_developer": (
        "Technical interview for a backend developer. Prefer APIs, databases, caching, queues, "
        "distributed systems, scalability, observability, security, and system design."
    ),
    "developer": (
        "General software developer technical interview. Interpret conservatively as a technical question."
    ),
}


def is_authorized(websocket: WebSocket, init_payload: dict) -> bool:
    expected_token = getattr(settings, "mac_client_auth_token", None)

    if not expected_token:
        return True

    header = websocket.headers.get("authorization", "")
    if header.startswith("Bearer "):
        candidate = header[len("Bearer "):]
        if secrets.compare_digest(candidate, expected_token):
            return True

    provided = init_payload.get("token")
    if isinstance(provided, str) and secrets.compare_digest(provided, expected_token):
        return True

    return False


async def send_event(
    websocket: WebSocket,
    event_type: str,
    *,
    text: Optional[str] = None,
    answer: Optional[str] = None,
    status: Optional[str] = None,
    message: Optional[str] = None,
    session_id: Optional[str] = None,
    normalized_question: Optional[str] = None,
    target_role: Optional[str] = None,
) -> None:
    payload = {"type": event_type}

    if text is not None:
        payload["text"] = text
    if answer is not None:
        payload["answer"] = answer
    if status is not None:
        payload["status"] = status
    if message is not None:
        payload["message"] = message
    if session_id is not None:
        payload["sessionid"] = session_id
    if normalized_question is not None:
        payload["normalized_question"] = normalized_question
    if target_role is not None:
        payload["target_role"] = target_role

    try:
        await websocket.send_text(json.dumps(payload))
    except Exception:
        logger.debug("Failed to send event %s; socket likely closed.", event_type)


def validate_start_payload(payload: dict) -> Optional[str]:
    if payload.get("type") != "startsession":
        return "Invalid start message type."

    if payload.get("source") != "remote_system_audio_text":
        return "Invalid source. Expected remote_system_audio_text."

    return None


def sanitize_role(value: Optional[str]) -> str:
    role = (value or "ios_developer").strip().lower()
    return role if role in ALLOWED_ROLES else "developer"


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def looks_question_like(text: str) -> bool:
    lowered = text.lower()
    if "?" in text:
        return True

    starters = (
        "what", "why", "how", "when", "where", "which",
        "can", "could", "would", "do", "did", "have",
        "has", "explain", "tell", "walk"
    )
    return lowered.startswith(starters)


def should_generate_answer(
    session: TranscriptSession,
    normalized_question: str,
    is_final: bool,
) -> bool:
    """
    Decide whether the (already-normalized) question is worth sending to the
    LLM for an answer. Mutates session.last_llm_input as a side effect when
    it decides to proceed, so callers should only call this once per
    transcript event.

    Timing (debouncing partials so we don't answer mid-sentence) is handled
    by the caller via PARTIAL_DEBOUNCE_SECONDS. This function is just the
    final dedup/sanity gate: don't re-answer the same question twice, and
    don't bother answering something too short to be a real question.
    """
    normalized = normalized_question.strip()

    if not normalized:
        return False

    if len(normalized) < MIN_QUESTION_CHARS:
        return False

    # Never re-answer the exact same normalized question twice in a row.
    if normalized == session.last_llm_input:
        return False

    if not is_final and len(normalized.split()) < MIN_PARTIAL_WORD_COUNT:
        return False

    session.last_llm_input = normalized
    return True


async def normalize_transcript(
    session: TranscriptSession,
    raw_transcript: str,
    *,
    is_final: bool,
) -> str:
    raw_text = normalize_whitespace(raw_transcript)[:MAX_TRANSCRIPT_CHARS]
    if not raw_text:
        return ""

    role_context = ROLE_CONTEXT.get(session.target_role, ROLE_CONTEXT["developer"])
    locale_hint = session.detected_language or "unspecified"

    try:
        normalized = await llm_service.get_reply(
            conversation=[
                {
                    "role": "user",
                    "content": (
                        f"Target role: {session.target_role}\n"
                        f"Role context: {role_context}\n"
                        f"Speaker locale/accent hint: {locale_hint}\n"
                        f"Raw Apple Speech transcript: {raw_text}\n"
                        f"Transcript state: {'final' if is_final else 'partial'}\n\n"
                        "Rewrite this into the most likely technical interview question. "
                        "Repair ASR mistakes in framework names, APIs, libraries, architecture terms, and developer jargon. "
                        "Do not answer the question. "
                        "Do not explain changes. "
                        "Always return a likely technical question, even if partial. "
                        "If uncertain, stay conservative and close to the original wording."
                    ),
                }
            ],
            detected_language=session.detected_language,
            system_instruction=(
                "You are a technical ASR correction engine for live interview transcripts. "
                "Your job is to repair speech recognition mistakes and infer the most likely technical interview question. "
                "The audio may be spoken in a non-US English accent (e.g. Indian English) and the "
                "on-device speech recognizer frequently mis-hears it: articles and short words get "
                "dropped, technical terms get mangled into similar-sounding but wrong words (e.g. "
                "'view controller' misheard as 'view controllers' or split oddly, 'coroutine' misheard "
                "as 'core routine'), and words may be transliterated oddly. Treat unfamiliar or "
                "malformed words as likely mis-transcriptions of a technical term that fits the role "
                "context, not as intentional phrasing. "
                "Never return OUT_OF_SCOPE. "
                "Never refuse. "
                "Never explain. "
                "Return only the corrected question text."
            ),
            temperature=0.0,
            max_output_tokens=80,
        )
    except Exception:
        logger.exception("Transcript normalization failed for session %s", session.session_id)
        return raw_text

    normalized = normalize_whitespace(normalized)
    return normalized or raw_text


def build_answer_conversation(
    normalized_question: str,
    target_role: str,
) -> list[dict]:
    role_context = ROLE_CONTEXT.get(target_role, ROLE_CONTEXT["developer"])

    return [
        {
            "role": "user",
            "content": (
                f"Target role: {target_role}\n"
                f"Role context: {role_context}\n"
                f"Normalized interviewer question: {normalized_question}\n\n"
                "Answer this as the candidate in a technical interview."
            ),
        }
    ]


async def generate_answer(
    websocket: WebSocket,
    session: TranscriptSession,
    *,
    is_final: bool,
) -> None:
    normalized_question = normalize_whitespace(session.normalized_question)
    if not normalized_question:
        return

    if is_final:
        session.last_final_question = normalized_question
    else:
        session.last_partial_question = normalized_question

    session.last_answer_at = time.monotonic()

    role_context = ROLE_CONTEXT.get(session.target_role, ROLE_CONTEXT["developer"])

    try:
        answer = await llm_service.get_reply(
            conversation=build_answer_conversation(
                normalized_question=normalized_question,
                target_role=session.target_role,
            ),
            detected_language=session.detected_language,
            system_instruction=(
                f"""
You are an expert AI Interview Copilot for technical interviews.

Target role: {session.target_role}
Role context: {role_context}

Rules:
1. Always answer the latest normalized interviewer question.
2. Answer from a technical interview perspective.
3. Tailor the answer to the target role.
4. If role is ios_developer, prefer Swift/UIKit/SwiftUI/Core Data/MVVM/iOS examples where relevant.
5. If role is android_developer, prefer Kotlin/Android/Jetpack examples where relevant.
6. If role is python_developer, prefer Python/FastAPI/Django/backend examples where relevant.
7. If role is mern_developer, prefer MongoDB/Express/React/Node examples where relevant.
8. If role is backend_developer, prefer backend, API, database, scalability, caching, and system-design examples where relevant.
9. If the question is partial, give a concise draft answer based on the most likely intended meaning.
10. If the question is final, give a complete polished answer.
11. Use natural spoken English and first-person wording where appropriate.
12. Never say OUT_OF_SCOPE.
13. Never refuse unless the input is truly empty.
"""
            ),
            temperature=0.2,
            max_output_tokens=120 if not is_final else 220,
        )
    except Exception as exc:
        # Log full detail server-side, but never echo raw exception text to
        # the client — it can leak internal error bodies / stack info.
        logger.exception(
            "LLM answer generation failed for session %s", session.session_id
        )
        await send_event(
            websocket,
            "error",
            message="Answer generation failed. Please try again.",
            session_id=session.session_id,
        )
        return

    session.answer_text = answer

    await send_event(
        websocket,
        "answer",
        text=session.transcript_text,
        answer=answer,
        status="Answer ready" if is_final else "Draft answer ready",
        session_id=session.session_id,
        normalized_question=normalized_question,
        target_role=session.target_role,
    )


def _cancel_pending_task(session: TranscriptSession) -> None:
    if session.pending_task and not session.pending_task.done():
        session.pending_task.cancel()
    session.pending_task = None


async def _debounced_partial_answer(
    websocket: WebSocket,
    session: TranscriptSession,
) -> None:
    """
    Waits for a pause in speech before drafting an answer to a partial
    transcript. If a newer partial arrives first, process_transcript cancels
    this task and starts a fresh one, so only the settled question ever
    reaches the LLM.
    """
    try:
        await asyncio.sleep(PARTIAL_DEBOUNCE_SECONDS)
    except asyncio.CancelledError:
        return

    if should_generate_answer(session, session.normalized_question, is_final=False):
        await generate_answer(websocket=websocket, session=session, is_final=False)


async def process_transcript(
    websocket: WebSocket,
    session: TranscriptSession,
    transcript: str,
    is_final: bool,
) -> None:
    raw_text = normalize_whitespace(transcript)[:MAX_TRANSCRIPT_CHARS]
    if not raw_text:
        return

    session.transcript_text = raw_text
    session.normalized_question = await normalize_transcript(
        session=session,
        raw_transcript=raw_text,
        is_final=is_final,
    )

    await send_event(
        websocket,
        "transcript",
        text=raw_text,
        status="Transcript updated",
        session_id=session.session_id,
        normalized_question=session.normalized_question,
        target_role=session.target_role,
    )

    # Any earlier "wait for a pause" timer is now stale — a newer update
    # just arrived, so drop it before deciding what to do next.
    _cancel_pending_task(session)

    if is_final:
        if should_generate_answer(session, session.normalized_question, is_final=True):
            await generate_answer(websocket=websocket, session=session, is_final=True)
        return

    # Partial transcript: only worth answering if it already reads like a
    # real question. Even then, don't answer immediately — wait to see if
    # the speaker keeps talking (debounce), so one long question produces
    # at most one draft answer instead of one every couple of seconds.
    if not looks_question_like(session.normalized_question):
        return

    session.pending_task = asyncio.create_task(
        _debounced_partial_answer(websocket, session)
    )


@router.websocket("/api/v1/interview/mac/ws")
async def mac_system_audio_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    session: Optional[TranscriptSession] = None

    try:
        init_text = await websocket.receive_text()

        try:
            init_payload = json.loads(init_text)
        except json.JSONDecodeError:
            await send_event(websocket, "error", message="Invalid JSON in start message.")
            await websocket.close(code=1008)
            return

        if not is_authorized(websocket, init_payload):
            await send_event(websocket, "error", message="Unauthorized.")
            await websocket.close(code=1008)
            return

        error_message = validate_start_payload(init_payload)
        if error_message:
            await send_event(websocket, "error", message=error_message)
            await websocket.close(code=1008)
            return

        session_id = (
            init_payload.get("sessionid")
            or init_payload.get("session_id")
            or str(uuid.uuid4())
        )
        target_role = sanitize_role(init_payload.get("target_role"))
        # Accept a locale/language hint from the client (e.g. "en-IN") so the
        # ASR-correction step knows what kind of accent/mis-hearing pattern
        # to expect. Falls back to None (generic) if the client doesn't send one.
        detected_language = (
            init_payload.get("locale")
            or init_payload.get("language")
            or init_payload.get("detected_language")
        )

        session = TranscriptSession(
            session_id=session_id,
            target_role=target_role,
            detected_language=detected_language,
        )

        await send_event(
            websocket,
            "session_started",
            status=f"Session started for {target_role}",
            session_id=session.session_id,
            target_role=target_role,
        )

        while True:
            text = await websocket.receive_text()

            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                await send_event(
                    websocket,
                    "error",
                    message="Invalid JSON message.",
                    session_id=session.session_id,
                )
                continue

            message_type = payload.get("type")

            if message_type == "stop":
                _cancel_pending_task(session)
                await send_event(
                    websocket,
                    "status",
                    status="Session stopped",
                    session_id=session.session_id,
                )
                break

            if message_type != "transcript":
                await send_event(
                    websocket,
                    "error",
                    message="Unsupported message type.",
                    session_id=session.session_id,
                )
                continue

            transcript = payload.get("text", "")
            is_final = bool(payload.get("isFinal", False))

            await process_transcript(
                websocket=websocket,
                session=session,
                transcript=transcript,
                is_final=is_final,
            )

    except WebSocketDisconnect:
        logger.info(
            "Transcript websocket disconnected for session %s",
            session.session_id if session else "unknown",
        )

    except RuntimeError as exc:
        if "disconnect message has been received" in str(exc):
            logger.info(
                "Transcript websocket already disconnected for session %s",
                session.session_id if session else "unknown",
            )
        else:
            logger.exception("Transcript WebSocket runtime failure: %s", exc)
            with contextlib.suppress(Exception):
                await send_event(
                    websocket,
                    "error",
                    message="Internal server error.",
                    session_id=session.session_id if session else None,
                )

    except Exception:
        logger.exception(
            "Transcript WebSocket failed for session %s",
            session.session_id if session else "unknown",
        )
        with contextlib.suppress(Exception):
            await send_event(
                websocket,
                "error",
                message="Internal server error.",
                session_id=session.session_id if session else None,
            )

    finally:
        if session is not None:
            _cancel_pending_task(session)
        with contextlib.suppress(Exception):
            await websocket.close()