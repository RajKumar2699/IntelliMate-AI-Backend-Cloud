from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from app.core.config import get_settings
from app.services.speaker_embedding_service import SpeakerEmbeddingService
from app.services.voice_profile_store import VoiceProfileStore

SpeakerLabel = Literal["candidate", "interviewer", "unknown"]


@dataclass
class InterviewSession:
    session_id: str
    candidate_profile_id: str
    interviewer_profile_id: str | None = None
    transcript: list[dict] = field(default_factory=list)

    diarized_speaker_map: dict[str, str] = field(default_factory=dict)
    diarized_candidate_scores: dict[str, float] = field(default_factory=dict)
    interviewer_confirmed: bool = False
    processed_segment_keys: set[str] = field(default_factory=set)
    answered_question_keys: set[str] = field(default_factory=set)

    candidate_score_history: list[float] = field(default_factory=list)
    interviewer_score_history: list[float] = field(default_factory=list)
    interviewer_embedding_buffer: list[np.ndarray] = field(default_factory=list)


class InterviewSessionService:
    def __init__(
        self,
        profile_store: VoiceProfileStore,
        embedding_service: SpeakerEmbeddingService,
    ) -> None:
        settings = get_settings()
        self.profile_store = profile_store
        self.embedding_service = embedding_service
        self.strong_accept = settings.interview_strong_accept
        self.weak_accept = settings.interview_weak_accept
        self.reject_below = settings.interview_reject_below

    def update_diarized_mapping(
        self,
        session: InterviewSession,
        speaker_map: dict[str, str],
        score_map: dict[str, float],
    ) -> None:
        session.diarized_speaker_map.update(speaker_map)
        session.diarized_candidate_scores.update(score_map)

        interviewer_exists = any(role == "interviewer" for role in session.diarized_speaker_map.values())
        session.interviewer_confirmed = interviewer_exists

    def role_for_diarized_speaker(
        self,
        session: InterviewSession,
        diarized_speaker: str,
    ) -> SpeakerLabel:
        role = session.diarized_speaker_map.get(diarized_speaker, "unknown")
        if role in ("candidate", "interviewer", "unknown"):
            return role
        return "unknown"

    def make_segment_key(
        self,
        diarized_speaker: str,
        start: float,
        end: float,
    ) -> str:
        return f"{diarized_speaker}:{start:.2f}:{end:.2f}"

    def should_process_segment(
        self,
        session: InterviewSession,
        diarized_speaker: str,
        start: float,
        end: float,
    ) -> bool:
        key = self.make_segment_key(diarized_speaker, start, end)
        if key in session.processed_segment_keys:
            return False
        session.processed_segment_keys.add(key)
        return True

    def make_question_key(self, text: str) -> str:
        normalized = " ".join(text.lower().strip().split())
        return normalized

    def should_answer_question(self, session: InterviewSession, text: str) -> bool:
        key = self.make_question_key(text)
        if key in session.answered_question_keys:
            return False
        session.answered_question_keys.add(key)
        return True

    def auto_register_interviewer_from_embedding(
        self,
        session: InterviewSession,
        embedding: np.ndarray,
    ) -> str:
        if session.interviewer_profile_id:
            return session.interviewer_profile_id

        candidate_embedding = self.profile_store.load_embedding(session.candidate_profile_id)
        candidate_score = self.embedding_service.similarity(embedding, candidate_embedding)

        if candidate_score >= self.weak_accept:
            raise ValueError(
                f"Cannot register interviewer: too similar to candidate (score: {candidate_score:.3f})"
            )

        avg_embedding = (
            self.embedding_service.centroid(session.interviewer_embedding_buffer)
            if session.interviewer_embedding_buffer
            else embedding
        )

        profile_id = self.profile_store.create_profile("interviewer", avg_embedding)
        session.interviewer_profile_id = profile_id
        session.interviewer_confirmed = True
        return profile_id

    def refresh_interviewer_embedding(
        self,
        session: InterviewSession,
        embedding: np.ndarray,
        alpha: float = 0.7,
    ) -> None:
        if not session.interviewer_profile_id:
            session.interviewer_embedding_buffer.append(embedding)
            session.interviewer_embedding_buffer = session.interviewer_embedding_buffer[-8:]
            return

        candidate_embedding = self.profile_store.load_embedding(session.candidate_profile_id)
        candidate_score = self.embedding_service.similarity(embedding, candidate_embedding)

        if candidate_score >= self.weak_accept:
            return

        current = self.profile_store.load_embedding(session.interviewer_profile_id)
        mixed = alpha * current + (1.0 - alpha) * embedding
        norm = np.linalg.norm(mixed, axis=1, keepdims=True)
        updated = mixed / np.clip(norm, 1e-12, None)
        self.profile_store.update_embedding(session.interviewer_profile_id, updated)