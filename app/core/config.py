# app/core/config.py
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = Field(default="IntelliMate AI Backend", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    debug: bool = Field(default=True, alias="DEBUG")
    gnani_api_key: str | None = None
    gnani_stt_base_url: str = "https://api.vachana.ai/api/v1"
    gnani_language_code: str = "en-IN"


    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    openai_chat_model: str = Field(default="gpt-4o-mini", alias="OPENAI_CHAT_MODEL")
    openai_tts_model: str = Field(default="gpt-4o-mini-tts", alias="OPENAI_TTS_MODEL")
    openai_tts_voice: str = Field(default="alloy", alias="OPENAI_TTS_VOICE")
    openai_realtime_model: str = Field(default="gpt-realtime", alias="OPENAI_REALTIME_MODEL")
    openai_stt_model: str = Field(default="gpt-4o-mini-transcribe", alias="OPENAI_STT_MODEL")

    sarvam_api_key: str | None = Field(default=None, alias="SARVAM_API_KEY")
    sarvam_stt_model: str = Field(default="saaras:v3", alias="SARVAM_STT_MODEL")
    sarvam_stt_language_code: str = Field(default="en-IN", alias="SARVAM_STT_LANGUAGE_CODE")
    sarvam_stt_mode: str = Field(default="formal", alias="SARVAM_STT_MODE")

    interview_strong_accept: float = Field(default=0.74, alias="INTERVIEW_STRONG_ACCEPT")
    interview_weak_accept: float = Field(default=0.67, alias="INTERVIEW_WEAK_ACCEPT")
    interview_reject_below: float = Field(default=0.58, alias="INTERVIEW_REJECT_BELOW")
    interview_confirm_interviewer_runs: int = Field(default=2, alias="INTERVIEW_CONFIRM_INTERVIEWER_RUNS")
    interview_min_enroll_files: int = Field(default=3, alias="INTERVIEW_MIN_ENROLL_FILES")
    interview_min_utterance_seconds: float = Field(default=1.2, alias="INTERVIEW_MIN_UTTERANCE_SECONDS")
    interview_low_rms_threshold: float = Field(default=120.0, alias="INTERVIEW_LOW_RMS_THRESHOLD")


    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()