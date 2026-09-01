import base64
import os
import tempfile
import logging
from pathlib import Path

import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class SarvamSTTService:
    def __init__(self):
        settings = get_settings()
        self.api_key = settings.sarvam_api_key
        self.model = settings.sarvam_stt_model
        self.default_language_code = settings.sarvam_stt_language_code
        self.default_mode = settings.sarvam_stt_mode
        self.base_url = "https://api.sarvam.ai"

    async def transcribe_file(
        self,
        file_path: str,
        language_code: str | None = None,
        mode: str | None = None,
    ) -> tuple[str, str | None]:
        if not self.api_key:
            raise RuntimeError("SARVAM_API_KEY is not configured")

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(file_path)

        lang = language_code or self.default_language_code or "unknown"
        selected_mode = mode or self.default_mode or "codemix"

        headers = {"api-subscription-key": self.api_key}
        data = {
            "model": self.model,
            "language_code": lang,
            "mode": selected_mode,
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            with open(path, "rb") as audio_file:
                files = {"file": (path.name, audio_file, "audio/wav")}
                response = await client.post(
                    f"{self.base_url}/speech-to-text",
                    headers=headers,
                    data=data,
                    files=files,
                )
            response.raise_for_status()
            payload = response.json()

        transcript = (
            payload.get("transcript")
            or payload.get("text")
            or payload.get("data", {}).get("text")
            or ""
        ).strip()

        detected_language = (
            payload.get("language_code")
            or payload.get("detected_language")
            or payload.get("data", {}).get("language_code")
        )

        return transcript, detected_language

    async def transcribe_base64_wav(
        self,
        audio_base64: str,
        language_code: str | None = None,
        mode: str | None = None,
    ) -> tuple[str, str | None]:
        raw = base64.b64decode(audio_base64)
        fd, temp_path = tempfile.mkstemp(suffix=".wav")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(raw)
            return await self.transcribe_file(
                file_path=temp_path,
                language_code=language_code,
                mode=mode,
            )
        finally:
            try:
                os.remove(temp_path)
            except FileNotFoundError:
                pass