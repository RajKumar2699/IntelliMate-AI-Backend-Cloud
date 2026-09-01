import logging
import numpy as np
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class STTService:
    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe_pcm16(
        self,
        pcm_bytes: bytes,
        sample_rate: int = 16000,
        language_hint: str | None = None,
    ) -> tuple[str, str | None]:
        if not pcm_bytes:
            return "", None

        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        try:
            kwargs = {"vad_filter": True}
            # Only bias toward a language if we actually have a hint for it —
            # don't hardcode Hindi for every user.
            if language_hint:
                kwargs["language"] = language_hint

            segments, info = self.model.transcribe(audio, **kwargs)
            text = " ".join(
                segment.text.strip() for segment in segments if segment.text
            ).strip()
            language = getattr(info, "language", None)
            return text, language
        except Exception:
            logger.exception("STT transcription failed")
            return "", None