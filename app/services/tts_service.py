import asyncio
import audioop
import logging

import numpy as np
from openai import OpenAI

logger = logging.getLogger(__name__)


class TTSService:
    def __init__(
        self,
        api_key: str,
        voice_id: str,
        model: str = "gpt-4o-mini-tts",
        gain: float = 2.2,
    ):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.voice_id = voice_id or "alloy"
        self.gain = gain

    async def synthesize(self, text: str) -> bytes:
        if not text or not text.strip():
            return b""
        try:
            raw_pcm_24k_mono = await asyncio.to_thread(self._synthesize_pcm, text)
            pcm_48k_stereo = self._pcm24k_mono_to_pcm48k_stereo(raw_pcm_24k_mono)
            boosted = self._boost_volume(pcm_48k_stereo, self.gain)
            return boosted
        except Exception:
            logger.exception("TTS synthesis failed")
            return b""

    def _synthesize_pcm(self, text: str) -> bytes:
        response = self.client.audio.speech.create(
            model=self.model,
            voice=self.voice_id,
            input=text,
            response_format="pcm",
        )
        return response.read()

    def _pcm24k_mono_to_pcm48k_stereo(self, pcm_bytes: bytes) -> bytes:
        if not pcm_bytes:
            return b""
        pcm_48k_mono, _ = audioop.ratecv(pcm_bytes, 2, 1, 24000, 48000, None)
        stereo = audioop.tostereo(pcm_48k_mono, 2, 1.0, 1.0)
        return stereo

    def _boost_volume(self, pcm_bytes: bytes, gain: float) -> bytes:
        if not pcm_bytes:
            return b""
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        boosted = samples * gain
        peak = np.max(np.abs(boosted)) if boosted.size else 0
        ceiling = 30000.0
        if peak > ceiling and peak > 0:
            boosted *= ceiling / peak
        out = np.clip(boosted, -32768, 32767).astype(np.int16)
        return out.tobytes()