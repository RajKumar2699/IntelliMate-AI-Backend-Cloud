import numpy as np


class UtteranceDetector:
    def __init__(
        self,
        silence_rms_threshold: float = 150.0,
        silence_hangover_chunks: int = 20,
        min_speech_chunks: int = 6,
        max_utterance_chunks: int = 150,
    ):
        self.silence_rms_threshold = silence_rms_threshold
        self.silence_hangover_chunks = silence_hangover_chunks
        self.min_speech_chunks = min_speech_chunks
        self.max_utterance_chunks = max_utterance_chunks

        self._buffer = bytearray()
        self._speaking = False
        self._silence_count = 0
        self._speech_count = 0

    @staticmethod
    def _rms(pcm_bytes: bytes) -> float:
        if not pcm_bytes:
            return 0.0
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        return float(np.sqrt(np.mean(samples ** 2)))

    def process_chunk(self, pcm16_chunk: bytes) -> bytes | None:
        is_speech = self._rms(pcm16_chunk) > self.silence_rms_threshold

        if is_speech:
            self._speaking = True
            self._silence_count = 0
            self._speech_count += 1
            self._buffer += pcm16_chunk

            if self._speech_count >= self.max_utterance_chunks:
                utterance = bytes(self._buffer)
                self._reset()
                return utterance
            return None

        if self._speaking:
            self._buffer += pcm16_chunk
            self._silence_count += 1

            if self._silence_count >= self.silence_hangover_chunks:
                utterance = bytes(self._buffer) if self._speech_count >= self.min_speech_chunks else None
                self._reset()
                return utterance

        return None

    def _reset(self) -> None:
        self._buffer = bytearray()
        self._speaking = False
        self._silence_count = 0
        self._speech_count = 0