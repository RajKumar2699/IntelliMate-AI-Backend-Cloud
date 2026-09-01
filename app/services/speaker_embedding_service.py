from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from speechbrain.inference.speaker import EncoderClassifier


class SpeakerEmbeddingService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": self.device},
        )

    def embedding_from_bytes(self, audio_bytes: bytes, suffix: str = ".wav") -> np.ndarray:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as src:
            src.write(audio_bytes)
            src_path = Path(src.name)

        wav_path = src_path.with_suffix(".decoded.wav")

        try:
            self._decode_to_wav(src_path, wav_path)

            data, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=True)
            if data.size == 0:
                raise ValueError("Decoded audio is empty")

            signal = torch.from_numpy(data.T)

            if signal.shape[0] > 1:
                signal = torch.mean(signal, dim=0, keepdim=True)

            if sample_rate != 16000:
                import torchaudio
                signal = torchaudio.functional.resample(signal, sample_rate, 16000)

            with torch.no_grad():
                emb = self.model.encode_batch(signal.to(self.device))

            emb = emb.squeeze().detach().cpu().numpy().astype(np.float32)
            emb = np.asarray(emb).reshape(1, -1)
            return self.normalize(emb)

        finally:
            src_path.unlink(missing_ok=True)
            wav_path.unlink(missing_ok=True)

    def _decode_to_wav(self, src_path: Path, wav_path: Path) -> None:
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(src_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(wav_path),
        ]

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg decode failed: {result.stderr.strip()}")

        if not wav_path.exists() or wav_path.stat().st_size == 0:
            raise RuntimeError("ffmpeg decode failed: output wav file was not created")

    def normalize(self, emb: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb / np.clip(norm, 1e-12, None)

    def similarity(self, emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        a = self.normalize(emb_a)
        b = self.normalize(emb_b)
        return float(np.dot(a[0], b[0]))

    def centroid(self, embeddings: list[np.ndarray]) -> np.ndarray:
        stacked = np.vstack(embeddings)
        centroid = np.mean(stacked, axis=0, keepdims=True)
        return self.normalize(centroid)