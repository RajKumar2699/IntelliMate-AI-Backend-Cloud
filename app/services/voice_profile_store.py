from __future__ import annotations

import json
import uuid
from pathlib import Path

import numpy as np


class VoiceProfileStore:
    def __init__(self, base_dir: str = "data/voice_profiles"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_profile(self, role: str, embedding: np.ndarray) -> str:
        profile_id = str(uuid.uuid4())
        profile_dir = self.base_dir / profile_id
        profile_dir.mkdir(parents=True, exist_ok=True)

        np.save(profile_dir / "embedding.npy", embedding)
        (profile_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "profile_id": profile_id,
                    "role": role,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return profile_id

    def load_embedding(self, profile_id: str) -> np.ndarray:
        return np.load(self.base_dir / profile_id / "embedding.npy")

    def update_embedding(self, profile_id: str, embedding: np.ndarray) -> None:
        np.save(self.base_dir / profile_id / "embedding.npy", embedding)

    def exists(self, profile_id: str) -> bool:
        return (self.base_dir / profile_id / "embedding.npy").exists()