# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reusable text-to-speech backend for pretrained MagpieTTS models."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


SPEAKER_MAP = {"Aria": 0, "Jason": 1, "John": 2, "Leo": 3, "Sofia": 4}
SUPPORTED_LANGUAGES = {"ar", "de", "en", "es", "fr", "hi", "it", "ja", "ko", "pt", "vi", "zh"}


@dataclass(frozen=True)
class MagpieTTSConfig:
    """Configuration needed to load and run a MagpieTTS checkpoint."""

    model: str = "nvidia/magpie_tts_multilingual_357m"
    revision: str | None = "v2607"
    model_filename: str = "magpie_tts_multilingual_357m.nemo"
    device: str = "cuda:0"
    speaker: str | int = "Sofia"
    use_cfg: bool = True
    cfg_scale: float = 2.5
    apply_text_normalization: bool = True
    seed: int = 42

    def __post_init__(self):
        if self.cfg_scale <= 0:
            raise ValueError("cfg_scale must be positive")
        parse_speaker(self.speaker)


@dataclass(frozen=True)
class SynthesisOutput:
    """One synthesized waveform and its measured generation metadata."""

    waveform: np.ndarray
    sample_rate: int
    generation_time_seconds: float

    @property
    def audio_duration_seconds(self) -> float:
        return self.waveform.size / self.sample_rate

    @property
    def rtf(self) -> float:
        duration = self.audio_duration_seconds
        return self.generation_time_seconds / duration if duration else float("inf")


def parse_speaker(value: str | int) -> tuple[str, int]:
    """Resolve a speaker name or numeric index to its canonical pair."""
    if isinstance(value, int) or str(value).isdigit():
        index = int(value)
        names = {index: name for name, index in SPEAKER_MAP.items()}
        if index not in names:
            raise ValueError(f"speaker index must be in [0, {len(SPEAKER_MAP) - 1}], got {index}")
        return names[index], index

    name = str(value).strip().title()
    if name not in SPEAKER_MAP:
        raise ValueError(f"unknown speaker {value!r}; choose one of {', '.join(SPEAKER_MAP)}")
    return name, SPEAKER_MAP[name]


def resolve_model_path(model: str, revision: str | None, filename: str) -> str:
    """Resolve a local checkpoint or download a pinned Hugging Face revision."""
    local_path = Path(model).expanduser()
    if local_path.exists():
        return str(local_path.resolve())

    from huggingface_hub import hf_hub_download

    if revision:
        return hf_hub_download(repo_id=model, filename=filename, revision=revision)
    return model


class MagpieTTSBackend:
    """Lazy-loadable MagpieTTS inference backend used by examples and cascades."""

    def __init__(self, config: MagpieTTSConfig, model=None):
        self.config = config
        self.device = torch.device(config.device)
        self.speaker, self.speaker_index = parse_speaker(config.speaker)
        self.checkpoint = (
            resolve_model_path(config.model, config.revision, config.model_filename) if model is None else config.model
        )

        load_start = time.perf_counter()
        self.model = model if model is not None else self._load_model()
        self.model = self.model.to(self.device).eval()
        self.model.inference_parameters.cfg_scale = config.cfg_scale
        self._synchronize()
        self.model_load_seconds = time.perf_counter() - load_start
        self.sample_rate = int(getattr(self.model, "output_sample_rate", self.model.sample_rate))
        self.seed(config.seed)

    def _load_model(self):
        from nemo.collections.tts.models import MagpieTTSModel

        if Path(self.checkpoint).exists():
            return MagpieTTSModel.restore_from(self.checkpoint, map_location=self.device)
        return MagpieTTSModel.from_pretrained(self.checkpoint, map_location=self.device)

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @staticmethod
    def seed(value: int) -> None:
        torch.manual_seed(value)
        np.random.seed(value)

    def synthesize(self, text: str, language: str, speaker: str | int | None = None) -> SynthesisOutput:
        """Synthesize one non-empty text span."""
        text = text.strip()
        language = language.strip().lower()
        if not text:
            raise ValueError("text must not be empty")
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"unsupported language {language!r}")
        speaker_index = self.speaker_index if speaker is None else parse_speaker(speaker)[1]

        self._synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            audio, audio_len = self.model.do_tts(
                text,
                language=language,
                apply_TN=self.config.apply_text_normalization,
                use_cfg=self.config.use_cfg,
                speaker_index=speaker_index,
            )
        self._synchronize()
        generation_time_seconds = time.perf_counter() - start
        num_samples = int(audio_len[0].item())
        waveform = audio[0, :num_samples].float().detach().cpu().numpy()
        return SynthesisOutput(waveform, self.sample_rate, generation_time_seconds)
