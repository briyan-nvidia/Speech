# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline speech synthesis for finalized translations from an ASR-to-MT cascade."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import soundfile as sf

from nemo.collections.tts.inference.magpietts_backend import SUPPORTED_LANGUAGES, MagpieTTSBackend

LANGUAGE_NAME_TO_CODE = {
    "arabic": "ar",
    "chinese": "zh",
    "english": "en",
    "french": "fr",
    "german": "de",
    "hindi": "hi",
    "italian": "it",
    "japanese": "ja",
    "korean": "ko",
    "portuguese": "pt",
    "spanish": "es",
    "vietnamese": "vi",
}


def resolve_language(language: str | None, target_language: str | None) -> str:
    """Resolve an explicit code or NMT language name to a Magpie language code."""
    value = str(language or target_language or "").strip().lower()
    code = LANGUAGE_NAME_TO_CODE.get(value, value)
    if code not in SUPPORTED_LANGUAGES:
        raise ValueError("TTS language must be a supported code or NMT target language name")
    return code


def _audio_filename(audio_filepath: str, stream_id: int) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(audio_filepath).stem).strip("._") or "audio"
    return f"{stream_id:05d}_{stem}.wav"


def synthesize_finalized_translations(
    pipeline_output: dict,
    backend: MagpieTTSBackend,
    output_dir: str | Path,
    language: str,
    inter_segment_silence_ms: int = 0,
) -> dict:
    """Synthesize and concatenate each stream's finalized MT units.

    The function mutates and returns ``pipeline_output`` by adding TTS artifacts and
    timing metadata. Existing ASR and translation fields are not modified.
    """
    if inter_segment_silence_ms < 0:
        raise ValueError("inter_segment_silence_ms must be non-negative")

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    silence = np.zeros(round(backend.sample_rate * inter_segment_silence_ms / 1000), dtype=np.float32)

    for stream_id, stream_output in pipeline_output.items():
        finalized_units = stream_output.get("final_translation_segments", [])
        waveforms = []
        tts_segments = []
        audio_offset = 0
        total_generation_seconds = 0.0
        for text, source_finalization_seconds in finalized_units:
            text = text.strip()
            if not text:
                continue
            synthesis = backend.synthesize(text, language)
            if synthesis.sample_rate != backend.sample_rate:
                raise ValueError("TTS backend changed sample rate within a stream")
            if waveforms and silence.size:
                waveforms.append(silence)
                audio_offset += silence.size

            start_seconds = audio_offset / backend.sample_rate
            waveforms.append(synthesis.waveform)
            audio_offset += synthesis.waveform.size
            end_seconds = audio_offset / backend.sample_rate
            total_generation_seconds += synthesis.generation_time_seconds
            tts_segments.append(
                {
                    "text": text,
                    "source_finalization_seconds": source_finalization_seconds,
                    "audio_start_seconds": start_seconds,
                    "audio_end_seconds": end_seconds,
                    "generation_time_seconds": synthesis.generation_time_seconds,
                    "rtf": synthesis.rtf,
                }
            )

        speech_filepath = None
        if waveforms:
            waveform = np.concatenate(waveforms)
            speech_path = output_dir / _audio_filename(stream_output["audio_filepath"], stream_id)
            sf.write(speech_path, waveform, backend.sample_rate, subtype="PCM_16")
            speech_filepath = str(speech_path)

        stream_output.update(
            {
                "speech_filepath": speech_filepath,
                "tts_segments": tts_segments,
                "tts_sample_rate": backend.sample_rate,
                "tts_audio_duration_seconds": audio_offset / backend.sample_rate,
                "tts_generation_time_seconds": total_generation_seconds,
            }
        )
    return pipeline_output
