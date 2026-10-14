# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline and incremental speech synthesis for an ASR-to-MT cascade."""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from nemo.collections.tts.inference.magpietts_backend import (
    SUPPORTED_LANGUAGES,
    MagpieTTSBackend,
    MagpieTTSStreamingSession,
)

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
RATE_CONTROL_MODES = {"none", "fixed", "adaptive", "source_paced"}
CHUNKING_POLICIES = {"minimum", "queue_aware"}
PHRASE_PUNCTUATION = set(",;:，；：、")
TERMINAL_PUNCTUATION = set(".!?。！？")
MAX_CJK_TTS_CHUNK_UNITS = 80  # Bound text-encoder inputs below Magpie's fixed attention mask.


def select_playback_rate(
    mode: str,
    raw_audio_duration_seconds: float,
    playback_start_seconds: float,
    source_commit_seconds: float,
    fixed_rate: float,
    maximum_rate: float,
    target_lag_seconds: float,
    cumulative_raw_audio_seconds: float | None = None,
) -> float:
    """Select a pitch-preserving playback rate for one synthesized chunk."""
    if mode not in RATE_CONTROL_MODES:
        raise ValueError(f"rate_control must be one of {sorted(RATE_CONTROL_MODES)}, got {mode!r}")
    if fixed_rate <= 0 or maximum_rate <= 0:
        raise ValueError("playback rates must be positive")
    if mode in {"adaptive", "source_paced"} and maximum_rate < 1.0:
        raise ValueError("maximum_rate must be at least 1.0 for adaptive or source-paced rate control")
    if target_lag_seconds < 0:
        raise ValueError("target_lag_seconds must be non-negative")
    if mode == "none":
        return 1.0
    if mode == "fixed":
        return fixed_rate
    if mode == "source_paced":
        if cumulative_raw_audio_seconds is None:
            raise ValueError("cumulative_raw_audio_seconds is required for source_paced rate control")
        if source_commit_seconds <= 0:
            return maximum_rate
        estimated_rate = cumulative_raw_audio_seconds / source_commit_seconds
        return min(maximum_rate, max(1.0, estimated_rate))

    playback_deadline = source_commit_seconds + target_lag_seconds
    available_seconds = playback_deadline - playback_start_seconds
    required_rate = raw_audio_duration_seconds / available_seconds if available_seconds > 0 else maximum_rate
    return min(maximum_rate, max(1.0, required_rate))


def time_stretch_waveform(waveform: np.ndarray, playback_rate: float) -> np.ndarray:
    """Change duration without changing pitch."""
    if playback_rate == 1.0:
        return waveform
    import librosa

    return librosa.effects.time_stretch(waveform, rate=playback_rate)


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


class IncrementalSynthesisManager:
    """Synthesize committed MT deltas immediately and retain one Magpie state per stream."""

    def __init__(
        self,
        backend: MagpieTTSBackend,
        output_dir: str | Path,
        language: str,
        min_chunk_units: int = 4,
        max_chunk_units: int = 12,
        chunking_policy: str = "minimum",
        target_buffer_seconds: float = 2.0,
        inter_segment_silence_ms: int = 0,
        rate_control: str = "none",
        playback_rate: float = 1.0,
        maximum_playback_rate: float = 1.4,
        target_lag_seconds: float = 5.5,
    ):
        if inter_segment_silence_ms < 0:
            raise ValueError("inter_segment_silence_ms must be non-negative")
        if min_chunk_units < 1:
            raise ValueError("min_chunk_units must be positive")
        if max_chunk_units < min_chunk_units:
            raise ValueError("max_chunk_units must be at least min_chunk_units")
        if chunking_policy not in CHUNKING_POLICIES:
            raise ValueError(f"chunking_policy must be one of {sorted(CHUNKING_POLICIES)}")
        if target_buffer_seconds < 0:
            raise ValueError("target_buffer_seconds must be non-negative")
        select_playback_rate(
            mode=rate_control,
            raw_audio_duration_seconds=1.0,
            playback_start_seconds=0.0,
            source_commit_seconds=0.0,
            fixed_rate=playback_rate,
            maximum_rate=maximum_playback_rate,
            target_lag_seconds=target_lag_seconds,
            cumulative_raw_audio_seconds=1.0,
        )
        self.backend = backend
        self.output_dir = Path(output_dir).resolve()
        self.chunk_dir = self.output_dir / "chunks"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        self.language = resolve_language(language, None)
        self.min_chunk_units = min_chunk_units
        self.max_chunk_units = max_chunk_units
        self.chunking_policy = chunking_policy
        self.target_buffer_seconds = target_buffer_seconds
        self.rate_control = rate_control
        self.playback_rate = playback_rate
        self.maximum_playback_rate = maximum_playback_rate
        self.target_lag_seconds = target_lag_seconds
        self.silence = np.zeros(
            round(backend.sample_rate * inter_segment_silence_ms / 1000),
            dtype=np.float32,
        )
        self.sessions: dict[int, MagpieTTSStreamingSession] = {}
        self.pending_text: dict[int, str] = {}
        self.pending_source_seconds: dict[int, float] = {}
        self.waveforms: dict[int, list[np.ndarray]] = {}
        self.segments: dict[int, list[dict]] = {}
        self.playback_end_seconds: dict[int, float] = {}
        self.cumulative_raw_audio_seconds: dict[int, float] = {}
        self.compute_available_seconds = 0.0
        self.wall_start = time.perf_counter()

    def _append_text(self, existing: str, delta: str) -> str:
        if not existing:
            return delta.strip()
        if self.language in {"ja", "zh"}:
            return f"{existing}{delta.strip()}"
        existing = existing.rstrip()
        delta = delta.lstrip()
        no_space_before = set(",.!?;:%)]}»”’。！？；：、")
        no_space_after = set("([{«“‘")
        separator = "" if delta[0] in no_space_before or existing[-1] in no_space_after else " "
        return f"{existing}{separator}{delta}"

    def _count_units(self, text: str) -> int:
        if self.language in {"ja", "zh"}:
            return len("".join(text.split()))
        return len(text.split())

    def accept_committed(
        self,
        stream_id: int,
        text: str,
        source_commit_seconds: float,
        is_boundary: bool = False,
        is_last: bool = False,
    ) -> dict | None:
        """Buffer a monotonic MT delta and synthesize according to the configured chunk policy."""
        text = text.strip()
        if not text:
            return None
        pending = self._append_text(self.pending_text.get(stream_id, ""), text)
        self.pending_text[stream_id] = pending
        self.pending_source_seconds[stream_id] = float(source_commit_seconds)
        if self.chunking_policy == "minimum":
            if self._count_units(pending) < self.min_chunk_units and not is_boundary and not is_last:
                return None
            self.pending_text.pop(stream_id, None)
            self.pending_source_seconds.pop(stream_id, None)
            return self.emit(
                stream_id,
                pending,
                source_commit_seconds,
                end_of_text=is_last,
                chunking_rule="boundary" if is_boundary else "minimum",
            )

        buffered_audio_seconds = max(0.0, self.playback_end_seconds.get(stream_id, 0.0) - float(source_commit_seconds))
        selection = _select_queue_aware_prefix(
            pending,
            language=self.language,
            minimum_units=self.min_chunk_units,
            maximum_units=self.max_chunk_units,
            first_chunk=not self.segments.get(stream_id),
            buffered_audio_seconds=buffered_audio_seconds,
            target_buffer_seconds=self.target_buffer_seconds,
            is_boundary=is_boundary,
            is_last=is_last,
        )
        if selection is None:
            return None
        emitted_text, consumed_characters, rule = selection
        remainder = pending[consumed_characters:].lstrip()
        if remainder:
            self.pending_text[stream_id] = remainder
            self.pending_source_seconds[stream_id] = float(source_commit_seconds)
        else:
            self.pending_text.pop(stream_id, None)
            self.pending_source_seconds.pop(stream_id, None)
        return self.emit(
            stream_id,
            emitted_text,
            source_commit_seconds,
            end_of_text=is_last and not remainder,
            chunking_rule=rule,
            buffered_audio_seconds=buffered_audio_seconds,
        )

    def emit(
        self,
        stream_id: int,
        text: str,
        source_commit_seconds: float,
        end_of_text: bool = False,
        chunking_rule: str = "explicit",
        buffered_audio_seconds: float | None = None,
    ) -> dict:
        """Generate and persist one audio chunk as soon as its MT text is committed."""
        text = text.strip()
        if not text:
            raise ValueError("text must not be empty")
        # Long CJK sentence/boundary chunks can exceed Magpie's 2048-position
        # text-attention mask after phoneme tokenization. Bound only such chunks.
        if self.language in {"zh", "ja"} and self._count_units(text) > MAX_CJK_TTS_CHUNK_UNITS:
            spans = _unit_spans(text, self.language)
            cut = spans[MAX_CJK_TTS_CHUNK_UNITS - 1][1]
            for start, end, character in spans[:MAX_CJK_TTS_CHUNK_UNITS]:
                if (
                    start >= spans[MAX_CJK_TTS_CHUNK_UNITS // 2][0]
                    and character in PHRASE_PUNCTUATION | TERMINAL_PUNCTUATION
                ):
                    cut = end
            self.emit(
                stream_id,
                text[:cut],
                source_commit_seconds,
                end_of_text=False,
                chunking_rule="safety_split",
                buffered_audio_seconds=buffered_audio_seconds,
            )
            return self.emit(
                stream_id,
                text[cut:],
                source_commit_seconds,
                end_of_text=end_of_text,
                chunking_rule=chunking_rule,
                buffered_audio_seconds=buffered_audio_seconds,
            )
        if stream_id not in self.sessions:
            self.sessions[stream_id] = self.backend.create_streaming_session(self.language)
        session = self.sessions[stream_id]
        stream_segments = self.segments.setdefault(stream_id, [])
        chunk_index = len(stream_segments)

        wall_generation_start = time.perf_counter() - self.wall_start
        synthesis = session.synthesize(text, end_of_text=end_of_text)
        wall_generation_end = time.perf_counter() - self.wall_start

        generation_start = max(float(source_commit_seconds), self.compute_available_seconds)
        model_audio_ready = generation_start + synthesis.generation_time_seconds
        has_previous_audio = stream_id in self.playback_end_seconds
        previous_playback_end = self.playback_end_seconds.get(stream_id, 0.0)
        silence_seconds = self.silence.size / self.backend.sample_rate if chunk_index else 0.0
        estimated_playback_start = max(model_audio_ready, previous_playback_end + silence_seconds)
        cumulative_raw_audio_seconds = (
            self.cumulative_raw_audio_seconds.get(stream_id, 0.0) + synthesis.audio_duration_seconds
        )
        playback_rate = select_playback_rate(
            mode=self.rate_control,
            raw_audio_duration_seconds=synthesis.audio_duration_seconds,
            playback_start_seconds=estimated_playback_start,
            source_commit_seconds=float(source_commit_seconds),
            fixed_rate=self.playback_rate,
            maximum_rate=self.maximum_playback_rate,
            target_lag_seconds=self.target_lag_seconds,
            cumulative_raw_audio_seconds=cumulative_raw_audio_seconds,
        )
        self.cumulative_raw_audio_seconds[stream_id] = cumulative_raw_audio_seconds
        if playback_rate == 1.0:
            waveform = synthesis.waveform
            rate_control_seconds = 0.0
        else:
            rate_control_start = time.perf_counter()
            waveform = time_stretch_waveform(synthesis.waveform, playback_rate)
            rate_control_seconds = time.perf_counter() - rate_control_start
        wall_audio_ready = time.perf_counter() - self.wall_start
        chunk_path = self.chunk_dir / f"{stream_id:05d}_{chunk_index:04d}.wav"
        sf.write(chunk_path, waveform, synthesis.sample_rate, subtype="PCM_16")

        audio_ready = model_audio_ready + rate_control_seconds
        self.compute_available_seconds = audio_ready
        playback_start = max(audio_ready, previous_playback_end + silence_seconds)
        audio_duration_seconds = waveform.size / synthesis.sample_rate
        playback_end = playback_start + audio_duration_seconds
        self.playback_end_seconds[stream_id] = playback_end
        segment = {
            "text": text,
            "chunking_rule": chunking_rule,
            "buffered_audio_seconds_before": buffered_audio_seconds,
            "source_commit_seconds": float(source_commit_seconds),
            "end_of_text": end_of_text,
            "generation_start_seconds": generation_start,
            "audio_ready_seconds": audio_ready,
            "playback_start_seconds": playback_start,
            "playback_end_seconds": playback_end,
            "playback_gap_seconds": (max(0.0, playback_start - previous_playback_end) if has_previous_audio else 0.0),
            "wall_generation_start_seconds": wall_generation_start,
            "wall_generation_end_seconds": wall_generation_end,
            "wall_audio_ready_seconds": wall_audio_ready,
            "generation_time_seconds": synthesis.generation_time_seconds,
            "rate_control_time_seconds": rate_control_seconds,
            "raw_audio_duration_seconds": synthesis.audio_duration_seconds,
            "cumulative_raw_audio_seconds": cumulative_raw_audio_seconds,
            "audio_duration_seconds": audio_duration_seconds,
            "playback_rate": playback_rate,
            "rtf": synthesis.rtf,
            "effective_rtf": (
                (synthesis.generation_time_seconds + rate_control_seconds) / audio_duration_seconds
                if audio_duration_seconds
                else float("inf")
            ),
            "audio_filepath": str(chunk_path),
        }
        stream_segments.append(segment)
        self.waveforms.setdefault(stream_id, []).append(waveform)
        return segment

    def finalize(self, pipeline_output: dict) -> dict:
        """Close all sessions, assemble their emitted chunks, and attach timing metadata."""
        for stream_id, stream_output in pipeline_output.items():
            pending = self.pending_text.pop(stream_id, "")
            source_seconds = self.pending_source_seconds.pop(stream_id, 0.0)
            if pending:
                self.emit(stream_id, pending, source_seconds, end_of_text=True, chunking_rule="final")
            if stream_id in self.sessions:
                self.sessions[stream_id].close()
            chunks = self.waveforms.get(stream_id, [])
            segments = self.segments.get(stream_id, [])
            assembled = []
            for chunk in chunks:
                if assembled and self.silence.size:
                    assembled.append(self.silence)
                assembled.append(chunk)

            speech_filepath = None
            if assembled:
                waveform = np.concatenate(assembled)
                speech_path = self.output_dir / _audio_filename(stream_output["audio_filepath"], stream_id)
                sf.write(speech_path, waveform, self.backend.sample_rate, subtype="PCM_16")
                speech_filepath = str(speech_path)

            stream_output.update(
                {
                    "speech_filepath": speech_filepath,
                    "tts_segments": segments,
                    "tts_sample_rate": self.backend.sample_rate,
                    "tts_audio_duration_seconds": sum(segment["audio_duration_seconds"] for segment in segments)
                    + max(0, len(segments) - 1) * self.silence.size / self.backend.sample_rate,
                    "tts_generation_time_seconds": sum(segment["generation_time_seconds"] for segment in segments),
                    "tts_rate_control_time_seconds": sum(segment["rate_control_time_seconds"] for segment in segments),
                    "tts_raw_audio_duration_seconds": sum(
                        segment["raw_audio_duration_seconds"] for segment in segments
                    ),
                    "tts_time_to_first_audio_seconds": (segments[0]["playback_start_seconds"] if segments else None),
                    "tts_wall_time_to_first_audio_seconds": (
                        segments[0]["wall_audio_ready_seconds"] if segments else None
                    ),
                    "tts_timing_basis": (
                        "source audio elapsed plus serialized TTS generation and rate-control compute; "
                        "ASR/MT compute excluded"
                    ),
                    "tts_incremental": True,
                    "tts_min_chunk_units": self.min_chunk_units,
                    "tts_max_chunk_units": self.max_chunk_units,
                    "tts_chunking_policy": self.chunking_policy,
                    "tts_target_buffer_seconds": self.target_buffer_seconds,
                    "tts_rate_control": self.rate_control,
                    "tts_playback_rate": self.playback_rate,
                    "tts_maximum_playback_rate": self.maximum_playback_rate,
                    "tts_target_lag_seconds": self.target_lag_seconds,
                    "tts_estimated_source_seconds": (
                        max(segment["source_commit_seconds"] for segment in segments) if segments else None
                    ),
                    "tts_raw_to_source_duration_ratio": (
                        sum(segment["raw_audio_duration_seconds"] for segment in segments)
                        / max(segment["source_commit_seconds"] for segment in segments)
                        if segments and max(segment["source_commit_seconds"] for segment in segments) > 0
                        else None
                    ),
                    "tts_played_to_source_duration_ratio": (
                        (
                            sum(segment["audio_duration_seconds"] for segment in segments)
                            + max(0, len(segments) - 1) * self.silence.size / self.backend.sample_rate
                        )
                        / max(segment["source_commit_seconds"] for segment in segments)
                        if segments and max(segment["source_commit_seconds"] for segment in segments) > 0
                        else None
                    ),
                }
            )
        return pipeline_output


def _unit_spans(text: str, language: str) -> list[tuple[int, int, str]]:
    if language in {"ja", "zh"}:
        return [(index, index + 1, character) for index, character in enumerate(text) if not character.isspace()]
    return [(match.start(), match.end(), match.group()) for match in re.finditer(r"\S+", text)]


def _maximal_sentence_end(text: str) -> int | None:
    end = None
    for index, character in enumerate(text):
        if character in TERMINAL_PUNCTUATION:
            end = index + 1
            while end < len(text) and text[end] in "\"'»”’)]":
                end += 1
    return end


def _phrase_prefix(text: str, language: str, minimum_units: int, maximum_units: int) -> tuple[str, int, bool] | None:
    spans = _unit_spans(text, language)
    if len(spans) < minimum_units:
        return None
    upper = min(len(spans), maximum_units)
    for index in range(minimum_units - 1, upper):
        if spans[index][2][-1] in PHRASE_PUNCTUATION:
            end = spans[index][1]
            return text[:end].strip(), end, False
    if len(spans) >= maximum_units:
        end = spans[maximum_units - 1][1]
        return text[:end].strip(), end, True
    return None


def _select_queue_aware_prefix(
    text: str,
    language: str,
    minimum_units: int,
    maximum_units: int,
    first_chunk: bool,
    buffered_audio_seconds: float,
    target_buffer_seconds: float,
    is_boundary: bool,
    is_last: bool,
) -> tuple[str, int, str] | None:
    """Select a sentence when possible and a bounded phrase when playback risks underrun."""
    if is_last:
        return text.strip(), len(text), "final"
    if first_chunk:
        phrase = _phrase_prefix(text, language, minimum_units, maximum_units)
        if phrase is not None:
            phrase_text, end, forced = phrase
            rule = "first_phrase_forced" if forced else "first_phrase"
            return phrase_text, end, rule
    sentence_end = _maximal_sentence_end(text)
    if sentence_end is not None:
        return text[:sentence_end].strip(), sentence_end, "sentence"
    if is_boundary:
        return text.strip(), len(text), "boundary"
    if not first_chunk and buffered_audio_seconds > target_buffer_seconds:
        return None
    phrase = _phrase_prefix(text, language, minimum_units, maximum_units)
    if phrase is None:
        return None
    phrase_text, end, forced = phrase
    reason = "underrun_phrase"
    if forced:
        reason += "_forced"
    return phrase_text, end, reason
