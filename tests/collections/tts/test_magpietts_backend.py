# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from nemo.collections.tts.inference.cascade_synthesizer import (
    IncrementalSynthesisManager,
    resolve_language,
    select_playback_rate,
    synthesize_finalized_translations,
)
from nemo.collections.tts.inference.magpietts_backend import MagpieTTSBackend, MagpieTTSConfig, SynthesisOutput


class _Model:
    sample_rate = 4
    output_sample_rate = 4

    def __init__(self):
        self.inference_parameters = SimpleNamespace(cfg_scale=None)
        self.requests = []

    def to(self, device):
        return self

    def eval(self):
        return self

    def do_tts(self, text, language, apply_TN, use_cfg, speaker_index):
        self.requests.append((text, language, apply_TN, use_cfg, speaker_index))
        return torch.tensor([[0.25, -0.25, 0.5]]), torch.tensor([2])


class _StreamingModel(_Model):
    def __init__(self):
        super().__init__()
        self.chunk_requests = []

    def create_chunk_state(self, batch_size):
        return SimpleNamespace(batch_size=batch_size)

    def do_tts_chunk(self, text, **kwargs):
        self.chunk_requests.append((text, kwargs))
        return torch.tensor([[0.25, -0.25, 0.5]]), torch.tensor([2])


@pytest.mark.unit
def test_backend_runs_model_with_configured_inference_policy():
    model = _Model()
    backend = MagpieTTSBackend(MagpieTTSConfig(device="cpu", speaker="Leo", cfg_scale=2.0), model=model)

    output = backend.synthesize("Hallo", "de")

    assert model.inference_parameters.cfg_scale == 2.0
    assert model.requests == [("Hallo", "de", True, True, 3)]
    np.testing.assert_array_equal(output.waveform, [0.25, -0.25])
    assert output.audio_duration_seconds == 0.5


@pytest.mark.unit
def test_streaming_session_reuses_model_state_across_text_increments():
    model = _StreamingModel()
    backend = MagpieTTSBackend(MagpieTTSConfig(device="cpu", speaker="Leo"), model=model)
    session = backend.create_streaming_session("de")

    first = session.synthesize("Guten Morgen.")
    second = session.synthesize("Wie geht es Ihnen?", end_of_text=True)

    assert first.audio_duration_seconds == second.audio_duration_seconds == 0.5
    assert model.chunk_requests[0][1]["chunk_state"] is model.chunk_requests[1][1]["chunk_state"]
    assert model.chunk_requests[0][1]["beginning_of_text"] is True
    assert model.chunk_requests[1][1]["beginning_of_text"] is False
    assert model.chunk_requests[0][1]["end_of_text"] is False
    assert model.chunk_requests[1][1]["end_of_text"] is True


@pytest.mark.unit
def test_incremental_manager_emits_chunks_and_reports_time_to_first_audio(tmp_path):
    class _Session:
        def __init__(self):
            self.closed = False
            self.end_of_text = []

        def synthesize(self, text, end_of_text=False):
            self.end_of_text.append(end_of_text)
            value = 0.25 if text.startswith("First") else 0.5
            return SynthesisOutput(np.full(2, value, dtype=np.float32), 4, 0.25)

        def close(self):
            self.closed = True

    class _Backend:
        sample_rate = 4

        def __init__(self):
            self.sessions = []

        def create_streaming_session(self, language):
            session = _Session()
            self.sessions.append((language, session))
            return session

    backend = _Backend()
    manager = IncrementalSynthesisManager(backend, tmp_path, "de", min_chunk_units=2, inter_segment_silence_ms=250)
    assert manager.accept_committed(0, "First", 1.0) is None
    first = manager.accept_committed(0, "part.", 1.25)
    assert manager.accept_committed(0, "Second.", 2.0) is None
    output = manager.finalize({0: {"audio_filepath": "/data/talk.wav"}})
    second = output[0]["tts_segments"][1]

    assert len(backend.sessions) == 1
    assert backend.sessions[0][1].closed is True
    assert backend.sessions[0][1].end_of_text == [False, True]
    assert first["audio_ready_seconds"] == pytest.approx(1.5)
    assert first["playback_start_seconds"] == pytest.approx(1.5)
    assert first["playback_gap_seconds"] == 0.0
    assert second["playback_start_seconds"] == pytest.approx(2.25)
    assert second["playback_gap_seconds"] == pytest.approx(0.25)
    assert output[0]["tts_time_to_first_audio_seconds"] == pytest.approx(1.5)
    assert output[0]["tts_incremental"] is True
    assert output[0]["tts_min_chunk_units"] == 2
    assert output[0]["tts_rate_control"] == "none"
    assert first["playback_rate"] == 1.0
    assert first["raw_audio_duration_seconds"] == first["audio_duration_seconds"]
    assert all(segment["audio_filepath"] for segment in output[0]["tts_segments"])
    waveform, sample_rate = sf.read(output[0]["speech_filepath"])
    assert sample_rate == 4
    np.testing.assert_allclose(waveform, [0.25, 0.25, 0.0, 0.5, 0.5], atol=4e-5)


@pytest.mark.unit
def test_queue_aware_manager_starts_with_phrase_then_waits_for_sentence(tmp_path):
    class _Session:
        def __init__(self):
            self.texts = []

        def synthesize(self, text, end_of_text=False):
            self.texts.append((text, end_of_text))
            return SynthesisOutput(np.ones(8, dtype=np.float32), 4, 0.25)

        def close(self):
            pass

    class _Backend:
        sample_rate = 4

        def __init__(self):
            self.session = _Session()

        def create_streaming_session(self, language):
            return self.session

    backend = _Backend()
    manager = IncrementalSynthesisManager(
        backend,
        tmp_path,
        "de",
        min_chunk_units=2,
        max_chunk_units=4,
        chunking_policy="queue_aware",
        target_buffer_seconds=1.0,
    )

    assert manager.accept_committed(0, "one two", 1.0) is None
    first = manager.accept_committed(0, "three four five", 1.25)
    assert first["text"] == "one two three four"
    assert first["chunking_rule"] == "first_phrase_forced"

    # Two seconds of synthesized audio are buffered, so an incomplete tail waits.
    assert manager.accept_committed(0, "six seven", 1.5) is None
    second = manager.accept_committed(0, "eight.", 1.75, is_boundary=True)
    assert second["text"] == "five six seven eight."
    assert second["chunking_rule"] == "sentence"

    output = manager.finalize({0: {"audio_filepath": "/data/talk.wav"}})[0]
    assert backend.session.texts == [
        ("one two three four", False),
        ("five six seven eight.", False),
    ]
    assert output["tts_chunking_policy"] == "queue_aware"
    assert output["tts_max_chunk_units"] == 4
    assert output["tts_target_buffer_seconds"] == 1.0


@pytest.mark.unit
def test_queue_aware_manager_falls_back_when_playback_buffer_is_low(tmp_path):
    class _Session:
        def synthesize(self, text, end_of_text=False):
            return SynthesisOutput(np.ones(2, dtype=np.float32), 4, 0.1)

        def close(self):
            pass

    class _Backend:
        sample_rate = 4

        @staticmethod
        def create_streaming_session(language):
            return _Session()

    manager = IncrementalSynthesisManager(
        _Backend(),
        tmp_path,
        "de",
        min_chunk_units=2,
        max_chunk_units=4,
        chunking_policy="queue_aware",
        target_buffer_seconds=1.0,
    )
    first = manager.accept_committed(0, "one two three four", 1.0)
    second = manager.accept_committed(0, "five six seven eight", 2.0)

    assert first["chunking_rule"] == "first_phrase_forced"
    assert second["chunking_rule"] == "underrun_phrase_forced"


@pytest.mark.unit
def test_select_playback_rate_supports_fixed_deadline_and_source_paced_modes():
    common = {
        "raw_audio_duration_seconds": 4.0,
        "playback_start_seconds": 10.0,
        "source_commit_seconds": 8.0,
        "fixed_rate": 1.4,
        "maximum_rate": 1.5,
        "target_lag_seconds": 5.0,
    }

    assert select_playback_rate(mode="none", **common) == 1.0
    assert select_playback_rate(mode="fixed", **common) == 1.4
    assert select_playback_rate(mode="adaptive", **common) == pytest.approx(4 / 3)
    assert select_playback_rate(mode="adaptive", **{**common, "playback_start_seconds": 14.0}) == 1.5
    assert select_playback_rate(mode="source_paced", cumulative_raw_audio_seconds=18.0, **common) == 1.5


@pytest.mark.unit
def test_incremental_manager_applies_fixed_pitch_preserving_rate(monkeypatch, tmp_path):
    class _Session:
        def synthesize(self, text, end_of_text=False):
            return SynthesisOutput(np.arange(8, dtype=np.float32), 4, 0.25)

        def close(self):
            pass

    class _Backend:
        sample_rate = 4

        @staticmethod
        def create_streaming_session(language):
            return _Session()

    monkeypatch.setattr(
        "nemo.collections.tts.inference.cascade_synthesizer.time_stretch_waveform",
        lambda waveform, playback_rate: waveform[:: int(playback_rate)],
    )
    manager = IncrementalSynthesisManager(
        _Backend(),
        tmp_path,
        "de",
        min_chunk_units=1,
        rate_control="fixed",
        playback_rate=2.0,
    )
    segment = manager.accept_committed(0, "Hallo.", 1.0, is_last=True)
    output = manager.finalize({0: {"audio_filepath": "/data/talk.wav"}})

    assert segment["playback_rate"] == 2.0
    assert segment["raw_audio_duration_seconds"] == 2.0
    assert segment["audio_duration_seconds"] == 1.0
    assert output[0]["tts_raw_audio_duration_seconds"] == 2.0
    assert output[0]["tts_audio_duration_seconds"] == 1.0


@pytest.mark.unit
def test_incremental_manager_source_paced_rate_uses_cumulative_source_time(monkeypatch, tmp_path):
    class _Session:
        def synthesize(self, text, end_of_text=False):
            return SynthesisOutput(np.arange(8, dtype=np.float32), 4, 0.25)

        def close(self):
            pass

    class _Backend:
        sample_rate = 4

        @staticmethod
        def create_streaming_session(language):
            return _Session()

    monkeypatch.setattr(
        "nemo.collections.tts.inference.cascade_synthesizer.time_stretch_waveform",
        lambda waveform, playback_rate: waveform[: round(len(waveform) / playback_rate)],
    )
    manager = IncrementalSynthesisManager(
        _Backend(),
        tmp_path,
        "de",
        min_chunk_units=1,
        rate_control="source_paced",
        maximum_playback_rate=1.5,
    )
    first = manager.accept_committed(0, "Hallo", 1.0)
    second = manager.accept_committed(0, "Welt", 4.0, is_last=True)
    output = manager.finalize({0: {"audio_filepath": "/data/talk.wav"}})[0]

    assert first["playback_rate"] == 1.5
    assert second["playback_rate"] == 1.0
    assert output["tts_raw_to_source_duration_ratio"] == 1.0
    assert output["tts_played_to_source_duration_ratio"] == pytest.approx(3.25 / 4)


@pytest.mark.unit
@pytest.mark.parametrize(("value", "expected"), [("German", "de"), ("zh", "zh"), (None, "it")])
def test_resolve_language(value, expected):
    target = "Italian" if value is None else None
    assert resolve_language(value, target) == expected


@pytest.mark.unit
def test_synthesize_finalized_translations_concatenates_only_final_units(tmp_path):
    class _Backend:
        sample_rate = 4

        def synthesize(self, text, language):
            value = 0.25 if text == "First." else 0.5
            return SynthesisOutput(np.full(2, value, dtype=np.float32), 4, 0.1)

    output = {
        0: {
            "audio_filepath": "/data/talk.wav",
            "translation": "First. Second.",
            "translation_segments": [("uncommitted hypothesis", 0.5)],
            "final_translation_segments": [("First.", 1.0), ("Second.", 2.0)],
        }
    }

    synthesize_finalized_translations(output, _Backend(), tmp_path, "de", inter_segment_silence_ms=250)

    stream = output[0]
    waveform, sample_rate = sf.read(stream["speech_filepath"])
    assert sample_rate == 4
    np.testing.assert_allclose(waveform, [0.25, 0.25, 0.0, 0.5, 0.5], atol=4e-5)
    assert [segment["text"] for segment in stream["tts_segments"]] == [
        "First.",
        "Second.",
    ]
    assert stream["tts_audio_duration_seconds"] == 1.25
    assert stream["tts_generation_time_seconds"] == pytest.approx(0.2)


@pytest.mark.unit
def test_long_chinese_tts_sentence_is_split_before_model_input(tmp_path):
    model = _StreamingModel()
    backend = MagpieTTSBackend(MagpieTTSConfig(device="cpu", speaker="Leo"), model=model)
    manager = IncrementalSynthesisManager(backend, tmp_path, "zh", min_chunk_units=6)
    text = "你" * 60 + "，" + "好" * 121 + "。"

    manager.emit(0, text, 1.0, end_of_text=True)

    requests = [request[0] for request in model.chunk_requests]
    assert len(requests) >= 3
    assert max(map(len, requests)) <= 80
    assert "".join(requests) == text
    assert [segment["chunking_rule"] for segment in manager.segments[0]][:-1] == [
        "safety_split",
    ] * (len(requests) - 1)
    assert model.chunk_requests[-1][1]["end_of_text"] is True
