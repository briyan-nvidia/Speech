# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from nemo.collections.tts.inference.cascade_synthesizer import resolve_language, synthesize_finalized_translations
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
    assert [segment["text"] for segment in stream["tts_segments"]] == ["First.", "Second."]
    assert stream["tts_audio_duration_seconds"] == 1.25
    assert stream["tts_generation_time_seconds"] == pytest.approx(0.2)
