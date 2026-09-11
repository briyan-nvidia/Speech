# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from examples.tts.magpietts_synthesize import load_requests, parse_speaker, safe_filename, summarize_results


@pytest.mark.unit
def test_load_requests_supports_multilingual_text_and_per_request_speaker(tmp_path):
    manifest = tmp_path / "requests.jsonl"
    records = [
        {"id": "de/1", "text": "Guten Morgen.", "language": "de"},
        {"id": "zh-1", "text": "早上好。", "language": "zh", "speaker": "Leo"},
    ]
    manifest.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in records), encoding="utf-8")

    requests = load_requests(manifest, default_speaker="Sofia")

    assert [(request.language, request.speaker_index) for request in requests] == [("de", 4), ("zh", 3)]
    assert requests[1].text == "早上好。"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "expected"),
    [("sofia", ("Sofia", 4)), (2, ("John", 2)), ("2", ("John", 2))],
)
def test_parse_speaker(value, expected):
    assert parse_speaker(value) == expected


@pytest.mark.unit
def test_load_requests_rejects_duplicate_ids(tmp_path):
    manifest = tmp_path / "requests.jsonl"
    manifest.write_text(
        '{"id":"same","text":"Hallo.","language":"de"}\n' '{"id":"same","text":"Ciao.","language":"it"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate id"):
        load_requests(manifest, default_speaker="Sofia")


@pytest.mark.unit
def test_safe_filename_removes_path_components():
    assert safe_filename("../../talk 1", 3) == "00003_talk_1.wav"


@pytest.mark.unit
def test_summarize_results_reports_interpolated_p95():
    results = [
        {"generation_time_seconds": 0.2, "audio_duration_seconds": 1.0, "rtf": 0.2},
        {"generation_time_seconds": 0.4, "audio_duration_seconds": 2.0, "rtf": 0.2},
    ]

    summary = summarize_results(results, model_load_seconds=3.0)

    assert summary["num_requests"] == 2
    assert summary["generation_time_median_seconds"] == pytest.approx(0.3)
    assert summary["generation_time_p95_seconds"] == pytest.approx(0.39)
    assert summary["rtf_mean"] == pytest.approx(0.2)
