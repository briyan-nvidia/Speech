# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from examples.tts.evaluate_streaming_tts import EvaluationItem, load_evaluation_items, score_transcriptions


@pytest.mark.unit
def test_load_evaluation_items_reads_chunks_and_assembled_utterance(tmp_path):
    chunk_0 = tmp_path / "chunk_0.wav"
    chunk_1 = tmp_path / "chunk_1.wav"
    utterance = tmp_path / "utterance.wav"
    for path in (chunk_0, chunk_1, utterance):
        path.touch()
    manifest = tmp_path / "output.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "speech_filepath": str(utterance),
                "tts_segments": [
                    {"text": "Guten Morgen,", "audio_filepath": str(chunk_0)},
                    {"text": "alle zusammen.", "audio_filepath": str(chunk_1)},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    items = load_evaluation_items(manifest, level="both", language="de")

    assert [item.item_id for item in items] == ["0:chunk:0", "0:chunk:1", "0:utterance"]
    assert items[-1].reference == "Guten Morgen, alle zusammen."


@pytest.mark.unit
def test_load_evaluation_items_concatenates_chinese_without_spaces(tmp_path):
    chunk = tmp_path / "chunk.wav"
    utterance = tmp_path / "utterance.wav"
    chunk.touch()
    utterance.touch()
    manifest = tmp_path / "output.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "speech_filepath": str(utterance),
                "tts_segments": [
                    {"text": "大家好，", "audio_filepath": str(chunk)},
                    {"text": "欢迎。", "audio_filepath": str(chunk)},
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    items = load_evaluation_items(manifest, level="utterance", language="zh")

    assert items[0].reference == "大家好，欢迎。"


@pytest.mark.unit
def test_score_transcriptions_reports_micro_averaged_wer_and_cer():
    items = [
        EvaluationItem("0", "chunk", "/tmp/0.wav", "Hallo Welt"),
        EvaluationItem("1", "chunk", "/tmp/1.wav", "Guten Morgen"),
    ]

    details, summary = score_transcriptions(items, ["Hallo Welt", "Guten Abend"], language="de")

    assert details[0]["wer"]["rate"] == 0.0
    assert details[1]["wer"]["rate"] == 0.5
    assert summary["levels"]["chunk"]["wer"]["rate"] == 0.25
    assert summary["levels"]["chunk"]["wer"]["errors"] == 1
    assert summary["levels"]["chunk"]["cer"]["rate"] > 0.0
