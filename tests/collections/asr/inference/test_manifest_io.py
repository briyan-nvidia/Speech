# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

from nemo.collections.asr.inference.utils.manifest_io import dump_output


def test_dump_output_preserves_incremental_translation_timing(tmp_path):
    output_filename = tmp_path / "predictions.jsonl"
    output = {
        0: {
            "audio_filepath": "/audio/example.wav",
            "text": "Hello.",
            "translation": "Hallo.",
            "segments": [],
            "translation_segments": [("Hallo", 0.48), (".", 0.96)],
            "final_translation_segments": [("Hallo.", 0.96)],
            "tts_min_chunk_units": 6,
            "tts_max_chunk_units": 12,
            "tts_chunking_policy": "queue_aware",
            "tts_target_buffer_seconds": 2.0,
            "tts_estimated_source_seconds": 0.96,
            "tts_raw_to_source_duration_ratio": 1.25,
            "tts_played_to_source_duration_ratio": 1.1,
        }
    }

    dump_output(output, str(output_filename), output_dir=str(tmp_path / "segments"))

    item = json.loads(output_filename.read_text())
    assert item["translation_segments"] == [["Hallo", 0.48], [".", 0.96]]
    assert item["tts_min_chunk_units"] == 6
    assert item["tts_max_chunk_units"] == 12
    assert item["tts_chunking_policy"] == "queue_aware"
    assert item["tts_target_buffer_seconds"] == 2.0
    assert item["tts_estimated_source_seconds"] == 0.96
    assert item["tts_raw_to_source_duration_ratio"] == 1.25
    assert item["tts_played_to_source_duration_ratio"] == 1.1
