# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evaluate the intelligibility of synthesized streaming-translation speech.

The input is the JSONL output produced by ``asr_streaming_infer.py`` with TTS
enabled. The script transcribes both the emitted TTS chunks and the assembled
utterance with multilingual Whisper, then computes WER and CER against the
exact text passed to MagpieTTS.

Example:
    python examples/tts/evaluate_streaming_tts.py \
        --input-jsonl outputs/s2s.jsonl \
        --output-dir outputs/s2s_intelligibility \
        --language de \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvaluationItem:
    """One synthesized audio file and the exact text used to produce it."""

    item_id: str
    level: str
    audio_filepath: str
    reference: str


def load_evaluation_items(input_jsonl: Path, level: str, language: str) -> list[EvaluationItem]:
    """Load chunk and/or assembled-utterance items from a streaming S2S manifest."""
    if level not in {"chunk", "utterance", "both"}:
        raise ValueError(f"unsupported evaluation level: {level}")

    records = []
    for line_number, line in enumerate(input_jsonl.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON on line {line_number}: {error.msg}") from error
    if not records:
        raise ValueError(f"no records found in {input_jsonl}")

    items = []
    include_chunks = level in {"chunk", "both"}
    include_utterances = level in {"utterance", "both"}
    for record_index, record in enumerate(records):
        segments = record.get("tts_segments") or []
        if not segments:
            raise ValueError(f"record {record_index} has no tts_segments")

        if include_chunks:
            for chunk_index, segment in enumerate(segments):
                items.append(
                    EvaluationItem(
                        item_id=f"{record_index}:chunk:{chunk_index}",
                        level="chunk",
                        audio_filepath=_resolve_audio_path(segment.get("audio_filepath"), input_jsonl),
                        reference=_require_text(segment.get("text"), f"record {record_index}, chunk {chunk_index}"),
                    )
                )

        if include_utterances:
            reference = _join_segment_text([segment.get("text", "") for segment in segments], language)
            items.append(
                EvaluationItem(
                    item_id=f"{record_index}:utterance",
                    level="utterance",
                    audio_filepath=_resolve_audio_path(record.get("speech_filepath"), input_jsonl),
                    reference=_require_text(reference, f"record {record_index}, assembled utterance"),
                )
            )
    return items


def score_transcriptions(
    items: list[EvaluationItem], hypotheses: list[str], language: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Normalize transcripts and compute per-item plus micro-averaged WER/CER."""
    if len(items) != len(hypotheses):
        raise ValueError(f"received {len(hypotheses)} hypotheses for {len(items)} evaluation items")

    from nemo.collections.tts.parts.utils.tts_dataset_utils import get_text_processor

    processor = get_text_processor(language)
    details = []
    for item, hypothesis in zip(items, hypotheses):
        reference_normalized = processor.process_text_for_wer(item.reference)
        hypothesis_normalized = processor.process_text_for_wer(hypothesis)
        details.append(
            {
                **asdict(item),
                "hypothesis": hypothesis,
                "reference_normalized": reference_normalized,
                "hypothesis_normalized": hypothesis_normalized,
                "wer": _error_statistics([reference_normalized], [hypothesis_normalized], False),
                "cer": _error_statistics([reference_normalized], [hypothesis_normalized], True),
            }
        )

    levels = {}
    for level in ("chunk", "utterance"):
        selected = [detail for detail in details if detail["level"] == level]
        if not selected:
            continue
        references = [detail["reference_normalized"] for detail in selected]
        predictions = [detail["hypothesis_normalized"] for detail in selected]
        levels[level] = {
            "num_items": len(selected),
            "wer": _error_statistics(references, predictions, False),
            "cer": _error_statistics(references, predictions, True),
        }

    normalized_language = language.replace("_", "-").lower().split("-")[0]
    summary = {
        "language": normalized_language,
        "levels": levels,
        "headline_level": "utterance" if "utterance" in levels else "chunk",
        "wer_note": (
            "WER is not linguistically meaningful for unsegmented Chinese/Japanese; use CER as the primary metric."
            if normalized_language in {"zh", "ja"}
            else "WER uses whitespace-delimited words after NeMo text normalization."
        ),
    }
    return details, summary


def evaluate(args: argparse.Namespace) -> None:
    """Transcribe synthesized speech and write item-level and aggregate scores."""
    from nemo.collections.tts.parts.utils.tts_dataset_utils import WhisperTranscriber

    items = load_evaluation_items(args.input_jsonl, args.level, args.language)
    transcriber = WhisperTranscriber(device=args.device, model_name=args.asr_model)
    hypotheses = transcriber.transcribe(
        audio_paths=[Path(item.audio_filepath) for item in items],
        batch_size=args.batch_size,
        language=args.language,
    )
    details, summary = score_transcriptions(items, hypotheses, args.language)
    summary.update(
        {
            "input_jsonl": str(args.input_jsonl.resolve()),
            "asr_model": args.asr_model,
            "supports_longform_audio": True,
            "asr_decoding": "Whisper timestamp-based long-form decoding for audio longer than 30 seconds",
            "normalization": "NeMo MagpieTTS language-aware evaluation normalization",
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    details_path = args.output_dir / "items.jsonl"
    details_path.write_text(
        "".join(json.dumps(detail, ensure_ascii=False) + "\n" for detail in details), encoding="utf-8"
    )
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--language", required=True, help="Spoken-language code, for example de, it, or zh.")
    parser.add_argument("--level", choices=("chunk", "utterance", "both"), default="both")
    parser.add_argument("--asr-model", default="openai/whisper-large-v3")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def _error_statistics(references: list[str], hypotheses: list[str], use_cer: bool) -> dict[str, Any]:
    from nemo.collections.asr.metrics.wer import word_error_rate_detail

    error_rate, reference_units, insertion_rate, deletion_rate, substitution_rate = word_error_rate_detail(
        hypotheses=hypotheses, references=references, use_cer=use_cer
    )
    total_errors = round(error_rate * reference_units) if reference_units else 0
    return {
        "rate": error_rate,
        "reference_units": reference_units,
        "errors": total_errors,
        "insertions": round(insertion_rate * reference_units) if reference_units else 0,
        "deletions": round(deletion_rate * reference_units) if reference_units else 0,
        "substitutions": round(substitution_rate * reference_units) if reference_units else 0,
    }


def _resolve_audio_path(value: Any, manifest_path: Path) -> str:
    if not value:
        raise ValueError("missing synthesized audio filepath")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"synthesized audio does not exist: {path}")
    return str(path)


def _require_text(value: Any, location: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"empty TTS reference text in {location}")
    return text


def _join_segment_text(texts: list[str], language: str) -> str:
    language = language.replace("_", "-").lower().split("-")[0]
    if language in {"ja", "zh"}:
        return "".join(text.strip() for text in texts)
    return " ".join(text.strip() for text in texts)


if __name__ == "__main__":
    evaluate(parse_args())
