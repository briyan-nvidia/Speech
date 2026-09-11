# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthesize a text-only JSONL manifest with a pretrained MagpieTTS model.

Each input line must contain ``text`` and ``language``. An optional ``id`` and
``speaker`` (name or integer index) may also be provided. The script writes one
WAV per input line, an output JSONL manifest, and aggregate timing statistics.

Example:
    python examples/tts/magpietts_synthesize.py \
        --input-jsonl requests.jsonl \
        --output-dir outputs/magpie \
        --model nvidia/magpie_tts_multilingual_357m \
        --revision v2607
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import soundfile as sf
import torch

from nemo.collections.tts.inference import MagpieTTSBackend, MagpieTTSConfig, parse_speaker
from nemo.collections.tts.inference.magpietts_backend import SPEAKER_MAP, SUPPORTED_LANGUAGES


@dataclass(frozen=True)
class SynthesisRequest:
    """A validated text synthesis request."""

    request_id: str
    text: str
    language: str
    speaker: str
    speaker_index: int


def load_requests(path: Path, default_speaker: str) -> list[SynthesisRequest]:
    """Load and validate synthesis requests from a JSONL file."""
    requests = []
    seen_ids = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON on line {line_number}: {error.msg}") from error

        text = str(record.get("text", "")).strip()
        language = str(record.get("language", "")).strip().lower()
        request_id = str(record.get("id", len(requests))).strip()
        if not text:
            raise ValueError(f"empty text on line {line_number}")
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"unsupported language {language!r} on line {line_number}")
        if not request_id:
            raise ValueError(f"empty id on line {line_number}")
        if request_id in seen_ids:
            raise ValueError(f"duplicate id {request_id!r} on line {line_number}")

        speaker, speaker_index = parse_speaker(record.get("speaker", default_speaker))
        requests.append(SynthesisRequest(request_id, text, language, speaker, speaker_index))
        seen_ids.add(request_id)

    if not requests:
        raise ValueError(f"no synthesis requests found in {path}")
    return requests


def safe_filename(request_id: str, index: int) -> str:
    """Return a deterministic filename that cannot escape the output directory."""
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", request_id).strip("._") or "request"
    return f"{index:05d}_{normalized}.wav"


def percentile(values: list[float], quantile: float) -> float:
    """Compute a linearly interpolated percentile without an optional dependency."""
    if not values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def summarize_results(results: list[dict[str, Any]], model_load_seconds: float) -> dict[str, Any]:
    """Aggregate synthesis latency and real-time-factor measurements."""
    generation_times = [result["generation_time_seconds"] for result in results]
    rtfs = [result["rtf"] for result in results]
    return {
        "num_requests": len(results),
        "model_load_seconds": model_load_seconds,
        "total_audio_seconds": sum(result["audio_duration_seconds"] for result in results),
        "total_generation_seconds": sum(generation_times),
        "generation_time_median_seconds": percentile(generation_times, 0.5),
        "generation_time_p95_seconds": percentile(generation_times, 0.95),
        "rtf_mean": sum(rtfs) / len(rtfs),
        "rtf_p95": percentile(rtfs, 0.95),
    }


def synthesize(args: argparse.Namespace) -> None:
    """Run text-only MagpieTTS synthesis and write audio and metrics."""
    if args.warmup_runs < 0:
        raise ValueError("warmup_runs must be non-negative")
    if args.cfg_scale <= 0:
        raise ValueError("cfg_scale must be positive")

    requests = load_requests(args.input_jsonl, args.speaker)
    output_dir = args.output_dir.resolve()
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    backend = MagpieTTSBackend(
        MagpieTTSConfig(
            model=args.model,
            revision=args.revision,
            model_filename=args.model_filename,
            device=args.device,
            speaker=args.speaker,
            use_cfg=args.use_cfg,
            cfg_scale=args.cfg_scale,
            apply_text_normalization=args.apply_text_normalization,
            seed=args.seed,
        )
    )
    if backend.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(backend.device)

    warmup_requests = list({(request.language, request.speaker_index): request for request in requests}.values())
    for _ in range(args.warmup_runs):
        for request in warmup_requests:
            backend.synthesize(request.text, request.language, request.speaker_index)
    backend.seed(args.seed)

    results = []
    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for index, request in enumerate(requests):
            synthesis = backend.synthesize(request.text, request.language, request.speaker_index)
            waveform = synthesis.waveform
            sample_rate = synthesis.sample_rate
            num_samples = waveform.size
            audio_duration_seconds = synthesis.audio_duration_seconds
            audio_path = audio_dir / safe_filename(request.request_id, index)
            sf.write(audio_path, waveform, sample_rate, subtype="PCM_16")

            result = {
                **asdict(request),
                "audio_filepath": str(audio_path),
                "sample_rate": sample_rate,
                "num_samples": num_samples,
                "audio_duration_seconds": audio_duration_seconds,
                "generation_time_seconds": synthesis.generation_time_seconds,
                "rtf": synthesis.rtf,
                "use_cfg": args.use_cfg,
                "apply_text_normalization": args.apply_text_normalization,
            }
            manifest.write(json.dumps(result, ensure_ascii=False) + "\n")
            results.append(result)

    summary = summarize_results(results, backend.model_load_seconds)
    summary.update(
        {
            "model": args.model,
            "revision": args.revision,
            "checkpoint": backend.checkpoint,
            "device": str(backend.device),
            "speaker_default": args.speaker,
            "use_cfg": args.use_cfg,
            "apply_text_normalization": args.apply_text_normalization,
            "seed": args.seed,
            "warmup_runs": args.warmup_runs,
            "cfg_scale": args.cfg_scale if args.use_cfg else None,
            "model_parameters": sum(parameter.numel() for parameter in backend.model.parameters()),
            "model_dtype": str(next(backend.model.parameters()).dtype),
            "cuda_device_name": (
                torch.cuda.get_device_name(backend.device) if backend.device.type == "cuda" else None
            ),
            "peak_cuda_memory_bytes": (
                torch.cuda.max_memory_allocated(backend.device) if backend.device.type == "cuda" else None
            ),
        }
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="nvidia/magpie_tts_multilingual_357m")
    parser.add_argument("--revision", default="v2607")
    parser.add_argument("--model-filename", default="magpie_tts_multilingual_357m.nemo")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--speaker", default="Sofia", choices=list(SPEAKER_MAP))
    parser.add_argument("--use-cfg", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cfg-scale", type=float, default=2.5)
    parser.add_argument("--apply-text-normalization", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=0,
        help="Untimed synthesis passes per language and speaker before measurement.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    synthesize(parse_args())
