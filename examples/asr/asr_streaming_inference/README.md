# Universal Streaming Inference

The `asr_streaming_infer.py` script enables streaming inference for both buffered (CTC/RNNT/TDT) and cache-aware (CTC/RNNT) ASR models. It supports processing a single audio file, a directory of audio files, or a manifest file.

Beyond streaming ASR, the script also supports:

* **Inverse Text Normalization (ITN)**
* **End-of-Utterance (EoU) Detection**
* **Streaming Speech Translation (requires vLLM installation)**
* **Incremental Speech Output with MagpieTTS**
* **Word-level and Segment-level Output**

All related configurations can be found in the `../conf/asr_streaming_inference/` directory.

## Speech-to-speech inference

The buffered RNNT pipeline can synthesize temporally committed translation increments. Enable both NMT and TTS and
provide the Magpie target-language code:

```bash
python asr_streaming_infer.py \
    --config-path=../conf/asr_streaming_inference \
    --config-name=buffered_rnnt \
    audio_file=/path/to/audio.wav \
    output_filename=/path/to/output.jsonl \
    enable_nmt=true \
    nmt.target_language=German \
    enable_tts=true \
    tts.language=de
```

With `tts.incremental=true` (the default), the pipeline preserves one Magpie generation state per input stream. It
buffers only monotonic MT text committed by the temporal LCP policy. The default `tts.chunking_policy=queue_aware`
emits a bounded first phrase for low time to first audio, then prefers complete sentences while the playback queue
remains above `tts.target_buffer_seconds`. If the queue approaches underrun, it emits a phrase between
`tts.min_chunk_units` and `tts.max_chunk_units`. Units are whitespace-delimited words for most languages and characters
for Chinese and Japanese. Set `tts.chunking_policy=minimum` to emit whenever accumulated text reaches
`tts.min_chunk_units` or an MT boundary. Uncommitted MT hypotheses are never synthesized. The output JSONL records
every committed MT delta and its source-audio time under `translation_segments`, plus the final `speech_filepath`, each
emitted audio chunk and its timing under `tts_segments`, and `tts_time_to_first_audio_seconds`.
Set `tts.incremental=false` to retain post-inference, stateless synthesis.

Incremental synthesis can optionally shorten each emitted waveform without changing its pitch. Set
`tts.rate_control=fixed` with `tts.playback_rate=<rate>` to apply a constant rate, or set
`tts.rate_control=adaptive` to use the minimum rate needed to target `tts.target_lag_seconds`, capped by
`tts.maximum_playback_rate`. `tts.rate_control=source_paced` instead estimates the native target-to-source duration
ratio cumulatively from MT commit times and uses it as the playback rate, also capped by
`tts.maximum_playback_rate`. The default, `tts.rate_control=none`, preserves the waveform produced by Magpie. Each
`tts_segments` entry records the selected `playback_rate`, raw and played audio durations, and rate-control time.

Magpie model loading and lazy text-normalizer construction are service startup costs. Warm the model and each target
language with `tts.warmup_text=<target-language text>` before comparing steady-state time to first audio. By default,
ASR and Magpie use GPU 0 while NMT uses GPU 1.
Co-locating NMT and Magpie is supported when memory permits, but can reduce TTS throughput because vLLM reserves GPU
memory for its KV cache.

Evaluate TTS intelligibility by transcribing both the incremental audio chunks and the assembled target utterance.
The references are the exact committed MT strings sent to MagpieTTS:

```bash
python examples/tts/evaluate_streaming_tts.py \
    --input-jsonl=/path/to/output.jsonl \
    --output-dir=/path/to/tts_intelligibility \
    --language=de \
    --device=cuda:0
```

The evaluator writes micro-averaged WER/CER to `summary.json` and per-item transcriptions and edit counts to
`items.jsonl`. Treat assembled-utterance scores as the headline intelligibility result and chunk scores as a
diagnostic for incremental synthesis boundaries. CER is the primary metric for unsegmented Chinese and Japanese.

### Multilingual buffered ASR

`nvidia/nemotron-labs-unified-asr-0.6b` can replace the English-only ASR checkpoint without changing the MT/TTS
buffering policy. Use buffered RNNT inference and specify the **source speech** language with `lang` (or a manifest
row's `lang` field). This is independent of the MT target language and the TTS language. A per-recording language
overrides the global default; otherwise the checkpoint's declared default is used. Unsupported prompt keys fail
explicitly. The checkpoint's prompt dictionary is not a guarantee of trained language coverage; consult its model card.

For example, German speech to English text:

```bash
python examples/asr/asr_streaming_inference/asr_streaming_infer.py \
    --config-path=../conf/asr_streaming_inference --config-name=buffered_rnnt \
    asr.model_name=nvidia/nemotron-labs-unified-asr-0.6b lang=de \
    audio_file=/path/to/german.jsonl output_filename=/path/to/results.jsonl \
    streaming.left_padding_size=5.6 streaming.chunk_size=0.56 streaming.right_padding_size=0.56 \
    streaming.batch_size=1 streaming.request_type=frame return_tail_result=true \
    endpointing.stop_history_eou=1600 \
    enable_nmt=true nmt.model_name=Qwen/Qwen3-4B-Instruct-2507 \
    nmt.source_language=German nmt.target_language=English nmt.source_buffer.enabled=true
```

Use 16 kHz mono input. These settings correspond to `[70,7,7]` encoder frames of 80 ms. Model access requires the
appropriate Hugging Face permissions. For reproducible runs, download a pinned checkpoint revision and pass its
local `.nemo` path as `asr.model_name`. Leave the English-only model configured to retain the previous behavior.
