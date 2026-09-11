# Universal Streaming Inference

The `asr_streaming_infer.py` script enables streaming inference for both buffered (CTC/RNNT/TDT) and cache-aware (CTC/RNNT) ASR models. It supports processing a single audio file, a directory of audio files, or a manifest file.

Beyond streaming ASR, the script also supports:

* **Inverse Text Normalization (ITN)**
* **End-of-Utterance (EoU) Detection**
* **Streaming Speech Translation (requires vLLM installation)**
* **Offline Speech Output with MagpieTTS**
* **Word-level and Segment-level Output**

All related configurations can be found in the `../conf/asr_streaming_inference/` directory.

## Speech-to-speech inference

The buffered RNNT pipeline can synthesize translations that have been finalized by the MT policy. Enable both NMT
and TTS and provide the Magpie target-language code:

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

The output JSONL includes `speech_filepath`, `final_translation_segments`, and per-segment TTS timing. The final WAV
is assembled from finalized translation units; temporary MT hypotheses are never synthesized. By default, ASR and
Magpie use GPU 0 while NMT uses GPU 1. Co-locating NMT and Magpie is supported when memory permits, but can reduce TTS
throughput because vLLM reserves GPU memory for its KV cache.
