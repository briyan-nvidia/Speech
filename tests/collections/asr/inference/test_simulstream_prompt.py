from __future__ import annotations

from types import SimpleNamespace
import numpy as np

from nemo.collections.asr.inference.pipelines.base_pipeline import TranscribeStepOutput
from nemo.collections.asr.inference.utils.simulstream_pipeline_adapter import NeMoStreamingPipelineAdapter


def test_source_language_is_forwarded_to_first_asr_request():
    class Pipeline:
        device = "cpu"
        nmt_enabled = False

        def __init__(self):
            self.requests = []

        def transcribe_step(self, requests):
            self.requests.extend(requests)
            return [TranscribeStepOutput(stream_id=0)]

    adapter = NeMoStreamingPipelineAdapter.__new__(NeMoStreamingPipelineAdapter)
    adapter.pipeline = Pipeline()
    adapter.config = SimpleNamespace(speech_chunk_size=0.08)
    adapter.stream_id = 0
    adapter.latency_unit = "word"
    adapter.per_stream_boosting_requests = None
    adapter.detailed_log_path = None
    adapter.set_source_language("de")
    adapter._reset_stream_state()

    adapter.process_chunk(np.zeros(1280, dtype=np.float32))
    adapter.process_chunk(np.zeros(1280, dtype=np.float32))

    assert adapter.pipeline.requests[0].options.language_code == "de"
    assert adapter.pipeline.requests[1].options is None
