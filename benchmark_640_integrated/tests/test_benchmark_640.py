from __future__ import annotations

import numpy as np

from bruise_repro.benchmark_640 import IMAGE_SIZE, _summarize


def test_fixed_image_size() -> None:
    assert IMAGE_SIZE == 640


def test_latency_summary() -> None:
    result = _summarize([10.0, 20.0, 30.0])
    assert result["mean_latency_ms"] == 20.0
    assert result["median_latency_ms"] == 20.0
    assert np.isclose(result["fps_from_mean"], 50.0)
