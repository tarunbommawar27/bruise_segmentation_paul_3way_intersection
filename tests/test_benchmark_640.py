"""
Pure-function tests for pipeline/benchmark_640.py -- no GPU, no dataset, no
trained checkpoint required, so these run on any machine (including this
CPU-only laptop) as a fast guard against accidentally breaking the timing
math while modularizing/commenting the benchmark code.
"""
from __future__ import annotations

import numpy as np

from pipeline.benchmark_640 import IMAGE_SIZE, _summarize


def test_fixed_image_size() -> None:
    # The whole point of this module is the 640 protocol -- if this ever
    # changes it should be a deliberate, visible edit, not a typo.
    assert IMAGE_SIZE == 640


def test_latency_summary_basic_stats() -> None:
    result = _summarize([10.0, 20.0, 30.0])
    assert result["mean_latency_ms"] == 20.0
    assert result["median_latency_ms"] == 20.0
    assert np.isclose(result["fps_from_mean"], 50.0)
    assert np.isclose(result["fps_from_median"], 50.0)


def test_latency_summary_percentiles() -> None:
    # 100 evenly spaced values 1..100ms: p90/p95 should land near 90/95.
    values = [float(v) for v in range(1, 101)]
    result = _summarize(values)
    assert np.isclose(result["p90_latency_ms"], 90.1, atol=0.5)
    assert np.isclose(result["p95_latency_ms"], 95.05, atol=0.5)


def test_latency_summary_rejects_empty() -> None:
    try:
        _summarize([])
    except ValueError:
        pass
    else:
        raise AssertionError("_summarize([]) should raise ValueError, not return silently.")
