"""Shared pytest fixtures for ForkCal unit tests."""

import pytest

from analysis import AudioAnalyzer


@pytest.fixture
def sample_rate() -> int:
    return 48000


@pytest.fixture
def acquisition_period() -> float:
    return 1.0


@pytest.fixture
def reference_freq() -> float:
    return 360.0


@pytest.fixture
def analyzer(sample_rate, acquisition_period, reference_freq) -> AudioAnalyzer:
    """AudioAnalyzer instance without starting PyAudio capture."""
    return AudioAnalyzer(
        device_name="0: Test Device",
        sample_rate=sample_rate,
        acquisition_period=acquisition_period,
        num_averages=3,
        reference_freq=reference_freq,
        freq_estimation_method="phase_fit",
    )
