"""Synthetic audio signals for unit tests."""

import numpy as np


def make_sine_int16(
    freq_hz: float,
    sample_rate: int,
    duration_s: float,
    *,
    amplitude: float = 0.5,
    phase_rad: float = 0.0,
) -> np.ndarray:
    """Build a normalized int16 sine wave like a microphone capture."""
    n_samples = int(sample_rate * duration_s)
    t = np.arange(n_samples, dtype=np.float64) / sample_rate
    x = amplitude * np.sin(2 * np.pi * freq_hz * t + phase_rad)
    return (x * 32767.0).astype(np.int16)


def make_noise_int16(
    sample_rate: int, duration_s: float, *, amplitude: float = 0.5
) -> np.ndarray:
    """Build random int16 noise for poor-fit scenarios."""
    n_samples = int(sample_rate * duration_s)
    rng = np.random.default_rng(42)
    x = amplitude * rng.standard_normal(n_samples)
    return (x * 32767.0).astype(np.int16)
