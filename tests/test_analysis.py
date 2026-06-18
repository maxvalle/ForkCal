"""Unit tests for analysis.py signal processing and timegrapher logic."""

import numpy as np
import pytest

from analysis import AudioAnalyzer
from tests.signals import make_noise_int16, make_sine_int16


def spd_from_freq_error(freq_hz: float, ref_hz: float) -> float:
    """Seconds per day from measured vs reference frequency."""
    return ((freq_hz - ref_hz) / ref_hz) * 86400.0


class TestAudioAnalyzerInit:
    def test_chunk_size_from_acquisition_period(self, sample_rate):
        analyzer = AudioAnalyzer(
            device_name="0: Test",
            sample_rate=sample_rate,
            acquisition_period=0.5,
            num_averages=1,
        )
        assert analyzer.chunk_size == sample_rate // 2

    def test_minimum_chunk_size(self):
        analyzer = AudioAnalyzer(
            device_name="0: Test",
            sample_rate=8000,
            acquisition_period=0.01,
            num_averages=1,
        )
        assert analyzer.chunk_size >= 256

    def test_device_index_parsed_from_name(self, analyzer):
        assert analyzer.device_index == 0


class TestBandpassFilter:
    def test_filter_center_matches_reference(self, analyzer, sample_rate, acquisition_period):
        n = int(sample_rate * acquisition_period)
        _, lowcut, highcut, center = analyzer._compute_bandpass_filter(n)
        assert center == analyzer.reference_freq
        assert lowcut < center < highcut

    def test_filter_is_cached(self, analyzer, sample_rate, acquisition_period):
        n = int(sample_rate * acquisition_period)
        first = analyzer._compute_bandpass_filter(n)
        second = analyzer._compute_bandpass_filter(n)
        assert first[0] is second[0]

    def test_set_reference_frequency_invalidates_cache(
        self, analyzer, sample_rate, acquisition_period
    ):
        n = int(sample_rate * acquisition_period)
        first_coeff, _, _, _ = analyzer._compute_bandpass_filter(n)
        analyzer.set_reference_frequency(720.0)
        second_coeff, lowcut, highcut, center = analyzer._compute_bandpass_filter(n)
        assert center == 720.0
        assert first_coeff is not second_coeff
        assert lowcut < 720.0 < highcut


class TestSpectrumAnalysis:
    def test_analyze_spectrum_populates_data(self, analyzer, sample_rate, reference_freq):
        audio = make_sine_int16(reference_freq, sample_rate, analyzer.acquisition_period)
        analyzer._analyze_spectrum(audio)

        frequencies, psd_db = analyzer.get_spectrum()
        assert frequencies is not None
        assert psd_db is not None
        assert len(frequencies) == len(psd_db)
        assert analyzer.nperseg is not None

    def test_compute_rbw_after_spectrum(self, analyzer, sample_rate, reference_freq):
        audio = make_sine_int16(reference_freq, sample_rate, analyzer.acquisition_period)
        analyzer._analyze_spectrum(audio)

        rbw = analyzer.compute_rbw()
        assert rbw is not None
        assert rbw > 0
        expected = (sample_rate / analyzer.nperseg) * 1.5
        assert rbw == pytest.approx(expected)

    def test_spectrum_moving_average(self, sample_rate, reference_freq):
        analyzer = AudioAnalyzer(
            device_name="0: Test",
            sample_rate=sample_rate,
            acquisition_period=1.0,
            num_averages=3,
            reference_freq=reference_freq,
        )
        audio = make_sine_int16(reference_freq, sample_rate, 1.0)
        for _ in range(3):
            analyzer._analyze_spectrum(audio)

        assert len(analyzer.spectrum_buffer) == 3
        _, psd_db = analyzer.get_spectrum()
        assert psd_db is not None


class TestTimegrapherPhaseFit:
    def test_estimates_on_frequency_sine(
        self, analyzer, sample_rate, reference_freq, acquisition_period
    ):
        audio = make_sine_int16(reference_freq, sample_rate, acquisition_period)
        analyzer._analyze_timegrapher(audio)

        freq, deviation = analyzer.get_timegrapher_data()
        assert analyzer.get_signal_quality() is True
        assert freq == pytest.approx(reference_freq, rel=1e-3)
        assert deviation == pytest.approx(0.0, abs=1.0)

    def test_detects_positive_frequency_error(
        self, analyzer, sample_rate, reference_freq, acquisition_period
    ):
        fast_freq = reference_freq + 0.1
        audio = make_sine_int16(fast_freq, sample_rate, acquisition_period)
        analyzer._analyze_timegrapher(audio)

        freq, deviation = analyzer.get_timegrapher_data()
        expected_spd = spd_from_freq_error(fast_freq, reference_freq)

        assert analyzer.get_signal_quality() is True
        assert freq == pytest.approx(fast_freq, rel=1e-3)
        assert deviation == pytest.approx(expected_spd, rel=0.05)

    def test_rejects_poor_signal(self, analyzer, sample_rate, acquisition_period):
        audio = make_noise_int16(sample_rate, acquisition_period)
        analyzer._analyze_timegrapher(audio, residual_threshold=0.01)

        freq, deviation = analyzer.get_timegrapher_data()
        assert analyzer.get_signal_quality() is False
        assert freq is None
        assert deviation is None


class TestTimegrapherSineFit:
    def test_estimates_on_frequency_sine(
        self, sample_rate, reference_freq, acquisition_period
    ):
        analyzer = AudioAnalyzer(
            device_name="0: Test",
            sample_rate=sample_rate,
            acquisition_period=acquisition_period,
            num_averages=1,
            reference_freq=reference_freq,
            freq_estimation_method="sine_fit",
        )
        audio = make_sine_int16(reference_freq, sample_rate, acquisition_period)
        analyzer._analyze_timegrapher(audio)

        freq, deviation = analyzer.get_timegrapher_data()
        assert analyzer.get_signal_quality() is True
        assert freq == pytest.approx(reference_freq, rel=1e-4)
        assert deviation == pytest.approx(0.0, abs=0.5)

    def test_poor_fit_marks_bad_quality(self, sample_rate, reference_freq, acquisition_period):
        analyzer = AudioAnalyzer(
            device_name="0: Test",
            sample_rate=sample_rate,
            acquisition_period=acquisition_period,
            num_averages=1,
            reference_freq=reference_freq,
            freq_estimation_method="sine_fit",
        )
        audio = make_noise_int16(sample_rate, acquisition_period)
        analyzer._analyze_timegrapher(audio, residual_threshold=0.01)

        freq, deviation = analyzer.get_timegrapher_data()
        assert analyzer.get_signal_quality() is False
        assert freq is None
        assert deviation is None


class TestInstantaneousPhaseFitDirect:
    def test_phase_fit_on_pure_sine(
        self, analyzer, sample_rate, reference_freq, acquisition_period
    ):
        n = int(sample_rate * acquisition_period)
        t = np.arange(n) / sample_rate
        x = 0.5 * np.sin(2 * np.pi * reference_freq * t)
        x_cropped = x[int(0.15 * n) : int(0.85 * n)]
        t_cropped = t[int(0.15 * n) : int(0.85 * n)]

        analyzer.instantaneous_phase_fit(
            x_cropped,
            t_cropped,
            reference_freq,
            phase_residual_threshold=0.5,
            debug_data={},
            debug=False,
        )

        freq, deviation = analyzer.get_timegrapher_data()
        assert analyzer.get_signal_quality() is True
        assert freq == pytest.approx(reference_freq, rel=1e-3)
        assert deviation == pytest.approx(0.0, abs=1.0)


class TestDebugMode:
    def test_debug_queue_receives_data(self, analyzer, sample_rate, reference_freq):
        analyzer.set_debug_mode(True)
        audio = make_sine_int16(reference_freq, sample_rate, analyzer.acquisition_period)
        analyzer._analyze_timegrapher(audio)

        debug_data = analyzer.get_debug_plot_data()
        assert debug_data is not None
        assert debug_data["fit_method"] == "phase_fit"
        assert "estimated_freq" in debug_data
        assert "deviation_spd" in debug_data

    def test_get_filter_params_after_analysis(
        self, analyzer, sample_rate, reference_freq, acquisition_period
    ):
        audio = make_sine_int16(reference_freq, sample_rate, acquisition_period)
        analyzer._analyze_timegrapher(audio)

        lowcut, highcut, center = analyzer.get_filter_params()
        assert center == reference_freq
        assert lowcut < center < highcut
