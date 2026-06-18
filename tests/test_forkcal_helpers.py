"""Unit tests for pure helper functions in forkcal.py."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from forkcal import (
    SpectrumAnalyzerGUI,
    _debug_acq_duration,
    _debug_mag_db,
    _debug_y_limits,
    _spectrum_band_hz,
    _spectrum_plot_xy,
    _spectrum_x_range_hz,
)


class TestSpectrumBandHz:
    def test_symmetric_band_when_half_ref_above_10hz(self):
        f_lo, f_hi = _spectrum_band_hz(360.0)
        assert f_lo < 360.0 < f_hi
        # Geometric mean of core band (180, 720) is 360 Hz
        assert np.isclose(np.sqrt(180.0 * 720.0), 360.0)

    def test_low_ref_clamps_core_band_and_keeps_reference_centered(self):
        f_ref = 15.0
        f_lo, f_hi = _spectrum_band_hz(f_ref)
        assert f_lo < f_ref < f_hi
        log_center = (np.log10(f_lo) + np.log10(f_hi)) / 2.0
        assert np.isclose(10**log_center, f_ref, rtol=1e-6)
        # Core low edge is 10 Hz; 20% log padding extends the view below that.
        assert f_lo < 10.0

    def test_reference_at_center_in_log_space(self):
        f_ref = 300.0
        f_lo, f_hi = _spectrum_band_hz(f_ref)
        log_center = (np.log10(f_lo) + np.log10(f_hi)) / 2.0
        assert np.isclose(10**log_center, f_ref, rtol=1e-6)


class TestSpectrumPlotHelpers:
    def test_x_range_is_log10_hz(self):
        fmin, fmax = 100.0, 1000.0
        assert _spectrum_x_range_hz(fmin, fmax) == (2.0, 3.0)

    def test_plot_xy_filters_and_converts_to_log(self):
        frequencies = np.array([50.0, 100.0, 200.0, 500.0, 2000.0])
        psd_db = np.array([-80.0, -70.0, -60.0, -55.0, -90.0])
        log_xrange = _spectrum_x_range_hz(100.0, 1000.0)

        x_log, y_db = _spectrum_plot_xy(frequencies, psd_db, log_xrange)

        assert len(x_log) == 3
        assert np.allclose(x_log, np.log10([100.0, 200.0, 500.0]))
        assert np.allclose(y_db, [-70.0, -60.0, -55.0])


class TestDebugHelpers:
    def test_mag_db_converts_linear_to_db(self):
        h = np.array([1.0, 0.1, 1e-20])
        db = _debug_mag_db(h)
        assert np.isclose(db[0], 0.0)
        assert np.isclose(db[1], -20.0)
        assert db[2] < -280.0

    def test_acq_duration_from_time_vector(self):
        debug_data = {"time": np.linspace(0.0, 2.5, 100)}
        assert _debug_acq_duration(debug_data) == pytest.approx(2.5)

    def test_acq_duration_fallback(self):
        assert _debug_acq_duration({}) == 1.0
        assert _debug_acq_duration({}, fallback=0.25) == 0.25

    def test_y_limits_adds_margin(self):
        y_lo, y_hi = _debug_y_limits(np.array([0.0, 1.0]))
        assert y_lo < 0.0
        assert y_hi > 1.0

    def test_y_limits_empty_defaults(self):
        assert _debug_y_limits() == (-1.0, 1.0)


class TestParseAcquisitionPeriod:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("250 ms", 0.25),
            ("500 ms", 0.5),
            ("1 s", 1.0),
            ("2 s", 2.0),
            ("60 s", 60.0),
        ],
    )
    def test_parses_common_periods(self, text, expected):
        gui = MagicMock()
        gui.acq_period_combo.currentText.return_value = text
        assert SpectrumAnalyzerGUI.parse_acquisition_period(gui) == expected
