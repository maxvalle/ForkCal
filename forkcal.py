"""
ForkCal Tuning Fork Watch Timegrapher

For the adjustment of tuning fork watches, such as the Bulova Accutron and Omega f300 Hz watches.
Originally developed by joncox123, all rights reserved.
Refactored by maxvalle, all rights reserved.

Main GUI application using PySide6 and PyQtGraph
"""

import sys

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QPen
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QGridLayout,
    QGroupBox,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from analysis import AudioAnalyzer

pg.setConfigOptions(antialias=True, background="w", foreground="k")

_PLOT_FONT = QFont()
_PLOT_FONT.setPointSize(12)
_PLOT_TITLE_PT = "14pt"
_PLOT_AXIS_LABEL_PT = "13pt"
_PLOT_GROUP_LABEL_FONT = QFont()
_PLOT_GROUP_LABEL_FONT.setPointSize(10)
_PLOT_GROUP_LABEL_FONT.setBold(True)

# Debug window: match original matplotlib weight and readability.
_DEBUG_FONT = QFont()
_DEBUG_FONT.setPointSize(11)
_DEBUG_TITLE_PT = "12pt"
_DEBUG_LEGEND_PT = "10pt"
_DEBUG_PEN_FILTER = pg.mkPen("b", width=2.5)
_DEBUG_PEN_SIGNAL = pg.mkPen("b", width=2.5)
_DEBUG_PEN_FIT = pg.mkPen("r", width=3.0, style=Qt.PenStyle.DotLine)
_DEBUG_PEN_PHASE = pg.mkPen("r", width=2.5)
_DEBUG_PEN_VLINE_LO = pg.mkPen("r", width=2.5, style=Qt.PenStyle.DashLine)
_DEBUG_PEN_VLINE_HI = pg.mkPen("r", width=2.5, style=Qt.PenStyle.DashLine)
_DEBUG_PEN_VLINE_CTR = pg.mkPen((0, 110, 0), width=2.5, style=Qt.PenStyle.DashLine)
_DEBUG_PEN_ZERO = pg.mkPen((128, 128, 128), width=2.0, style=Qt.PenStyle.DashLine)
_DEBUG_INFO_COLOR = "#1a1a1a"

# Shared plot chrome; timegrapher gets extra top room so yellow group labels are not clipped.
_PLOT_TOP_MARGIN = 10
_PLOT_TOP_MARGIN_TIME = 16
_PLOT_BOTTOM_MARGIN = 8
_PLOT_ROW_SPACING = 0
_PLOT_TITLE_ROW_HEIGHT = 32


class FrequencyHzAxis(pg.AxisItem):
    """Log-frequency axis with plain Hz labels (e.g. 300, 400) like matplotlib."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setStyle(tickTextHeight=28, autoExpandTextSpace=True)
        self.setTickFont(_PLOT_FONT)

    def logTickStrings(self, values, scale, _spacing):
        strings = []
        for v in values:
            hz = 10 ** (float(v) * scale)
            if hz >= 1000:
                strings.append(f"{hz / 1000:.0f}k")
            else:
                strings.append(f"{hz:.1f}")
        return strings


def _spectrum_band_hz(center_hz: float) -> tuple[float, float]:
    """
    Frequency band for the spectrum view, symmetric in log-space around the
    reference so it appears at the horizontal center of the log-frequency axis.

    Core band is half to double the reference (harmonics). If f_ref/2 < 10 Hz,
    the low edge is clamped to 10 Hz and the high edge is adjusted so the
    geometric mean stays at f_ref. Then the span is extended by 20% in log space
    on each side (wider, more compressed trace) without shifting the center.
    """
    f_ref = float(max(center_hz, 1e-6))
    gm_sq = f_ref * f_ref

    if 0.5 * f_ref >= 10.0:
        f_lo_core = 0.5 * f_ref
        f_hi_core = 2.0 * f_ref
    else:
        f_lo_core = 10.0
        f_hi_core = max(f_lo_core * 1.01, gm_sq / f_lo_core)

    log_lo0 = float(np.log10(f_lo_core))
    log_hi0 = float(np.log10(f_hi_core))
    span_log = log_hi0 - log_lo0
    pad_log = 0.2 * span_log
    log_lo = log_lo0 - pad_log
    log_hi = log_hi0 + pad_log
    return float(10**log_lo), float(10**log_hi)


def _apply_light_plot_style(plot: pg.PlotItem | pg.PlotWidget) -> None:
    """White plot background with dark axes, matching the original matplotlib look."""
    pi = plot.getPlotItem() if isinstance(plot, pg.PlotWidget) else plot
    pi.getViewBox().setBackgroundColor("w")
    for axis_name in ("bottom", "left"):
        axis = pi.getAxis(axis_name)
        axis.setPen(pg.mkPen("k"))
        axis.setTextPen(pg.mkPen("k"))


def _control_label(text: str) -> QLabel:
    """Right-aligned label placed immediately left of a control widget."""
    label = QLabel(text)
    label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    return label


def _set_panel_title(
    plot: pg.PlotItem, title: str, *, color: str = "k", bold: bool = False
) -> None:
    plot.setTitle(title, color=color, size=_PLOT_TITLE_PT, bold=bold)
    plot.layout.setRowFixedHeight(0, _PLOT_TITLE_ROW_HEIGHT)


def _apply_plot_border(plot: pg.PlotItem) -> None:
    """Thin border around the plot data area for contrast on white background."""
    plot.getViewBox().setBorder(pg.mkPen((100, 100, 100), width=1))


def _configure_plot_panel(
    plot: pg.PlotItem,
    *,
    title: str,
    y_label: str,
    x_label: str,
    y_axis_width: int = 78,
    left_margin: int = 4,
    top_margin: int = 8,
    bottom_margin: int = 14,
    grid_alpha: float = 0.45,
) -> None:
    """Shared layout styling so side-by-side panels align like the original UI."""
    _set_panel_title(plot, title)
    label_style = {"font-size": _PLOT_AXIS_LABEL_PT, "color": "#000"}
    plot.setLabel("left", y_label, **label_style)
    plot.setLabel("bottom", x_label, **label_style)
    plot.showGrid(x=True, y=True, alpha=grid_alpha)
    plot.showAxis("top", False)
    plot.showAxis("right", False)
    plot.hideButtons()
    plot.setMenuEnabled(False)
    left_axis = plot.getAxis("left")
    bottom_axis = plot.getAxis("bottom")
    left_axis.setWidth(y_axis_width)
    left_axis.setTickFont(_PLOT_FONT)
    bottom_axis.setTickFont(_PLOT_FONT)
    left_axis.setStyle(
        maxTickLevel=1,
        maxTextLevel=1,
        autoExpandTextSpace=False,
        tickTextWidth=40,
    )
    bottom_axis.setStyle(maxTickLevel=1, maxTextLevel=1, autoExpandTextSpace=False)
    plot.layout.setContentsMargins(left_margin, top_margin, 4, bottom_margin)
    plot.layout.setHorizontalSpacing(2)
    _apply_light_plot_style(plot)
    _apply_plot_border(plot)


def _spectrum_x_range_hz(fmin_hz: float, fmax_hz: float) -> tuple[float, float]:
    """Map Hz limits to PyQtGraph log-mode view coordinates (log10 Hz)."""
    return float(np.log10(fmin_hz)), float(np.log10(fmax_hz))


def _spectrum_plot_xy(
    frequencies: np.ndarray,
    psd_db: np.ndarray,
    log_xrange: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare spectrum data in log10(Hz) view coordinates (matplotlib semilog style)."""
    fmin_hz = 10 ** log_xrange[0]
    fmax_hz = 10 ** log_xrange[1]
    valid = (frequencies >= max(10.0, fmin_hz)) & (frequencies <= fmax_hz)
    x_log = np.log10(frequencies[valid].astype(np.float64))
    return x_log, psd_db[valid]


def _debug_disable_si_prefix(plot: pg.PlotWidget) -> None:
    """Matplotlib-style axis labels (no 0.5 s → '500m' SI scaling)."""
    pi = plot.getPlotItem()
    for axis_name in ("bottom", "left"):
        pi.getAxis(axis_name).enableAutoSIPrefix(False)


def _debug_style_plot_widget(plot: pg.PlotWidget, title: str) -> None:
    """Fonts, border, and axis styling for the debug plot window."""
    pi = plot.getPlotItem()
    _apply_light_plot_style(plot)
    pi.getViewBox().setBorder(pg.mkPen((70, 70, 70), width=1.5))
    _debug_disable_si_prefix(plot)
    pi.setTitle(title, color="k", size=_DEBUG_TITLE_PT)
    pi.showGrid(x=True, y=True, alpha=0.35)
    pi.hideButtons()
    pi.setMenuEnabled(False)
    pi.showAxis("top", False)
    pi.showAxis("right", False)
    for axis_name in ("bottom", "left"):
        axis = pi.getAxis(axis_name)
        axis.setTickFont(_DEBUG_FONT)
        axis.setTextPen(pg.mkPen("k"))
        axis.setPen(pg.mkPen("k"))
        axis.setStyle(autoExpandTextSpace=True)
    pi.layout.setContentsMargins(6, 8, 8, 10)


def _debug_clear_plot(plot: pg.PlotWidget) -> None:
    """Clear curves/markers and reset legend (PlotItem.clear keeps the legend)."""
    pi = plot.getPlotItem()
    pi.clear()
    if pi.legend is not None:
        pi.legend.clear()
    pi.getViewBox().setBorder(pg.mkPen((70, 70, 70), width=1.5))


def _debug_make_legend(
    plot: pg.PlotWidget, corner: str = "top-right"
) -> pg.LegendItem:
    """Legend anchored to a plot corner (avoids offset drift / overlap)."""
    legend = plot.getPlotItem().addLegend()
    legend.clear()
    legend.setLabelTextSize(_DEBUG_LEGEND_PT)
    legend.setFont(_DEBUG_FONT)
    legend.setBrush(pg.mkBrush(255, 255, 255, 230))
    legend.setPen(pg.mkPen("k", width=1))
    anchors = {
        "top-right": ((1, 0), (1, 0), (-12, 12)),
        "top-left": ((0, 0), (0, 0), (12, 12)),
    }
    item_pos, parent_pos, offset = anchors[corner]
    legend.anchor(itemPos=item_pos, parentPos=parent_pos, offset=offset)
    return legend


def _debug_text_item(text: str = "", *, html: str | None = None, **kwargs) -> pg.TextItem:
    if html is not None:
        item = pg.TextItem(html=html, **kwargs)
    else:
        item = pg.TextItem(text, color=_DEBUG_INFO_COLOR, **kwargs)
    item.setFont(_DEBUG_FONT)
    item.setZValue(100)
    return item


def _debug_place_in_view(
    plot: pg.PlotWidget,
    item: pg.TextItem,
    corner: str,
    *,
    margin_frac: float = 0.03,
) -> None:
    """Place a TextItem in a plot corner using data coordinates."""
    (x0, x1), (y0, y1) = plot.getPlotItem().getViewBox().viewRange()
    dx = margin_frac * max(x1 - x0, 1e-9)
    dy = margin_frac * max(y1 - y0, 1e-9)
    if corner == "upper_left":
        item.setPos(x0 + dx, y1 - dy)
        item.anchor = pg.Point(0, 0)
    elif corner == "upper_right":
        item.setPos(x1 - dx, y1 - dy)
        item.anchor = pg.Point(1, 0)
    elif corner == "center":
        item.setPos((x0 + x1) / 2, (y0 + y1) / 2)
        item.anchor = pg.Point(0.5, 0.5)
    elif corner == "middle_right":
        item.setPos(x1 - dx, (y0 + y1) / 2)
        item.anchor = pg.Point(1, 0.5)
    item.updateTextPos()
    plot.addItem(item)


def _debug_mag_db(h: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(h), 1e-15))


def _debug_acq_duration(debug_data: dict, fallback: float = 1.0) -> float:
    if "time" in debug_data:
        t = np.asarray(debug_data["time"], dtype=np.float64)
        if t.size > 1:
            return float(t[-1] - t[0])
        if t.size == 1:
            return float(t[0])
    return fallback


def _debug_y_limits(*arrays: np.ndarray, margin_frac: float = 0.08) -> tuple[float, float]:
    parts = [np.asarray(a, dtype=np.float64).ravel() for a in arrays if a is not None]
    if not parts:
        return -1.0, 1.0
    ys = np.concatenate(parts)
    y_min, y_max = float(np.min(ys)), float(np.max(ys))
    pad = margin_frac * (y_max - y_min) if y_max > y_min else max(0.1, 0.1 * abs(y_max))
    return y_min - pad, y_max + pad


def _debug_add_vline_legend(
    plot: pg.PlotWidget,
    legend: pg.LegendItem,
    x_pos: float,
    pen: QPen,
    label: str,
) -> None:
    """Vertical marker on plot; legend uses a PlotDataItem proxy (not InfiniteLine)."""
    plot.addItem(pg.InfiniteLine(pos=x_pos, angle=90, pen=pen))
    legend.addItem(pg.PlotDataItem(pen=pen), label)


def _debug_center_placeholder(
    plot: pg.PlotWidget,
    text: str,
    *,
    x_range: tuple[float, float],
    y_range: tuple[float, float] = (0.0, 1.0),
    align: str = "center",
) -> pg.TextItem:
    """Italic info message on empty debug panels (center or right-aligned)."""
    plot.setXRange(*x_range, padding=0)
    plot.setYRange(*y_range, padding=0)
    lines = text.replace("\n", "<br>")
    text_align = "right" if align == "right" else "center"
    corner = "middle_right" if align == "right" else "center"
    html = (
        f'<div style="color:{_DEBUG_INFO_COLOR}; font-style:italic; '
        f'text-align:{text_align};">{lines}</div>'
    )
    item = _debug_text_item(html=html)
    _debug_place_in_view(plot, item, corner)
    return item


class DebugPlotDialog(QDialog):
    """Separate window for timegrapher debug plots."""

    def __init__(self, parent: "SpectrumAnalyzerGUI"):
        super().__init__(parent)
        self.gui = parent
        self.setWindowTitle("Timegrapher Debug Plots")
        self.resize(1400, 900)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self.plot_filter = pg.PlotWidget()
        self.plot_filter.setLabel("bottom", "Frequency (Hz)")
        self.plot_filter.setLabel("left", "Magnitude (dB)")

        self.plot_filtered = pg.PlotWidget()
        self.plot_filtered.setLabel("bottom", "Time (s)")
        self.plot_filtered.setLabel("left", "Amplitude")

        self.plot_phase = pg.PlotWidget()
        self.plot_phase.setLabel("bottom", "Time (s)")
        self.plot_phase.setLabel("left", "Phase Residual (rad)")

        _debug_style_plot_widget(
            self.plot_filter, "Bandpass Filter Magnitude Response"
        )
        _debug_style_plot_widget(self.plot_filtered, "Bandpass Filtered Signal")
        _debug_style_plot_widget(self.plot_phase, "Phase Fit Residuals")

        for plot in (self.plot_filter, self.plot_filtered, self.plot_phase):
            layout.addWidget(plot, stretch=1)

    def closeEvent(self, event):
        self.gui.on_debug_dialog_closed()
        super().closeEvent(event)


class SpectrumAnalyzerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ForkCal Tuning Fork Watch Timegrapher")
        self.resize(1280, 960)
        self.setMinimumSize(880, 620)

        self.analyzer = None
        self.is_running = False

        self.frequencies = None
        self.psd_db = None

        self.time_data = list(range(100))
        self.deviation_data = [0.0] * 100
        self.deviation_data_raw = [0.0] * 100
        self.current_time = 0.0
        self.current_freq = None
        self.last_timegrapher_freq = None
        self.data_index = 0

        self.filter_lowcut = None
        self.filter_highcut = None
        self.filter_center = None

        self.debug_dialog: DebugPlotDialog | None = None
        self.debug_enabled = False
        self.acq_period = 0.5
        self.spectrum_log_xrange = _spectrum_x_range_hz(*_spectrum_band_hz(300))

        self._update_timer = QTimer(self)
        self._update_timer.setInterval(33)
        self._update_timer.timeout.connect(self.update_plot)

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(8, 4, 8, 4)
        root_layout.setSpacing(4)

        self.create_controls(root_layout)
        self.create_plot(root_layout)
        self.acq_period = self.parse_acquisition_period()

        self.update_device_list()

    def create_controls(self, parent_layout: QVBoxLayout):
        control_frame = QGroupBox("Controls")
        parent_layout.addWidget(control_frame)
        grid = QGridLayout(control_frame)
        grid.setHorizontalSpacing(4)
        grid.setVerticalSpacing(6)
        grid.setColumnStretch(1, 0)
        grid.setColumnStretch(2, 1)
        grid.setColumnStretch(3, 1)

        bold_font = QFont()
        bold_font.setBold(True)

        grid.addWidget(_control_label("Microphone:"), 0, 0)
        self.mic_combo = QComboBox()
        self.mic_combo.setMinimumWidth(280)
        self.mic_combo.currentTextChanged.connect(self.on_device_changed)
        grid.addWidget(self.mic_combo, 0, 1)

        grid.addWidget(_control_label("Sample Rate (Hz):"), 1, 0)
        self.sample_rate_combo = QComboBox()
        self.sample_rate_combo.addItems(
            ["8000", "16000", "22050", "44100", "48000", "96000", "192000"]
        )
        self.sample_rate_combo.setCurrentText("48000")
        grid.addWidget(self.sample_rate_combo, 1, 1)

        grid.addWidget(_control_label("Acquisition Period:"), 2, 0)
        self.acq_period_combo = QComboBox()
        self.acq_period_combo.addItems(
            ["250 ms", "500 ms", "1 s", "2 s", "5 s", "10 s", "20 s", "60 s"]
        )
        self.acq_period_combo.setCurrentText("500 ms")
        grid.addWidget(self.acq_period_combo, 2, 1)

        grid.addWidget(_control_label("Reference Frequency (Hz):"), 3, 0)
        self.ref_freq_combo = QComboBox()
        self.ref_freq_combo.addItems(["300", "360", "480", "600", "720", "960"])
        self.ref_freq_combo.setCurrentText("300")
        self.ref_freq_combo.currentTextChanged.connect(self.on_ref_freq_changed)
        grid.addWidget(self.ref_freq_combo, 3, 1)

        grid.addWidget(_control_label("Frequency Estimation:"), 4, 0)
        self.freq_method_combo = QComboBox()
        self.freq_method_combo.addItems(["Sine best fit", "Instantaneous phase fit"])
        self.freq_method_combo.setCurrentText("Instantaneous phase fit")
        grid.addWidget(self.freq_method_combo, 4, 1)

        grid.addWidget(_control_label("Resolution Bandwidth:"), 0, 2)
        self.rbw_label = QLabel("RBW: N/A")
        self.rbw_label.setFont(bold_font)
        self.rbw_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        grid.addWidget(self.rbw_label, 0, 3)

        grid.addWidget(_control_label("Estimated Frequency:"), 1, 2)
        self.freq_label = QLabel("Freq: N/A")
        self.freq_label.setFont(bold_font)
        self.freq_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        grid.addWidget(self.freq_label, 1, 3)

        grid.addWidget(_control_label("Timegrapher Stats:"), 2, 2)
        self.stats_label = QLabel("N/A")
        self.stats_label.setFont(bold_font)
        self.stats_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        grid.addWidget(self.stats_label, 2, 3)

        self.start_stop_btn = QPushButton("Start")
        self.start_stop_btn.setFixedWidth(120)
        self.start_stop_btn.clicked.connect(self.toggle_acquisition)
        grid.addWidget(self.start_stop_btn, 5, 1, 1, 2)

        self.debug_btn = QPushButton("Show Debug Plots")
        self.debug_btn.setFixedWidth(160)
        self.debug_btn.clicked.connect(self.toggle_debug_window)
        grid.addWidget(self.debug_btn, 5, 4)

        self.status_label = QLabel("Status: Stopped")
        self.status_label.setStyleSheet("color: red;")
        grid.addWidget(self.status_label, 6, 0, 1, 4)

    def create_plot(self, parent_layout: QVBoxLayout):
        self.plot_layout = pg.GraphicsLayoutWidget()
        self.plot_layout.setBackground("w")
        self.plot_layout.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        gl = self.plot_layout.ci.layout
        gl.setColumnStretchFactor(0, 1)
        gl.setRowStretchFactor(0, 1)
        gl.setRowStretchFactor(1, 1)
        gl.setRowSpacing(1, _PLOT_ROW_SPACING)
        gl.setContentsMargins(0, 0, 0, 0)

        self.plot_spectrum = self.plot_layout.addPlot(row=0, col=0)
        _configure_plot_panel(
            self.plot_spectrum,
            title="Spectrum Analyzer",
            y_label="Power Spectral Density (dB)",
            x_label="Frequency (Hz)",
            y_axis_width=66,
            left_margin=2,
            top_margin=_PLOT_TOP_MARGIN,
            bottom_margin=_PLOT_BOTTOM_MARGIN,
            grid_alpha=0.4,
        )
        self.plot_spectrum.setLogMode(x=True, y=False)
        freq_axis = FrequencyHzAxis(orientation="bottom")
        freq_axis.setLogMode(True)
        self.plot_spectrum.setAxisItems({"bottom": freq_axis})
        self.plot_spectrum.getViewBox().disableAutoRange()
        self.plot_spectrum.setXRange(*self.spectrum_log_xrange, padding=0)
        self.plot_spectrum.setYRange(-100, -20, padding=0)

        self.curve_spectrum = self.plot_spectrum.plot(
            pen=pg.mkPen((0, 0, 255), width=2.5)
        )
        self.curve_spectrum.setLogMode(False, False)
        self.curve_spectrum.setDownsampling(auto=False)
        self.curve_spectrum.setClipToView(False)
        self.curve_spectrum.setZValue(1)

        filter_pen = pg.mkPen((160, 160, 160), width=1, style=Qt.PenStyle.DashLine)
        self.vline_lowcut = pg.InfiniteLine(angle=90, pen=filter_pen)
        self.vline_highcut = pg.InfiniteLine(angle=90, pen=filter_pen)
        self.vline_center = pg.InfiniteLine(
            angle=90, pen=pg.mkPen((0, 140, 0), width=1, style=Qt.PenStyle.DashLine)
        )
        self.vline_estimate = pg.InfiniteLine(
            angle=90,
            pen=pg.mkPen((220, 0, 0), width=3.5),
        )
        self.vline_estimate.setZValue(20)
        self.vline_estimate.setVisible(False)
        for line in (
            self.vline_lowcut,
            self.vline_highcut,
            self.vline_center,
            self.vline_estimate,
        ):
            if line is not self.vline_estimate:
                line.setVisible(False)
            self.plot_spectrum.addItem(line)

        self.plot_time = self.plot_layout.addPlot(row=1, col=0)
        _configure_plot_panel(
            self.plot_time,
            title="Timegrapher",
            y_label="Deviation [spd]",
            x_label="Time [s]",
            y_axis_width=56,
            left_margin=4,
            top_margin=_PLOT_TOP_MARGIN_TIME,
            bottom_margin=_PLOT_BOTTOM_MARGIN,
            grid_alpha=0.4,
        )
        self.plot_time.getViewBox().disableAutoRange()
        self.plot_time.setXRange(0, 100, padding=0)
        self.timegrapher_y_range = 12
        self.plot_time.setYRange(
            -self.timegrapher_y_range, self.timegrapher_y_range, padding=0
        )

        self.scatter_time = pg.ScatterPlotItem(
            size=9,
            brush=pg.mkBrush(255, 0, 0, 255),
            pen=pg.mkPen((160, 0, 0), width=1),
        )
        self.plot_time.addItem(self.scatter_time)

        self.group_annotations: list[pg.TextItem] = []
        for _ in range(5):
            ann = pg.TextItem(
                anchor=(0.5, 1),
                fill=pg.mkBrush(255, 255, 0, 230),
                border=pg.mkPen((0, 0, 0), width=1.5),
                color=(0, 0, 0),
            )
            ann.setFont(_PLOT_GROUP_LABEL_FONT)
            ann.setVisible(False)
            self.plot_time.addItem(ann)
            self.group_annotations.append(ann)

        parent_layout.addWidget(self.plot_layout, stretch=1)

    def update_device_list(self):
        try:
            from analysis import get_audio_devices

            devices = get_audio_devices()
            self.mic_combo.blockSignals(True)
            self.mic_combo.clear()
            self.mic_combo.addItems(devices)
            self.mic_combo.blockSignals(False)
            if devices:
                self.mic_combo.setCurrentIndex(0)
                self.update_sample_rates()
        except Exception as e:
            self.mic_combo.clear()
            self.mic_combo.addItem(f"Error: {e}")

    def on_device_changed(self, _device_name: str):
        self.update_sample_rates()

    def on_ref_freq_changed(self, ref_freq: str):
        if self.analyzer:
            self.analyzer.set_reference_frequency(float(ref_freq))
        if ref_freq:
            self._set_spectrum_x_range_hz(*_spectrum_band_hz(float(ref_freq)))

    def update_sample_rates(self):
        device_name = self.mic_combo.currentText()
        if not device_name or device_name.startswith("Error:"):
            return

        try:
            from analysis import get_supported_sample_rates

            supported_rates = get_supported_sample_rates(device_name)
            if supported_rates:
                rate_strings = [str(rate) for rate in supported_rates]
                current_rate = self.sample_rate_combo.currentText()
                self.sample_rate_combo.clear()
                self.sample_rate_combo.addItems(rate_strings)
                if current_rate in rate_strings:
                    self.sample_rate_combo.setCurrentText(current_rate)
                elif "48000" in rate_strings:
                    self.sample_rate_combo.setCurrentText("48000")
                else:
                    self.sample_rate_combo.setCurrentIndex(0)
            else:
                self.sample_rate_combo.clear()
                self.sample_rate_combo.addItems(
                    ["8000", "16000", "22050", "44100", "48000", "96000", "192000"]
                )
        except Exception as e:
            print(f"Error updating sample rates: {e}")

    def _set_spectrum_x_range_hz(self, fmin_hz: float, fmax_hz: float) -> None:
        """Fix spectrum x-axis in log-mode view coordinates (not linear Hz)."""
        self.spectrum_log_xrange = _spectrum_x_range_hz(fmin_hz, fmax_hz)
        self.plot_spectrum.setXRange(*self.spectrum_log_xrange, padding=0)

    def parse_acquisition_period(self) -> float:
        period_str = self.acq_period_combo.currentText()
        value, unit = period_str.split()
        value_float = float(value)
        if unit == "us":
            return value_float * 1e-6
        if unit == "ms":
            return value_float * 1e-3
        if unit == "s":
            return value_float
        return 0.1

    def toggle_acquisition(self):
        if not self.is_running:
            self.start_acquisition()
        else:
            self.stop_acquisition()

    def _set_controls_enabled(self, enabled: bool):
        self.mic_combo.setEnabled(enabled)
        self.sample_rate_combo.setEnabled(enabled)
        self.acq_period_combo.setEnabled(enabled)
        self.ref_freq_combo.setEnabled(enabled)
        self.freq_method_combo.setEnabled(enabled)

    def start_acquisition(self):
        try:
            device_name = self.mic_combo.currentText()
            sample_rate = int(self.sample_rate_combo.currentText())
            acq_period = self.parse_acquisition_period()
            ref_freq = float(self.ref_freq_combo.currentText())

            freq_method_gui = self.freq_method_combo.currentText()
            freq_method = (
                "sine_fit" if freq_method_gui == "Sine best fit" else "phase_fit"
            )

            self.analyzer = AudioAnalyzer(
                device_name=device_name,
                sample_rate=sample_rate,
                acquisition_period=acq_period,
                num_averages=1,
                reference_freq=ref_freq,
                freq_estimation_method=freq_method,
            )
            self.analyzer.start()

            if self.debug_enabled:
                self.analyzer.set_debug_mode(True)

            self.time_data = [i * acq_period for i in range(100)]
            self.deviation_data = [0.0] * 100
            self.deviation_data_raw = [0.0] * 100
            self.current_time = 0.0
            self.last_timegrapher_freq = None
            self.data_index = 0
            self.acq_period = acq_period

            self.is_running = True
            self.start_stop_btn.setText("Stop")
            self.status_label.setText("Status: Running")
            self.status_label.setStyleSheet("color: green;")
            self._set_controls_enabled(False)
            self._set_spectrum_x_range_hz(*_spectrum_band_hz(ref_freq))
            self._update_timer.start()

        except Exception as e:
            self.status_label.setText(f"Error: {e}")
            self.status_label.setStyleSheet("color: red;")

    def stop_acquisition(self):
        try:
            self._update_timer.stop()
            if self.analyzer:
                self.analyzer.stop()
                self.analyzer = None

            self.is_running = False
            self.start_stop_btn.setText("Start")
            self.status_label.setText("Status: Stopped")
            self.status_label.setStyleSheet("color: red;")
            self.vline_lowcut.setVisible(False)
            self.vline_highcut.setVisible(False)
            self.vline_center.setVisible(False)
            marker_hz = self.current_freq or float(self.ref_freq_combo.currentText())
            self.vline_estimate.setPos(np.log10(marker_hz))
            self.vline_estimate.setVisible(True)
            self._set_controls_enabled(True)

        except Exception as e:
            self.status_label.setText(f"Error stopping: {e}")

    def update_plot(self, N_moving_avg: int = 10):
        if not self.is_running:
            return

        try:
            frequencies, psd_db = self.analyzer.get_spectrum()
            if frequencies is not None and psd_db is not None:
                x_log, y_db = _spectrum_plot_xy(
                    frequencies, psd_db, self.spectrum_log_xrange
                )
                self.curve_spectrum.setLogMode(False, False)
                self.curve_spectrum.setData(x_log, y_db)
                self.frequencies = frequencies
                self.psd_db = psd_db
                self.plot_spectrum.setXRange(*self.spectrum_log_xrange, padding=0)

                rbw = self.analyzer.compute_rbw()
                if rbw is not None:
                    if rbw >= 1000:
                        self.rbw_label.setText(f"RBW: {rbw / 1000:.2f} kHz")
                    else:
                        self.rbw_label.setText(f"RBW: {rbw:.2f} Hz")

            freq_estimate, deviation_spd = self.analyzer.get_timegrapher_data()
            if freq_estimate is not None and freq_estimate > 0:
                self.vline_estimate.setPos(np.log10(freq_estimate))
                self.vline_estimate.setVisible(True)
                self.vline_center.setVisible(False)
            else:
                self.vline_estimate.setVisible(False)

            if freq_estimate is not None and deviation_spd is not None:
                if freq_estimate != self.last_timegrapher_freq:
                    self.time_data[self.data_index] = self.current_time
                    self.deviation_data_raw[self.data_index] = deviation_spd

                    window_values = []
                    for j in range(N_moving_avg):
                        idx = (self.data_index - j) % 100
                        if self.deviation_data_raw[idx] != 0.0:
                            window_values.append(self.deviation_data_raw[idx])
                    self.deviation_data[self.data_index] = (
                        float(np.mean(window_values)) if window_values else 0.0
                    )

                    self.data_index = (self.data_index + 1) % 100

                    raw_data_array = np.array(self.deviation_data_raw)
                    valid_mask = raw_data_array != 0.0
                    time_data_array = np.array(self.time_data)
                    deviation_data_array = np.array(self.deviation_data)

                    valid_times = time_data_array[valid_mask]
                    valid_deviations = deviation_data_array[valid_mask]
                    self.scatter_time.setData(valid_times, valid_deviations)

                    for group_idx in range(5):
                        start_idx = group_idx * 20
                        end_idx = start_idx + 20
                        raw_data = np.array(self.deviation_data_raw[start_idx:end_idx])
                        if np.any(raw_data == 0.0):
                            self.group_annotations[group_idx].setVisible(False)
                        else:
                            filtered_data = self.deviation_data[start_idx:end_idx]
                            group_avg = np.mean(filtered_data)
                            center_time_idx = start_idx + 10
                            x_pos = self.time_data[center_time_idx]
                            y_lo, y_hi = self.plot_time.viewRange()[1]
                            y_span = y_hi - y_lo
                            # Anchor is bottom-center of text; keep labels inside plot (not clipped at top)
                            y_pos = y_hi - max(2.2, 0.22 * y_span)
                            self.group_annotations[group_idx].setText(
                                f"{group_avg:.1f}"
                            )
                            self.group_annotations[group_idx].setPos(x_pos, y_pos)
                            self.group_annotations[group_idx].setVisible(True)

                    if len(self.time_data) > 0:
                        time_range = max(self.time_data) - min(self.time_data)
                        if time_range > 0:
                            self.plot_time.setXRange(
                                min(self.time_data), max(self.time_data)
                            )

                    if len(self.deviation_data) > 0:
                        y_max_abs = max(
                            abs(min(self.deviation_data)),
                            abs(max(self.deviation_data)),
                        )
                        ranges = [12, 25, 50, 100, 200]
                        y_range = self.timegrapher_y_range
                        for r in ranges:
                            if y_max_abs <= r:
                                y_range = r
                                break
                        self.plot_time.setYRange(-y_range, y_range, padding=0)

                    self.freq_label.setText(f"Freq: {freq_estimate:.6f} Hz")
                    self.current_freq = freq_estimate
                    self.last_timegrapher_freq = freq_estimate
                    self.current_time += self.parse_acquisition_period()

                    raw_data = np.array(self.deviation_data_raw)
                    valid_data = raw_data[raw_data != 0.0]
                    if len(valid_data) > 0:
                        mean_spd = np.mean(valid_data)
                        std_spd = (
                            np.std(valid_data, ddof=1) if len(valid_data) > 1 else 0.0
                        )
                        self.stats_label.setText(f"{mean_spd:.2f} ± {std_spd:.2f} spd")
                    else:
                        self.stats_label.setText("N/A")

            if self.analyzer:
                lowcut, highcut, center = self.analyzer.get_filter_params()
                if center is not None:
                    self._set_spectrum_x_range_hz(*_spectrum_band_hz(center))
                    self.vline_lowcut.setPos(np.log10(lowcut))
                    self.vline_highcut.setPos(np.log10(highcut))
                    self.vline_center.setPos(np.log10(center))
                    show_filter_lines = self.vline_estimate.isVisible() is False
                    self.vline_lowcut.setVisible(show_filter_lines)
                    self.vline_highcut.setVisible(show_filter_lines)
                    self.vline_center.setVisible(show_filter_lines)

            if self.analyzer:
                signal_quality_good = self.analyzer.get_signal_quality()
                if signal_quality_good:
                    _set_panel_title(self.plot_spectrum, "Spectrum Analyzer")
                    _set_panel_title(self.plot_time, "Timegrapher")
                    self.status_label.setText("Status: Running")
                    self.status_label.setStyleSheet("color: green;")
                else:
                    _set_panel_title(
                        self.plot_spectrum, "Spectrum Analyzer", color="r", bold=True
                    )
                    _set_panel_title(
                        self.plot_time, "Timegrapher", color="r", bold=True
                    )
                    self.status_label.setText("Status: Running; Signal not found!")
                    self.status_label.setStyleSheet("color: red; font-weight: bold;")

            if self.analyzer and self.debug_enabled and self.debug_dialog is not None:
                debug_data = self.analyzer.get_debug_plot_data()
                if debug_data is not None:
                    self.update_debug_plots(debug_data)

        except Exception as e:
            print(f"Error updating plot: {e}")

    def toggle_debug_window(self):
        if self.debug_dialog is None or not self.debug_dialog.isVisible():
            self.debug_dialog = DebugPlotDialog(self)
            self.debug_dialog.show()
            self.debug_enabled = True
            self.debug_btn.setText("Hide Debug Plots")
            if self.analyzer:
                self.analyzer.set_debug_mode(True)
        else:
            self.debug_dialog.close()

    def on_debug_dialog_closed(self):
        self.debug_dialog = None
        self.debug_enabled = False
        self.debug_btn.setText("Show Debug Plots")
        if self.analyzer:
            self.analyzer.set_debug_mode(False)

    def update_debug_plots(self, debug_data: dict):
        if self.debug_dialog is None:
            return

        try:
            pf = self.debug_dialog.plot_filter
            pfiltered = self.debug_dialog.plot_filtered
            pphase = self.debug_dialog.plot_phase

            for plot in (pf, pfiltered, pphase):
                _debug_clear_plot(plot)

            t_end = _debug_acq_duration(
                debug_data, getattr(self, "acq_period", 1.0)
            )
            time_x = (0.0, t_end)

            if "filter_freq" in debug_data and "filter_h" in debug_data:
                w = np.asarray(debug_data["filter_freq"], dtype=np.float64)
                h = debug_data["filter_h"]
                lowcut = debug_data.get("lowcut", 0)
                highcut = debug_data.get("highcut", 0)
                center_freq = debug_data.get("center_freq", 0)

                pf.plot(w, _debug_mag_db(h), pen=_DEBUG_PEN_FILTER)
                pf.setXRange(0.0, float(np.max(w)), padding=0)
                pf.setYRange(-150.0, 0.0, padding=0)

                legend = _debug_make_legend(pf, "top-right")
                for x_pos, pen, label in (
                    (lowcut, _DEBUG_PEN_VLINE_LO, f"Low cutoff: {lowcut:.2f} Hz"),
                    (highcut, _DEBUG_PEN_VLINE_HI, f"High cutoff: {highcut:.2f} Hz"),
                    (
                        center_freq,
                        _DEBUG_PEN_VLINE_CTR,
                        f"Center: {center_freq:.2f} Hz",
                    ),
                ):
                    _debug_add_vline_legend(pf, legend, x_pos, pen, label)

                pf.setTitle(
                    "Bandpass Filter Magnitude Response",
                    color="k",
                    size=_DEBUG_TITLE_PT,
                )

            fit_method = debug_data.get("fit_method", "sine_fit")
            if (
                fit_method == "sine_fit"
                and "time" in debug_data
                and "x_filtered" in debug_data
            ):
                t = np.asarray(debug_data["time"], dtype=np.float64)
                x_filtered = np.asarray(debug_data["x_filtered"], dtype=np.float64)
                _debug_make_legend(pfiltered, "top-left")

                pfiltered.plot(
                    t,
                    x_filtered,
                    pen=_DEBUG_PEN_SIGNAL,
                    name="Full Signal",
                )

                if "t_cropped_fit" in debug_data and "fitted_signal" in debug_data:
                    t_crop = np.asarray(
                        debug_data["t_cropped_fit"], dtype=np.float64
                    )
                    fitted_signal = np.asarray(
                        debug_data["fitted_signal"], dtype=np.float64
                    )
                    estimated_freq = debug_data.get("estimated_freq", 0)
                    deviation_spd = debug_data.get("deviation_spd", 0)
                    A_fit = debug_data.get("A_fit", 0)
                    f_fit = debug_data.get("f_fit", 0)
                    phi_fit = debug_data.get("phi_fit", 0)

                    pfiltered.plot(
                        t_crop,
                        fitted_signal,
                        pen=_DEBUG_PEN_FIT,
                        name="Fitted Signal",
                    )

                    y_lo, y_hi = _debug_y_limits(x_filtered, fitted_signal)
                    pfiltered.setXRange(0.0, float(t[-1]), padding=0.02)
                    pfiltered.setYRange(y_lo, y_hi, padding=0)

                    param_text = (
                        f"Sine Fit Parameters:\n"
                        f"A = {A_fit:.6f}\n"
                        f"f = {f_fit:.6f} Hz\n"
                        f"φ = {phi_fit:.6f} rad ({np.degrees(phi_fit):.2f}°)\n"
                        f"\nEstimated Frequency = {estimated_freq:.6f} Hz\n"
                        f"Deviation = {deviation_spd:.3f} s/day"
                    )
                    text_item = _debug_text_item(
                        param_text,
                        fill=pg.mkBrush(245, 222, 179, 200),
                        border=pg.mkPen("k"),
                    )
                    _debug_place_in_view(pfiltered, text_item, "upper_right")
                else:
                    y_lo, y_hi = _debug_y_limits(x_filtered)
                    pfiltered.setXRange(0.0, float(t[-1]), padding=0.02)
                    pfiltered.setYRange(y_lo, y_hi, padding=0)

                pfiltered.setTitle(
                    "Bandpass Filtered Signal with Sine Fit",
                    color="k",
                    size=_DEBUG_TITLE_PT,
                )

            elif fit_method == "phase_fit":
                _debug_center_placeholder(
                    pfiltered,
                    "Signal reconstruction not used\n"
                    "for Instantaneous Phase Fit method\n\n"
                    "See Phase Residuals plot below",
                    x_range=time_x,
                    y_range=(0.0, 1.0),
                )
                pfiltered.setTitle(
                    "Bandpass Filtered Signal", color="k", size=_DEBUG_TITLE_PT
                )

            if (
                fit_method == "phase_fit"
                and "phase_residuals" in debug_data
                and "phase_residuals_time" in debug_data
            ):
                t_residuals = np.asarray(
                    debug_data["phase_residuals_time"], dtype=np.float64
                )
                phase_residuals = np.asarray(
                    debug_data["phase_residuals"], dtype=np.float64
                )
                _debug_make_legend(pphase, "top-left")
                pphase.plot(
                    t_residuals,
                    phase_residuals,
                    pen=_DEBUG_PEN_PHASE,
                    name="Phase Residuals (cropped)",
                )
                pphase.addLine(y=0, pen=_DEBUG_PEN_ZERO)

                t_pad = 0.03 * max(float(t_residuals[-1] - t_residuals[0]), 1e-6)
                y_lo, y_hi = _debug_y_limits(phase_residuals, margin_frac=0.12)
                pphase.setXRange(
                    float(t_residuals[0]) - t_pad,
                    float(t_residuals[-1]) + t_pad,
                    padding=0,
                )
                pphase.setYRange(y_lo, y_hi, padding=0)

                rms_residual = debug_data.get(
                    "rms_phase_residual",
                    float(np.sqrt(np.mean(phase_residuals**2))),
                )
                stats_text = (
                    f"Phase Fit Quality:\n"
                    f"RMS Residual = {rms_residual:.6f} rad\n"
                    f"RMS Residual = {np.degrees(rms_residual):.4f}°\n"
                    f"(Edge 5% excluded)"
                )
                stats_item = _debug_text_item(
                    stats_text,
                    fill=pg.mkBrush(173, 216, 230, 200),
                    border=pg.mkPen("k"),
                )
                _debug_place_in_view(pphase, stats_item, "upper_right")
                pphase.setTitle(
                    "Phase Fit Residuals (Unwrapped Phase - Linear Fit, Center 90%)",
                    color="k",
                    size=_DEBUG_TITLE_PT,
                )
            else:
                _debug_center_placeholder(
                    pphase,
                    "Phase residuals only available\n"
                    "for Instantaneous Phase Fit method",
                    x_range=time_x,
                    y_range=(0.0, 1.0),
                )
                pphase.setTitle(
                    "Phase Fit Residuals", color="k", size=_DEBUG_TITLE_PT
                )

        except Exception as e:
            print(f"Error updating debug plots: {e}")

    def closeEvent(self, event):
        if self.is_running:
            self.stop_acquisition()
        if self.debug_dialog is not None:
            self.debug_dialog.close()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = SpectrumAnalyzerGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
