#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          PREDICTIVE MAINTENANCE NODE - GROUND CONTROL STATION                ║
║                     Defense Industry Standard v1.0                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Real-time monitoring interface for ESP32-based anomaly detection system     ║
║  Author: Predictive Maintenance Project                                      ║
║  Date: February 2026                                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import re
import time
from datetime import datetime
from collections import deque
from typing import Optional, Tuple

# Third-party imports
try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QComboBox, QPushButton, QGroupBox, QGridLayout,
        QPlainTextEdit, QFrame, QSplitter, QStatusBar, QMessageBox
    )
    from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
    from PyQt5.QtGui import QFont, QPalette, QColor, QPainter, QBrush
except ImportError:
    print("ERROR: PyQt5 not installed. Run: pip install PyQt5")
    sys.exit(1)

try:
    import pyqtgraph as pg
except ImportError:
    print("ERROR: pyqtgraph not installed. Run: pip install pyqtgraph")
    sys.exit(1)

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    print("ERROR: pyserial not installed. Run: pip install pyserial")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Application configuration constants."""
    WINDOW_TITLE = "Predictive Maintenance - Ground Control Station"
    WINDOW_SIZE = (1400, 1050)
    
    # Serial
    BAUD_RATE = 115200
    SERIAL_TIMEOUT = 0.1
    
    # Plot
    PLOT_HISTORY_SIZE = 200  # Number of data points to display
    PLOT_UPDATE_RATE_MS = 100  # Update plot every 100ms
    
    # Threshold (should match ESP32 firmware)
    DEFAULT_ANOMALY_THRESHOLD = 0.035
    
    # Colors (Defense/Military aesthetic)
    COLOR_BG_DARK = "#0a0a0f"
    COLOR_BG_PANEL = "#12141a"
    COLOR_ACCENT = "#00ff88"
    COLOR_WARNING = "#ff4444"
    COLOR_TEXT = "#e0e0e0"
    COLOR_TEXT_DIM = "#808080"
    COLOR_GRID = "#1a1a2e"
    COLOR_PLOT_LINE = "#00ccff"
    COLOR_THRESHOLD = "#ff3333"
    

# =============================================================================
# LOG PARSER
# =============================================================================

class LogParser:
    """Parse ESP32 serial output for MSE/RMS values and status."""
    
    # New format with severity and band:
    # "Normal: MSE=0.0487 (th=0.0587) | RMS=0.181g (th=0.482g) | Temp=25.5C | SEV=NORMAL | BAND=LOW"
    # "WARNING [AUDIO] MSE=0.0963 (th=0.0588) | RMS=0.184g (th=0.481g) | Temp=25.5C | SEV=WARNING | BAND=HIGH"
    PATTERN_NORMAL = re.compile(
        r'Normal.*MSE=([0-9.]+).*th=([0-9.]+).*RMS=([0-9.]+)g.*th=([0-9.]+)g.*Temp=([0-9.-]+)C',
        re.IGNORECASE
    )
    PATTERN_ANOMALY = re.compile(
        r'(WATCH|WARNING|CRITICAL)\s*\[(\w+)\].*MSE=([0-9.]+).*th=([0-9.]+).*RMS=([0-9.]+)g.*th=([0-9.]+)g.*Temp=([0-9.-]+)C',
        re.IGNORECASE
    )
    # Legacy format for backwards compatibility
    PATTERN_ANOMALY_LEGACY = re.compile(
        r'ANOMALY\s*\[(\w+)\].*MSE=([0-9.]+).*th=([0-9.]+).*RMS=([0-9.]+)g.*th=([0-9.]+)g.*Temp=([0-9.-]+)C',
        re.IGNORECASE
    )
    PATTERN_SEV_BAND = re.compile(
        r'SEV=(\w+).*BAND=(\w+)',
        re.IGNORECASE
    )
    PATTERN_CALIBRATION = re.compile(
        r'CALIBRATING\s+(\d+)/(\d+).*MSE=([0-9.]+).*RMS=([0-9.]+)g',
        re.IGNORECASE
    )
    PATTERN_STATS = re.compile(
        r'Stats:\s*(\d+)\s*detections.*?(\d+)\s*anomalies.*?([0-9.]+)%',
        re.IGNORECASE
    )
    
    @staticmethod
    def parse_line(line: str) -> dict:
        """
        Parse a single log line and extract relevant data.
        
        Returns:
            dict with keys: mse, mse_threshold, rms, rms_threshold, 
                           is_anomaly, anomaly_source, severity, dominant_band,
                           calibrating
        """
        result = {
            'mse': None,
            'mse_threshold': None,
            'rms': None,
            'rms_threshold': None,
            'temp': None,
            'is_anomaly': None,
            'anomaly_source': None,
            'severity': 'NORMAL',
            'dominant_band': None,
            'calibrating': False,
            'cal_progress': None,
            'raw': line.strip()
        }
        
        # Extract SEV= and BAND= if present (works for both normal and anomaly lines)
        sev_match = LogParser.PATTERN_SEV_BAND.search(line)
        if sev_match:
            result['severity'] = sev_match.group(1).upper()
            result['dominant_band'] = sev_match.group(2).upper()
        
        # Check for anomaly first (new format: WATCH/WARNING/CRITICAL [SOURCE])
        match = LogParser.PATTERN_ANOMALY.search(line)
        if match:
            result['severity'] = match.group(1).upper()
            result['anomaly_source'] = match.group(2)  # AUDIO, VIBRATION, BOTH
            result['mse'] = float(match.group(3))
            result['mse_threshold'] = float(match.group(4))
            result['rms'] = float(match.group(5))
            result['rms_threshold'] = float(match.group(6))
            result['temp'] = float(match.group(7))
            result['is_anomaly'] = True
            return result
        
        # Check legacy ANOMALY format for backwards compatibility
        match = LogParser.PATTERN_ANOMALY_LEGACY.search(line)
        if match:
            result['anomaly_source'] = match.group(1)
            result['mse'] = float(match.group(2))
            result['mse_threshold'] = float(match.group(3))
            result['rms'] = float(match.group(4))
            result['rms_threshold'] = float(match.group(5))
            result['temp'] = float(match.group(6))
            result['is_anomaly'] = True
            result['severity'] = 'WARNING'  # Default for legacy
            return result
            
        # Check for normal status
        match = LogParser.PATTERN_NORMAL.search(line)
        if match:
            result['mse'] = float(match.group(1))
            result['mse_threshold'] = float(match.group(2))
            result['rms'] = float(match.group(3))
            result['rms_threshold'] = float(match.group(4))
            result['temp'] = float(match.group(5))
            result['is_anomaly'] = False
            return result
        
        # Check for calibration
        match = LogParser.PATTERN_CALIBRATION.search(line)
        if match:
            result['calibrating'] = True
            result['cal_progress'] = f"{match.group(1)}/{match.group(2)}"
            result['mse'] = float(match.group(3))
            result['rms'] = float(match.group(4))
            return result
            
        return result


# =============================================================================
# SERIAL READER THREAD
# =============================================================================

class SerialReaderThread(QThread):
    """Background thread for reading serial data."""
    
    data_received = pyqtSignal(str)
    connection_error = pyqtSignal(str)
    
    def __init__(self, port: str, baud_rate: int = Config.BAUD_RATE):
        super().__init__()
        self.port = port
        self.baud_rate = baud_rate
        self.serial_conn: Optional[serial.Serial] = None
        self.running = False
        
    def run(self):
        """Main thread loop - read serial data continuously."""
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=Config.SERIAL_TIMEOUT
            )
            self.running = True
            
            while self.running:
                if self.serial_conn.in_waiting > 0:
                    try:
                        line = self.serial_conn.readline().decode('utf-8', errors='replace')
                        if line:
                            self.data_received.emit(line)
                    except Exception as e:
                        self.connection_error.emit(f"Read error: {str(e)}")
                else:
                    time.sleep(0.01)  # Small delay to prevent busy-waiting
                    
        except serial.SerialException as e:
            self.connection_error.emit(f"Connection failed: {str(e)}")
        finally:
            if self.serial_conn and self.serial_conn.is_open:
                self.serial_conn.close()
                
    def stop(self):
        """Stop the thread gracefully."""
        self.running = False
        self.wait(1000)  # Wait up to 1 second


# =============================================================================
# TRAFFIC LIGHT WIDGET
# =============================================================================

class TrafficLightWidget(QWidget):
    """4-state traffic light: NORMAL(green), WATCH(yellow), WARNING(orange), CRITICAL(red)."""
    
    # Color scheme for each severity level
    SEVERITY_COLORS = {
        'NORMAL':   {'active': '#00ff88', 'dim': '#003300', 'pos': 3},
        'WATCH':    {'active': '#ffdd00', 'dim': '#333300', 'pos': 2},
        'WARNING':  {'active': '#ff8800', 'dim': '#332200', 'pos': 1},
        'CRITICAL': {'active': '#ff0000', 'dim': '#330000', 'pos': 0},
    }
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.status = "IDLE"  # IDLE, NORMAL, WATCH, WARNING, CRITICAL
        self.setMinimumSize(120, 320)
        self.setMaximumWidth(150)
        
    def set_status(self, status: str):
        """Update the traffic light status."""
        self.status = status.upper()
        self.update()
        
    def paintEvent(self, event):
        """Custom paint for 4-light traffic light."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w, h = self.width(), self.height()
        
        # Housing
        painter.setBrush(QBrush(QColor("#1a1a2e")))
        painter.setPen(QColor("#333366"))
        painter.drawRoundedRect(10, 10, w - 20, h - 20, 15, 15)
        
        # 4 light positions: CRITICAL(top), WARNING, WATCH, NORMAL(bottom)
        light_radius = min(w - 60, 35)
        center_x = w // 2
        lights = ['CRITICAL', 'WARNING', 'WATCH', 'NORMAL']
        spacing = (h - 80) / (len(lights))
        
        for i, level in enumerate(lights):
            light_y = int(50 + spacing * i + spacing / 2)
            colors = self.SEVERITY_COLORS[level]
            
            if self.status == level:
                # Active glow
                painter.setBrush(QBrush(QColor(colors['active'])))
                for g in range(3):
                    painter.setOpacity(0.3 - g * 0.1)
                    painter.drawEllipse(
                        center_x - light_radius - g*4,
                        light_y - light_radius - g*4,
                        (light_radius + g*4) * 2,
                        (light_radius + g*4) * 2
                    )
                painter.setOpacity(1.0)
            else:
                painter.setBrush(QBrush(QColor(colors['dim'])))
            
            painter.drawEllipse(
                center_x - light_radius, light_y - light_radius,
                light_radius * 2, light_radius * 2
            )
        
        # Status text
        painter.setPen(QColor("#ffffff"))
        painter.setFont(QFont("Consolas", 10, QFont.Bold))
        painter.drawText(0, h - 25, w, 20, Qt.AlignCenter, self.status)


# =============================================================================
# MAIN WINDOW
# =============================================================================

class GroundControlStation(QMainWindow):
    """Main application window - Ground Control Station."""
    
    def __init__(self):
        super().__init__()
        
        # Data storage - dual sensor
        self.mse_history = deque(maxlen=Config.PLOT_HISTORY_SIZE)
        self.rms_history = deque(maxlen=Config.PLOT_HISTORY_SIZE)
        self.time_history = deque(maxlen=Config.PLOT_HISTORY_SIZE)
        self.start_time = time.time()
        self.mse_threshold = Config.DEFAULT_ANOMALY_THRESHOLD
        self.rms_threshold = 0.5  # Default RMS threshold in g
        
        # Serial connection
        self.serial_thread: Optional[SerialReaderThread] = None
        self.is_connected = False
        
        # Statistics
        self.total_samples = 0
        self.anomaly_count = 0
        self.last_mse = 0.0
        self.last_rms = 0.0
        self.is_calibrating = False
        
        # Setup UI
        self._setup_window()
        self._setup_styles()
        self._create_ui()
        self._setup_plot()
        self._setup_timers()
        
        # Populate COM ports
        self._refresh_ports()
        
    def _setup_window(self):
        """Configure main window properties."""
        self.setWindowTitle(Config.WINDOW_TITLE)
        self.resize(*Config.WINDOW_SIZE)
        self.setMinimumSize(1000, 700)
        
    def _setup_styles(self):
        """Apply dark theme styling."""
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background-color: {Config.COLOR_BG_DARK};
                color: {Config.COLOR_TEXT};
                font-family: 'Segoe UI', 'Consolas', monospace;
            }}
            QGroupBox {{
                background-color: {Config.COLOR_BG_PANEL};
                border: 1px solid #333366;
                border-radius: 8px;
                margin-top: 12px;
                padding: 10px;
                font-weight: bold;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px;
                color: {Config.COLOR_ACCENT};
            }}
            QPushButton {{
                background-color: #1a1a3e;
                border: 1px solid #333366;
                border-radius: 5px;
                padding: 8px 20px;
                color: {Config.COLOR_TEXT};
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #2a2a5e;
                border-color: {Config.COLOR_ACCENT};
            }}
            QPushButton:pressed {{
                background-color: #0a0a2e;
            }}
            QPushButton:disabled {{
                background-color: #0a0a1e;
                color: #555555;
            }}
            QComboBox {{
                background-color: #1a1a3e;
                border: 1px solid #333366;
                border-radius: 5px;
                padding: 5px 10px;
                min-width: 150px;
            }}
            QComboBox::drop-down {{
                border: none;
                width: 30px;
            }}
            QComboBox QAbstractItemView {{
                background-color: #1a1a3e;
                selection-background-color: #2a2a5e;
            }}
            QTextEdit {{
                background-color: #0a0a15;
                border: 1px solid #222244;
                border-radius: 5px;
                font-family: 'Consolas', monospace;
                font-size: 11px;
            }}
            QLabel {{
                color: {Config.COLOR_TEXT};
            }}
            QStatusBar {{
                background-color: {Config.COLOR_BG_PANEL};
                border-top: 1px solid #333366;
            }}
        """)
        
    def _create_ui(self):
        """Create the main user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Header
        header = self._create_header()
        main_layout.addWidget(header)
        
        # Main content splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Left side - Plot and metrics
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # Plot
        plot_group = self._create_plot_group()
        left_layout.addWidget(plot_group, stretch=5)
        
        # Metrics
        metrics_group = self._create_metrics_group()
        left_layout.addWidget(metrics_group, stretch=1)
        
        splitter.addWidget(left_widget)
        
        # Right side - Status and log
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # Traffic light
        status_group = self._create_status_group()
        right_layout.addWidget(status_group)
        
        # Log console
        log_group = self._create_log_group()
        right_layout.addWidget(log_group)
        
        splitter.addWidget(right_widget)
        splitter.setSizes([900, 400])
        
        main_layout.addWidget(splitter)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("◉ Disconnected - Select COM port and click Connect")
        
    def _create_header(self) -> QWidget:
        """Create the header with connection controls."""
        header = QFrame()
        header.setStyleSheet(f"""
            QFrame {{
                background-color: {Config.COLOR_BG_PANEL};
                border: 1px solid #333366;
                border-radius: 8px;
                padding: 10px;
            }}
        """)
        layout = QHBoxLayout(header)
        
        # Title
        title = QLabel("⬡ GROUND CONTROL STATION")
        title.setFont(QFont("Consolas", 16, QFont.Bold))
        title.setStyleSheet(f"color: {Config.COLOR_ACCENT};")
        layout.addWidget(title)
        
        layout.addStretch()
        
        # COM Port selection
        port_label = QLabel("COM Port:")
        port_label.setFont(QFont("Consolas", 10))
        layout.addWidget(port_label)
        
        self.port_combo = QComboBox()
        self.port_combo.setMinimumWidth(180)
        layout.addWidget(self.port_combo)
        
        # Refresh button
        refresh_btn = QPushButton("⟳")
        refresh_btn.setFixedWidth(40)
        refresh_btn.clicked.connect(self._refresh_ports)
        refresh_btn.setToolTip("Refresh COM ports")
        layout.addWidget(refresh_btn)
        
        # Connect button
        self.connect_btn = QPushButton("CONNECT")
        self.connect_btn.setMinimumWidth(120)
        self.connect_btn.clicked.connect(self._toggle_connection)
        layout.addWidget(self.connect_btn)
        
        # Clear button
        clear_btn = QPushButton("CLEAR")
        clear_btn.clicked.connect(self._clear_data)
        layout.addWidget(clear_btn)
        
        return header
        
    def _create_plot_group(self) -> QGroupBox:
        """Create the dual-plot display group."""
        group = QGroupBox("◈ LIVE TELEMETRY")
        layout = QVBoxLayout(group)
        
        # Configure pyqtgraph
        pg.setConfigOptions(antialias=True, background=Config.COLOR_BG_DARK)
        
        # --- Audio MSE Graph (Top) ---
        self.mse_plot = pg.PlotWidget(title="Audio Reconstruction Error (MSE)")
        self.mse_plot.setBackground(Config.COLOR_BG_DARK)
        self.mse_plot.showGrid(x=True, y=True, alpha=0.3)
        self.mse_plot.setLabel('left', 'MSE', color=Config.COLOR_TEXT)
        self.mse_plot.setLabel('bottom', 'Time (s)', color=Config.COLOR_TEXT)
        self.mse_plot.getAxis('left').setTextPen(Config.COLOR_TEXT)
        self.mse_plot.getAxis('bottom').setTextPen(Config.COLOR_TEXT)
        self.mse_plot.setMinimumHeight(220)
        layout.addWidget(self.mse_plot)
        
        # --- Vibration RMS Graph (Bottom) ---
        self.rms_plot = pg.PlotWidget(title="Vibration RMS (g)")
        self.rms_plot.setBackground(Config.COLOR_BG_DARK)
        self.rms_plot.showGrid(x=True, y=True, alpha=0.3)
        self.rms_plot.setLabel('left', 'RMS (g)', color=Config.COLOR_TEXT)
        self.rms_plot.setLabel('bottom', 'Time (s)', color=Config.COLOR_TEXT)
        self.rms_plot.getAxis('left').setTextPen(Config.COLOR_TEXT)
        self.rms_plot.getAxis('bottom').setTextPen(Config.COLOR_TEXT)
        self.rms_plot.setMinimumHeight(220)
        layout.addWidget(self.rms_plot)
        
        # Link X-axes so both graphs scroll together
        self.rms_plot.setXLink(self.mse_plot)
        
        return group
        
    def _setup_plot(self):
        """Setup plot curves and threshold lines for both graphs."""
        # --- Audio MSE Graph ---
        self.mse_curve = self.mse_plot.plot(
            pen=pg.mkPen(color=Config.COLOR_PLOT_LINE, width=2),
            name="Audio MSE"
        )
        self.mse_threshold_line = pg.InfiniteLine(
            pos=self.mse_threshold,
            angle=0,
            pen=pg.mkPen(color=Config.COLOR_THRESHOLD, width=2, style=Qt.DashLine),
            label=f'Threshold: {self.mse_threshold}',
            labelOpts={'color': Config.COLOR_THRESHOLD, 'position': 0.95}
        )
        self.mse_plot.addItem(self.mse_threshold_line)
        self.mse_anomaly_markers = pg.ScatterPlotItem(
            size=10, pen=pg.mkPen(None), brush=pg.mkBrush(Config.COLOR_WARNING)
        )
        self.mse_plot.addItem(self.mse_anomaly_markers)
        self.mse_plot.setYRange(0, 0.1)
        
        # --- Vibration RMS Graph ---
        self.rms_curve = self.rms_plot.plot(
            pen=pg.mkPen(color='#ffaa00', width=2),
            name="Vibration RMS"
        )
        self.rms_threshold_line = pg.InfiniteLine(
            pos=self.rms_threshold,
            angle=0,
            pen=pg.mkPen(color=Config.COLOR_THRESHOLD, width=2, style=Qt.DashLine),
            label=f'Threshold: {self.rms_threshold}',
            labelOpts={'color': Config.COLOR_THRESHOLD, 'position': 0.95}
        )
        self.rms_plot.addItem(self.rms_threshold_line)
        self.rms_anomaly_markers = pg.ScatterPlotItem(
            size=10, pen=pg.mkPen(None), brush=pg.mkBrush(Config.COLOR_WARNING)
        )
        self.rms_plot.addItem(self.rms_anomaly_markers)
        self.rms_plot.setYRange(0, 1.0)
        
    def _create_metrics_group(self) -> QGroupBox:
        """Create the metrics display group."""
        group = QGroupBox("◈ DUAL-SENSOR METRICS")
        layout = QGridLayout(group)
        layout.setSpacing(20)
        
        # Create metric displays - updated for dual-sensor + temperature
        metrics = [
            ("AUDIO MSE", "mse_value", "0.0000"),
            ("MSE THRESHOLD", "mse_th_value", "---"),
            ("TEMPERATURE", "temp_value", "---"),
            ("VIBRATION RMS", "rms_value", "0.000 g"),
            ("RMS THRESHOLD", "rms_th_value", "---"),
            ("SEVERITY", "severity_value", "NORMAL"),
            ("FAULT BAND", "band_value", "---"),
            ("ANOMALIES", "anomalies_value", "0"),
            ("ANOMALY RATE", "rate_value", "0.0%"),
        ]
        
        for i, (label_text, attr_name, default_value) in enumerate(metrics):
            row, col = divmod(i, 3)
            
            container = QFrame()
            container.setStyleSheet(f"""
                QFrame {{
                    background-color: #0a0a15;
                    border: 1px solid #222244;
                    border-radius: 5px;
                    padding: 10px;
                }}
            """)
            container.setMinimumHeight(70)
            container_layout = QVBoxLayout(container)
            container_layout.setSpacing(2)
            container_layout.setContentsMargins(5, 5, 5, 5)
            
            label = QLabel(label_text)
            label.setFont(QFont("Consolas", 9))
            label.setStyleSheet(f"color: {Config.COLOR_TEXT_DIM};")
            label.setAlignment(Qt.AlignCenter)
            container_layout.addWidget(label)
            
            value = QLabel(default_value)
            value.setFont(QFont("Consolas", 14, QFont.Bold))
            value.setStyleSheet(f"color: {Config.COLOR_ACCENT};")
            value.setAlignment(Qt.AlignCenter)
            container_layout.addWidget(value)
            
            setattr(self, attr_name, value)
            layout.addWidget(container, row, col)
            
        return group
        
    def _create_status_group(self) -> QGroupBox:
        """Create the status indicator group."""
        group = QGroupBox("◈ SYSTEM STATUS")
        layout = QVBoxLayout(group)
        layout.setAlignment(Qt.AlignCenter)
        
        self.traffic_light = TrafficLightWidget()
        layout.addWidget(self.traffic_light, alignment=Qt.AlignCenter)
        
        # Status text
        self.status_label = QLabel("AWAITING CONNECTION")
        self.status_label.setFont(QFont("Consolas", 12, QFont.Bold))
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet(f"color: {Config.COLOR_TEXT_DIM};")
        layout.addWidget(self.status_label)
        
        return group
        
    def _create_log_group(self) -> QGroupBox:
        """Create the log console group."""
        group = QGroupBox("◈ SERIAL CONSOLE")
        layout = QVBoxLayout(group)
        
        self.log_console = QPlainTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setMaximumBlockCount(500)  # Limit log lines
        layout.addWidget(self.log_console)
        
        return group
        
    def _setup_timers(self):
        """Setup update timers."""
        self.plot_timer = QTimer()
        self.plot_timer.timeout.connect(self._update_plot)
        self.plot_timer.start(Config.PLOT_UPDATE_RATE_MS)
        
    def _refresh_ports(self):
        """Refresh available COM ports."""
        self.port_combo.clear()
        ports = serial.tools.list_ports.comports()
        for port in ports:
            self.port_combo.addItem(f"{port.device} - {port.description}", port.device)
        
        if self.port_combo.count() == 0:
            self.port_combo.addItem("No COM ports found", None)
            
    def _toggle_connection(self):
        """Connect or disconnect from serial port."""
        if self.is_connected:
            self._disconnect()
        else:
            self._connect()
            
    def _connect(self):
        """Establish serial connection."""
        port = self.port_combo.currentData()
        if not port:
            QMessageBox.warning(self, "Error", "Please select a valid COM port")
            return
            
        self.serial_thread = SerialReaderThread(port)
        self.serial_thread.data_received.connect(self._on_data_received)
        self.serial_thread.connection_error.connect(self._on_connection_error)
        self.serial_thread.start()
        
        self.is_connected = True
        self.connect_btn.setText("DISCONNECT")
        self.connect_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #3a1a1a;
                border-color: {Config.COLOR_WARNING};
            }}
        """)
        self.port_combo.setEnabled(False)
        self.status_bar.showMessage(f"◉ Connected to {port} @ {Config.BAUD_RATE} baud")
        self._log_message(f"[SYSTEM] Connected to {port}", "#00ff88")
        
        # Update UI status
        self.status_label.setText("SYSTEM READY - WAITING FOR DATA")
        self.status_label.setStyleSheet(f"color: {Config.COLOR_ACCENT};")
        self.traffic_light.set_status("NORMAL")
        
    def _disconnect(self):
        """Close serial connection."""
        if self.serial_thread:
            self.serial_thread.stop()
            self.serial_thread = None
            
        self.is_connected = False
        self.connect_btn.setText("CONNECT")
        self.connect_btn.setStyleSheet("")
        self.port_combo.setEnabled(True)
        self.status_bar.showMessage("◉ Disconnected")
        self.traffic_light.set_status("IDLE")
        self.status_label.setText("DISCONNECTED")
        self.status_label.setStyleSheet(f"color: {Config.COLOR_TEXT_DIM};")
        self._log_message("[SYSTEM] Disconnected", "#ffaa00")
        
    def _on_data_received(self, line: str):
        """Handle incoming serial data."""
        # Parse the line
        parsed = LogParser.parse_line(line)
        
        # Log the raw line with severity-based coloring
        severity = parsed.get('severity', 'NORMAL')
        if parsed['is_anomaly'] is True:
            sev_colors = {'CRITICAL': '#ff0000', 'WARNING': '#ff8800', 'WATCH': '#ffdd00'}
            self._log_message(line.strip(), sev_colors.get(severity, Config.COLOR_WARNING))
        elif parsed['mse'] is not None:
            self._log_message(line.strip(), Config.COLOR_ACCENT)
        else:
            self._log_message(line.strip(), Config.COLOR_TEXT_DIM)
        
        # Update data if sensor readings found
        if parsed['mse'] is not None:
            rms = parsed.get('rms', 0.0)
            temp = parsed.get('temp', 0.0)
            mse_th = parsed.get('mse_threshold')
            rms_th = parsed.get('rms_threshold')
            anomaly_source = parsed.get('anomaly_source')
            self._update_sensor_data(
                parsed['mse'], rms, temp,
                parsed['is_anomaly'], anomaly_source,
                mse_th, rms_th,
                parsed.get('calibrating', False),
                severity,
                parsed.get('dominant_band')
            )
            
    def _update_sensor_data(self, mse: float, rms: float, temp: float, is_anomaly: bool, 
                            anomaly_source: str, mse_th: float, rms_th: float,
                            calibrating: bool, severity: str = 'NORMAL',
                            dominant_band: str = None):
        """Update dual-sensor data and metrics."""
        current_time = time.time() - self.start_time
        
        # Store data
        self.mse_history.append(mse)
        self.rms_history.append(rms)
        self.time_history.append(current_time)
        
        # Update thresholds if provided
        if mse_th is not None:
            self.mse_threshold = mse_th
            self.mse_th_value.setText(f"{mse_th:.4f}")
            self.mse_threshold_line.setPos(mse_th)
        if rms_th is not None:
            self.rms_threshold = rms_th
            self.rms_th_value.setText(f"{rms_th:.3f} g")
            self.rms_threshold_line.setPos(rms_th)
        
        # Update statistics (only if not calibrating)
        if not calibrating:
            self.total_samples += 1
            self.last_mse = mse
            self.last_rms = rms
            
            if is_anomaly:
                self.anomaly_count += 1
                # Add anomaly markers to the correct graph(s)
                if anomaly_source in ['AUDIO', 'BOTH']:
                    self.mse_anomaly_markers.addPoints([{'pos': (current_time, mse)}])
                if anomaly_source in ['VIBRATION', 'BOTH']:
                    self.rms_anomaly_markers.addPoints([{'pos': (current_time, rms)}])
        
        # Update UI - MSE
        self.mse_value.setText(f"{mse:.4f}")
        if is_anomaly and anomaly_source in ['AUDIO', 'BOTH']:
            self.mse_value.setStyleSheet(f"color: {Config.COLOR_WARNING};")
        else:
            self.mse_value.setStyleSheet(f"color: {Config.COLOR_ACCENT};")
        
        # Update UI - RMS
        self.rms_value.setText(f"{rms:.3f} g")
        if is_anomaly and anomaly_source in ['VIBRATION', 'BOTH']:
            self.rms_value.setStyleSheet(f"color: {Config.COLOR_WARNING};")
        else:
            self.rms_value.setStyleSheet(f"color: {Config.COLOR_ACCENT};")
        
        # Update UI - Temperature
        if temp > 0:
            self.temp_value.setText(f"{temp:.1f}°C")
            self.temp_value.setStyleSheet(f"color: {Config.COLOR_ACCENT};")
        
        # Update UI - Severity
        sev_colors = {
            'NORMAL': Config.COLOR_ACCENT,
            'WATCH': '#ffdd00',
            'WARNING': '#ff8800',
            'CRITICAL': '#ff0000'
        }
        self.severity_value.setText(severity)
        self.severity_value.setStyleSheet(f"color: {sev_colors.get(severity, Config.COLOR_ACCENT)};")
        
        # Update UI - Fault Band
        if dominant_band and is_anomaly:
            self.band_value.setText(dominant_band)
            self.band_value.setStyleSheet(f"color: {Config.COLOR_WARNING};")
        elif dominant_band:
            self.band_value.setText(dominant_band)
            self.band_value.setStyleSheet(f"color: {Config.COLOR_TEXT_DIM};")
            
        self.anomalies_value.setText(str(self.anomaly_count))
        
        rate = (self.anomaly_count / self.total_samples * 100) if self.total_samples > 0 else 0
        self.rate_value.setText(f"{rate:.1f}%")
        
        # Update traffic light with severity-based status
        if calibrating:
            self.traffic_light.set_status("IDLE")
            self.status_label.setText("⏳ CALIBRATING...")
            self.status_label.setStyleSheet(f"color: #ffaa00;")
        elif is_anomaly:
            self.traffic_light.set_status(severity)  # WATCH, WARNING, or CRITICAL
            sev_emoji = {'WATCH': '⚠️', 'WARNING': '🟠', 'CRITICAL': '🚨'}
            emoji = sev_emoji.get(severity, '🚨')
            source_text = f"{emoji} {severity} [{anomaly_source}]" if anomaly_source else f"{emoji} {severity}"
            self.status_label.setText(source_text)
            self.status_label.setStyleSheet(f"color: {sev_colors.get(severity, Config.COLOR_WARNING)};")
        else:
            self.traffic_light.set_status("NORMAL")
            self.status_label.setText("SYSTEM NOMINAL")
            self.status_label.setStyleSheet(f"color: {Config.COLOR_ACCENT};")
            
    def _update_plot(self):
        """Update both plots with current data."""
        if len(self.time_history) > 0:
            times = list(self.time_history)
            self.mse_curve.setData(times, list(self.mse_history))
            self.rms_curve.setData(times, list(self.rms_history))
            
            # Auto-scale MSE Y axis
            if len(self.mse_history) > 0:
                max_mse = min(
                    max(max(self.mse_history), self.mse_threshold * 2),
                    0.2
                )
                self.mse_plot.setYRange(0, max_mse * 1.2)
            
            # Auto-scale RMS Y axis
            if len(self.rms_history) > 0:
                max_rms = min(
                    max(max(self.rms_history), self.rms_threshold * 2),
                    5.0
                )
                self.rms_plot.setYRange(0, max_rms * 1.2)
                
    def _log_message(self, message: str, color: str = Config.COLOR_TEXT):
        """Add a message to the log console."""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        # QPlainTextEdit - use plain text with timestamp
        formatted = f"[{timestamp}] {message}"
        self.log_console.appendPlainText(formatted)
        
        # Auto-scroll to bottom
        scrollbar = self.log_console.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
    def _on_connection_error(self, error: str):
        """Handle serial connection errors."""
        self._log_message(f"[ERROR] {error}", Config.COLOR_WARNING)
        self._disconnect()
        QMessageBox.critical(self, "Connection Error", error)
        
    def _clear_data(self):
        """Clear all data and reset."""
        self.mse_history.clear()
        self.rms_history.clear()
        self.time_history.clear()
        self.start_time = time.time()
        self.total_samples = 0
        self.anomaly_count = 0
        
        # Reset UI
        self.mse_value.setText("0.0000")
        self.mse_th_value.setText("---")
        self.rms_value.setText("0.000 g")
        self.rms_th_value.setText("---")
        self.severity_value.setText("NORMAL")
        self.severity_value.setStyleSheet(f"color: {Config.COLOR_ACCENT};")
        self.band_value.setText("---")
        self.band_value.setStyleSheet(f"color: {Config.COLOR_TEXT_DIM};")
        self.anomalies_value.setText("0")
        self.rate_value.setText("0.0%")
        self.log_console.clear()
        self.mse_anomaly_markers.clear()
        self.rms_anomaly_markers.clear()
        
        self._log_message("[SYSTEM] Data cleared", "#00ff88")
        
    def closeEvent(self, event):
        """Handle window close - cleanup serial connection."""
        if self.serial_thread:
            self.serial_thread.stop()
        event.accept()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Application entry point."""
    # High DPI support
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    app.setApplicationName("Predictive Maintenance GCS")
    
    window = GroundControlStation()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
