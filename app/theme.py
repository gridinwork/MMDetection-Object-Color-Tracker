"""Dark theme for the studio window."""

from __future__ import annotations

THEME = """
QWidget {
    background: #1b1d22;
    color: #e8eaed;
    font-family: "Segoe UI";
    font-size: 10pt;
}
QMainWindow, QScrollArea, QFrame {
    background: #1b1d22;
}
QGroupBox {
    border: 1px solid #3a3f4b;
    border-radius: 8px;
    margin-top: 12px;
    padding: 8px;
    background: #23262d;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: #9ec1ff;
}
QLabel#muted {
    color: #9aa0a6;
}
QLabel#recordTimer {
    color: #ffe8e4;
    background: #3a2220;
    border: 1px solid #c4564e;
    border-radius: 6px;
    padding: 4px 12px;
    font-size: 14pt;
    font-weight: 700;
    min-width: 168px;
}
QLineEdit, QComboBox, QSpinBox, QListWidget, QPlainTextEdit, QTableWidget {
    background: #2c3038;
    border: 1px solid #3a3f4b;
    border-radius: 6px;
    padding: 4px 6px;
    selection-background-color: #3d8bfd;
}
QComboBox::drop-down {
    border: none;
    width: 22px;
}
QComboBox QAbstractItemView {
    background: #2c3038;
    color: #e8eaed;
    selection-background-color: #3d8bfd;
    border: 1px solid #3a3f4b;
}
QPushButton {
    background: #343a46;
    border: 1px solid #4a5160;
    border-radius: 6px;
    padding: 6px 12px;
}
QPushButton:hover {
    background: #3e4656;
}
QPushButton:pressed {
    background: #2a3140;
}
QPushButton#start {
    background: #1f7a4d;
    border-color: #2ea86a;
}
QPushButton#stop, QPushButton#record {
    background: #8a3530;
    border-color: #c4564e;
}
QPushButton#accent {
    background: #2457b5;
    border-color: #3d8bfd;
}
QCheckBox {
    spacing: 6px;
    background: transparent;
}
QSlider::groove:horizontal {
    height: 6px;
    background: #3a3f4b;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    width: 14px;
    margin: -5px 0;
    border-radius: 7px;
    background: #3d8bfd;
}
QScrollArea {
    border: none;
}
QStatusBar {
    background: #14161a;
    color: #c5c9d1;
}
QHeaderView::section {
    background: #2c3038;
    color: #e8eaed;
    border: none;
    padding: 4px;
}
QProgressBar {
    border: 1px solid #3a3f4b;
    border-radius: 4px;
    text-align: center;
    background: #2c3038;
}
QProgressBar::chunk {
    background: #3d8bfd;
}
"""
