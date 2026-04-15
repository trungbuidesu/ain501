"""Optional QLabel debug overlay for caption text."""

from __future__ import annotations


class CaptionDebugOverlay:
    """Small wrapper with no-op fallback if Qt UI is unavailable."""

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self._widget = None
        if not enabled:
            return
        try:
            from PySide6.QtCore import Qt
            from PySide6.QtWidgets import QApplication, QLabel
        except Exception:
            self.enabled = False
            return
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        label = QLabel("")
        label.setWindowTitle("Phase2 Caption Debug")
        label.setWindowFlags(label.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        label.setStyleSheet(
            "background-color: rgba(0,0,0,180); color: white; padding: 10px; "
            "font-size: 14px;"
        )
        label.resize(720, 60)
        label.show()
        self._widget = label

    def show_text(self, text: str) -> None:
        if self._widget is not None:
            self._widget.setText(text)

    def close(self) -> None:
        if self._widget is not None:
            self._widget.close()
            self._widget = None
