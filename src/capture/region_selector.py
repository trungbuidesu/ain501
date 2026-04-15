"""Fullscreen drag-to-select rectangle for screen capture (dev/debug)."""

from __future__ import annotations

import sys

from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QColor, QGuiApplication, QPainter, QPen
from PySide6.QtWidgets import QApplication, QLabel, QWidget

from src.capture.types import ScreenRegion


def _virtual_desktop_rect() -> QRect:
    geo = QRect()
    for scr in QGuiApplication.screens():
        geo = geo.united(scr.geometry())
    if geo.isNull() or geo.width() < 1:
        ps = QGuiApplication.primaryScreen()
        if ps is not None:
            geo = ps.virtualGeometry()
    return geo


def select_screen_region_interactive() -> ScreenRegion | None:
    """Fullscreen overlay: drag a rectangle; Esc cancels. Returns None if cancelled."""

    print(
        "Screen region: kéo chuột trái chọn vùng, thả để xác nhận, Esc = hủy.",
        file=sys.stderr,
        flush=True,
    )

    app = QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QApplication([])

    result: list[ScreenRegion | None] = [None]

    class Overlay(QWidget):
        def __init__(self) -> None:
            super().__init__(
                None,
                Qt.WindowType.FramelessWindowHint
                | Qt.WindowType.WindowStaysOnTopHint
                | Qt.WindowType.Tool,
            )
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
            # Dimming in paintEvent; window opacity 1 so the overlay is visible.
            self.setWindowOpacity(1.0)
            desktop = _virtual_desktop_rect()
            self.setGeometry(desktop)
            self.setCursor(Qt.CursorShape.CrossCursor)
            self._origin: QPoint | None = None
            self._rubber = QRect()

            self._hint = QLabel(
                "Kéo chuột trái để chọn vùng màn hình  •  Esc = hủy",
                self,
            )
            self._hint.setStyleSheet(
                "color: white; background-color: rgba(0,0,0,160); "
                "padding: 10px 16px; font-size: 14px; border-radius: 6px;"
            )
            self._hint.adjustSize()
            self._hint.move(
                max(8, (self.width() - self._hint.width()) // 2),
                24,
            )

        def resizeEvent(self, event) -> None:
            super().resizeEvent(event)
            self._hint.move(
                max(8, (self.width() - self._hint.width()) // 2),
                24,
            )

        def mousePressEvent(self, event) -> None:
            if event.button() == Qt.MouseButton.LeftButton:
                self._origin = event.globalPosition().toPoint()
                self._rubber = QRect()

        def mouseMoveEvent(self, event) -> None:
            if self._origin is not None:
                gp = event.globalPosition().toPoint()
                self._rubber = QRect(self._origin, gp).normalized()
                self.update()

        def mouseReleaseEvent(self, event) -> None:
            if event.button() == Qt.MouseButton.LeftButton and self._origin is not None:
                r = QRect(self._origin, event.globalPosition().toPoint()).normalized()
                self._origin = None
                if r.width() > 2 and r.height() > 2:
                    result[0] = ScreenRegion(
                        left=r.left(),
                        top=r.top(),
                        width=r.width(),
                        height=r.height(),
                    )
                self.close()
                app_now = QApplication.instance()
                if app_now is not None:
                    app_now.quit()

        def keyPressEvent(self, event) -> None:
            if event.key() == Qt.Key.Key_Escape:
                result[0] = None
                self.close()
                app_now = QApplication.instance()
                if app_now is not None:
                    app_now.quit()

        def paintEvent(self, event) -> None:
            painter = QPainter(self)
            painter.fillRect(self.rect(), QColor(0, 0, 0, 120))
            if self._rubber.isValid() and self._rubber.width() > 0:
                painter.setPen(QPen(QColor(255, 220, 0), 3))
                tl = self.mapFromGlobal(self._rubber.topLeft())
                br = self.mapFromGlobal(self._rubber.bottomRight())
                painter.drawRect(QRect(tl, br).normalized())

    overlay = Overlay()
    overlay.setWindowTitle("Chọn vùng màn hình")
    overlay.show()
    overlay.raise_()
    overlay.activateWindow()
    overlay.setFocus()
    app.exec()
    if owns_app:
        app.quit()
    return result[0]
