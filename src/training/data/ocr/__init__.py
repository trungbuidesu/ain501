# -*- coding: utf-8 -*-
"""OCR dataset helpers."""

from src.training.data.ocr.textocr import (
    IcdarTextBox,
    TextOcrTextBox,
    parse_icdar_gt,
    parse_textocr_annotations,
)

__all__ = [
    "IcdarTextBox",
    "TextOcrTextBox",
    "parse_icdar_gt",
    "parse_textocr_annotations",
]
