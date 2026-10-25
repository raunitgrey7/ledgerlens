"""Tesseract adapter through pytesseract, used only when the binary is installed.

The build machine had no ``tesseract`` binary, so this adapter is exercised by a unit
test with a fake pytesseract module rather than by the benchmark. It converts the
``image_to_data`` table into the shared OCRPage structure.
"""

from __future__ import annotations

import shutil
from collections import defaultdict
from typing import Any

import numpy as np
from PIL import Image

from ledgerlens.pipeline.ocr.base import OCRLine, OCRPage, OCRWord


def tesseract_available() -> bool:
    """True when both the binary and the Python wrapper can be found."""
    if shutil.which("tesseract") is None:
        return False
    try:
        import pytesseract  # noqa: F401
    except ImportError:
        return False
    return True


class TesseractOCR:
    """Runs ``tesseract`` with page segmentation mode 4 (columns of variable-size text)."""

    name = "tesseract"

    def __init__(self, module: Any | None = None, config: str = "--psm 4") -> None:
        if module is None:
            import pytesseract

            module = pytesseract
        self.module = module
        self.config = config

    def recognize(self, gray: np.ndarray, ink: np.ndarray) -> OCRPage:
        table = self.module.image_to_data(
            Image.fromarray(gray), config=self.config, output_type=self.module.Output.DICT
        )
        return page_from_table(table, gray.shape[1], gray.shape[0], self.name)


def page_from_table(table: dict[str, list[Any]], width: int, height: int, engine: str) -> OCRPage:
    """Group the flat pytesseract rows into lines keyed by (block, paragraph, line)."""
    grouped: dict[tuple[int, int, int], list[OCRWord]] = defaultdict(list)
    for index, text in enumerate(table["text"]):
        token = str(text).strip()
        if not token:
            continue
        conf = float(table["conf"][index])
        word = OCRWord(
            token,
            max(0.0, min(1.0, conf / 100.0)),
            int(table["left"][index]),
            int(table["top"][index]),
            int(table["left"][index]) + int(table["width"][index]),
            int(table["top"][index]) + int(table["height"][index]),
        )
        key = (int(table["block_num"][index]), int(table["par_num"][index]), int(table["line_num"][index]))
        grouped[key].append(word)
    lines: list[OCRLine] = []
    for words in grouped.values():
        words.sort(key=lambda w: w.x0)
        y1 = max(w.y1 for w in words)
        lines.append(
            OCRLine(
                words,
                min(w.x0 for w in words),
                min(w.y0 for w in words),
                max(w.x1 for w in words),
                y1,
                float(y1),
                float(np.median([w.y1 - w.y0 for w in words])),
            )
        )
    lines.sort(key=lambda line: (line.y0, line.x0))
    return OCRPage(lines, width, height, engine)
