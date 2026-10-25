"""The OCR contract every engine fulfils: words with boxes and confidences, grouped in lines.

Downstream code (layout, extraction, metrics) only ever sees these records, so a
different engine can be dropped in without touching anything else. Coordinates are
in the deskewed page frame produced by ``pipeline.preprocess``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class OCRWord:
    """One recognised token."""

    text: str
    confidence: float
    x0: int
    y0: int
    x1: int
    y1: int
    char_confidences: list[float] = field(default_factory=list)
    char_scores: list[tuple[float, float]] = field(default_factory=list)

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0


@dataclass
class OCRLine:
    """Words sharing a baseline, left to right."""

    words: list[OCRWord]
    x0: int
    y0: int
    x1: int
    y1: int
    baseline: float
    height: float

    @property
    def text(self) -> str:
        return " ".join(word.text for word in self.words)

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0


@dataclass
class OCRPage:
    """Everything an engine returns for one page."""

    lines: list[OCRLine]
    width: int
    height: int
    engine: str

    @property
    def words(self) -> list[OCRWord]:
        return [word for line in self.lines for word in line.words]

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)


class OCREngine(Protocol):
    """Recognise text on a preprocessed page."""

    name: str

    def recognize(self, gray: np.ndarray, ink: np.ndarray) -> OCRPage:
        """``gray`` is the deskewed greyscale page, ``ink`` its binary mask (True = ink)."""
        ...
