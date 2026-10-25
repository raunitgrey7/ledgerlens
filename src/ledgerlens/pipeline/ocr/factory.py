"""Pick the OCR engine: Tesseract when installed and requested, else the shipped atlas."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from ledgerlens.pipeline.ocr.atlas import AtlasOCR
from ledgerlens.pipeline.ocr.base import OCREngine
from ledgerlens.pipeline.ocr.tesseract import TesseractOCR, tesseract_available


def resolve_engine_name(requested: str) -> str:
    """'auto' prefers Tesseract when available; explicit names are honoured or rejected."""
    if requested == "auto":
        return "tesseract" if tesseract_available() else "atlas"
    if requested == "tesseract" and not tesseract_available():
        raise RuntimeError("tesseract requested but the binary or pytesseract is missing")
    if requested not in ("atlas", "tesseract"):
        raise ValueError(f"unknown OCR engine {requested!r}")
    return requested


@lru_cache(maxsize=4)
def get_engine(requested: str, cache_dir: str) -> OCREngine:
    """Engines are expensive to build, so one instance per (name, cache) is shared."""
    name = resolve_engine_name(requested)
    if name == "tesseract":
        return TesseractOCR()
    return AtlasOCR.load_or_build(Path(cache_dir))
