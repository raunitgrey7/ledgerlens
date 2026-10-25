"""Runtime settings read from the environment once, with safe defaults for offline use."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Every knob the API, CLI and benchmark share."""

    data_dir: Path
    ocr_engine: str
    threshold: float
    api_key: str | None
    db_path: Path
    llm_enabled: bool
    llm_model: str
    log_level: str
    max_upload_bytes: int
    calibration_path: Path

    @property
    def atlas_dir(self) -> Path:
        return self.data_dir / "atlas"


def _read_threshold(default: float) -> float:
    """The acceptance threshold: environment first, then the benchmark's chosen value, then the default."""
    raw = os.environ.get("LEDGERLENS_THRESHOLD")
    if raw:
        return float(raw)
    chosen = Path(os.environ.get("LEDGERLENS_THRESHOLD_FILE", "benchmarks/threshold.json"))
    if chosen.exists():
        import json

        try:
            return float(json.loads(chosen.read_text())["threshold"])
        except (KeyError, ValueError, TypeError):
            return default
    return default


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    data_dir = Path(os.environ.get("LEDGERLENS_DATA_DIR", "data"))
    return Settings(
        data_dir=data_dir,
        ocr_engine=os.environ.get("LEDGERLENS_OCR", "auto"),
        threshold=_read_threshold(0.5),
        api_key=os.environ.get("APP_API_KEY") or None,
        db_path=Path(os.environ.get("LEDGERLENS_DB", str(data_dir / "ledgerlens.db"))),
        llm_enabled=os.environ.get("LEDGERLENS_LLM", "0") == "1",
        llm_model=os.environ.get("LEDGERLENS_LLM_MODEL", "claude-haiku-5-5"),
        log_level=os.environ.get("LEDGERLENS_LOG_LEVEL", "INFO"),
        max_upload_bytes=int(os.environ.get("LEDGERLENS_MAX_UPLOAD_BYTES", str(20 * 1024 * 1024))),
        calibration_path=Path(os.environ.get("LEDGERLENS_CALIBRATION_FILE", "benchmarks/calibration.json")),
    )
