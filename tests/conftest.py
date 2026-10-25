"""Shared fixtures: one OCR engine per session, a small cached set of generated pages, an isolated app."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ledgerlens.api.app import create_app
from ledgerlens.api.storage import Storage
from ledgerlens.config import Settings
from ledgerlens.generator.dataset import SplitSpec, build_document
from ledgerlens.pipeline.ocr.atlas import AtlasOCR
from ledgerlens.pipeline.run import Pipeline
from ledgerlens.schemas import DocType, GroundTruth

REPO = Path(__file__).resolve().parents[1]
ATLAS_DIR = REPO / "data" / "atlas"
SEED = 4242


@pytest.fixture(scope="session")
def engine() -> AtlasOCR:
    return AtlasOCR.load_or_build(ATLAS_DIR)


@pytest.fixture(scope="session")
def pipeline(engine: AtlasOCR) -> Pipeline:
    return Pipeline(engine, threshold=0.5)


@pytest.fixture(scope="session")
def clean_pages() -> list[tuple[Image.Image, GroundTruth]]:
    """Clean, seen-font pages: one of each template, deterministic."""
    spec = SplitSpec("fixture", 200, SEED, severity_weights=(1.0, 0.0, 0.0), unseen_fraction=0.0)
    found: dict[str, tuple[Image.Image, GroundTruth]] = {}
    for index in range(200):
        page, truth = build_document(spec, index)
        if truth.template not in found:
            found[truth.template] = (page, truth)
        if len(found) == 10:
            break
    return list(found.values())


@pytest.fixture(scope="session")
def page_by_type(clean_pages: list[tuple[Image.Image, GroundTruth]]) -> dict[DocType, tuple[Image.Image, GroundTruth]]:
    by_type: dict[DocType, tuple[Image.Image, GroundTruth]] = {}
    for page, truth in clean_pages:
        by_type.setdefault(truth.doc_type, (page, truth))
    return by_type


def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    base = {
        "data_dir": REPO / "data",
        "ocr_engine": "atlas",
        "threshold": 0.5,
        "api_key": None,
        "db_path": tmp_path / "test.db",
        "llm_enabled": False,
        "llm_model": "claude-opus-5-5",
        "log_level": "WARNING",
        "max_upload_bytes": 20 * 1024 * 1024,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


@pytest.fixture
def client(engine: AtlasOCR, tmp_path: Path) -> TestClient:
    app = create_app(make_settings(tmp_path), engine=engine, storage=Storage(":memory:"))
    return TestClient(app)
