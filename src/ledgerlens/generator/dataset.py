"""Build a seeded dataset split on disk: PNG pages next to JSON ground truth.

A split is fully determined by (name, size, seed, mix). The manifest records those
inputs plus a fingerprint of every ground-truth file so the benchmark can refuse to
score a stale or hand-edited split. Each page derives its own RNG from the split seed
and its index, so page i is identical whether the split has 10 pages or 10,000.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from PIL import Image
from pydantic import BaseModel

from ledgerlens.generator.content import generate_content
from ledgerlens.generator.degrade import degrade
from ledgerlens.generator.spec import choose_spec, family_by_name
from ledgerlens.generator.templates import render_document, template_for
from ledgerlens.schemas import DocType, GroundTruth, Severity


@dataclass(frozen=True)
class SplitSpec:
    """Inputs that define a split."""

    name: str
    size: int
    seed: int
    type_weights: tuple[float, float, float] = (0.5, 0.3, 0.2)
    severity_weights: tuple[float, float, float] = (0.2, 0.4, 0.4)
    unseen_fraction: float = 0.15
    families: tuple[str, ...] = field(default_factory=tuple)


class Manifest(BaseModel):
    """What was built, from which inputs, with a content fingerprint."""

    spec: dict[str, object]
    doc_ids: list[str]
    fingerprint: str
    generator_version: str


GENERATOR_VERSION = "1"


def page_rng(seed: int, index: int) -> random.Random:
    """Independent stream per page: the index is mixed into the split seed."""
    return random.Random(seed * 1_000_003 + index)


def build_document(spec: SplitSpec, index: int) -> tuple[Image.Image, GroundTruth]:
    """Render and degrade page ``index`` of the split."""
    rng = page_rng(spec.seed, index)
    doc_type = rng.choices(list(DocType), weights=spec.type_weights, k=1)[0]
    severity = rng.choices(list(Severity), weights=spec.severity_weights, k=1)[0]
    content = generate_content(rng, doc_type)
    family = family_by_name(rng.choice(spec.families)) if spec.families else None
    render_spec = choose_spec(rng, content, family, spec.unseen_fraction)
    template = template_for(content, rng)
    canvas = render_document(content, template, render_spec, rng)
    page, record = degrade(canvas.image, severity, rng)
    truth = GroundTruth(
        doc_id=f"{spec.name}-{index:05d}",
        doc_type=doc_type,
        template=template,
        font=render_spec.family.name,
        font_seen=render_spec.family.seen,
        severity=severity,
        seed=spec.seed,
        width=page.width,
        height=page.height,
        fields=content.fields,
        line_items=content.line_items,
        transactions=content.transactions,
        words=canvas.words,
        degradation=record,
    )
    return page, truth


def split_dir(root: Path, name: str) -> Path:
    return root / name


def load_manifest(directory: Path) -> Manifest | None:
    path = directory / "manifest.json"
    if not path.exists():
        return None
    return Manifest.model_validate_json(path.read_text())


def build_split(root: Path, spec: SplitSpec, force: bool = False) -> Manifest:
    """Write the split unless an identical one is already on disk."""
    directory = split_dir(root, spec.name)
    existing = load_manifest(directory)
    if (
        existing is not None
        and not force
        and existing.spec == _spec_dict(spec)
        and _fingerprint_ok(directory, existing)
    ):
        return existing
    directory.mkdir(parents=True, exist_ok=True)
    for stale in directory.glob("*"):
        stale.unlink()
    digest = hashlib.sha256()
    doc_ids: list[str] = []
    for index in range(spec.size):
        page, truth = build_document(spec, index)
        page.save(directory / f"{truth.doc_id}.png", format="PNG", compress_level=3)
        payload = truth.model_dump_json()
        (directory / f"{truth.doc_id}.json").write_text(payload)
        digest.update(payload.encode())
        doc_ids.append(truth.doc_id)
    manifest = Manifest(
        spec=_spec_dict(spec), doc_ids=doc_ids, fingerprint=digest.hexdigest(), generator_version=GENERATOR_VERSION
    )
    (directory / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    return manifest


def _spec_dict(spec: SplitSpec) -> dict[str, object]:
    raw = asdict(spec)
    return json.loads(json.dumps(raw))


def _fingerprint_ok(directory: Path, manifest: Manifest) -> bool:
    digest = hashlib.sha256()
    for doc_id in manifest.doc_ids:
        path = directory / f"{doc_id}.json"
        if not path.exists() or not (directory / f"{doc_id}.png").exists():
            return False
        digest.update(path.read_bytes())
    return digest.hexdigest() == manifest.fingerprint


def load_truth(directory: Path, doc_id: str) -> GroundTruth:
    return GroundTruth.model_validate_json((directory / f"{doc_id}.json").read_text())


def load_page(directory: Path, doc_id: str) -> Image.Image:
    return Image.open(directory / f"{doc_id}.png").convert("RGB")
