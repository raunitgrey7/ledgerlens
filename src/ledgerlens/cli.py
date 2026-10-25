"""Command line interface: generate data, extract one page, run the benchmark, serve the API."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image


def _cmd_generate(args: argparse.Namespace) -> int:
    from ledgerlens.generator.dataset import SplitSpec, build_split

    manifest = build_split(
        args.out, SplitSpec(args.name, args.size, args.seed, unseen_fraction=args.unseen), force=args.force
    )
    print(f"{len(manifest.doc_ids)} pages in {args.out / args.name} (fingerprint {manifest.fingerprint[:12]})")
    return 0


def _cmd_extract(args: argparse.Namespace) -> int:
    from ledgerlens.api.app import build_reviewer
    from ledgerlens.config import get_settings
    from ledgerlens.pipeline.calibration import FieldCalibrator
    from ledgerlens.pipeline.ocr.factory import get_engine
    from ledgerlens.pipeline.run import Pipeline

    settings = get_settings()
    pipeline = Pipeline(
        get_engine(args.engine or settings.ocr_engine, str(settings.atlas_dir)),
        settings.threshold,
        build_reviewer(settings),
        FieldCalibrator.load(settings.calibration_path),
    )
    result = pipeline.process(Image.open(args.image).convert("RGB"))
    if args.json:
        print(result.model_dump_json(indent=2))
        return 0
    print(
        f"{result.doc_type.value} ({result.doc_type_confidence:.2f}) decision={result.decision.value} "
        f"confidence={result.document_confidence:.3f} latency={result.latency_ms} ms engine={result.ocr_engine}"
    )
    for field in result.fields:
        print(f"  {field.name:16s} {field.value!s:32s} {field.confidence:.3f}  {field.evidence}")
    for item in result.line_items:
        print(f"  item: {item.description} x{item.quantity} @ {item.unit_price} = {item.amount}")
    for row in result.transactions:
        print(f"  tx: {row.date} {row.description} debit={row.debit} credit={row.credit} balance={row.balance}")
    for rule in result.rules:
        print(f"  rule {rule.name}: {rule.passed} {rule.detail}")
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    from ledgerlens.bench.__main__ import main

    return main(args.bench_args)


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from ledgerlens.api.app import create_app

    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="warning")
    return 0


def _cmd_atlas(args: argparse.Namespace) -> int:
    from ledgerlens.config import get_settings
    from ledgerlens.pipeline.ocr.atlas import AtlasOCR

    engine = AtlasOCR.load_or_build(get_settings().atlas_dir)
    print(
        json.dumps(
            {
                "templates": len(engine.atlas.labels),
                "calibration": list(engine.atlas.calibration),
                "fingerprint": engine.atlas.fingerprint,
            }
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ledgerlens", description="document intelligence for finance")
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="render a seeded synthetic split")
    gen.add_argument("--out", type=Path, default=Path("data/synthetic"))
    gen.add_argument("--name", default="sample")
    gen.add_argument("--size", type=int, default=20)
    gen.add_argument("--seed", type=int, default=7)
    gen.add_argument("--unseen", type=float, default=0.15)
    gen.add_argument("--force", action="store_true")
    gen.set_defaults(func=_cmd_generate)
    ext = sub.add_parser("extract", help="run the pipeline on one image")
    ext.add_argument("image", type=Path)
    ext.add_argument("--json", action="store_true")
    ext.add_argument("--engine", default=None)
    ext.set_defaults(func=_cmd_extract)
    bench = sub.add_parser("bench", help="run the benchmark (arguments pass through)")
    bench.add_argument("bench_args", nargs=argparse.REMAINDER)
    bench.set_defaults(func=_cmd_bench)
    serve = sub.add_parser("serve", help="start the HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=_cmd_serve)
    atlas = sub.add_parser("atlas", help="build or load the OCR glyph atlas")
    atlas.set_defaults(func=_cmd_atlas)
    return parser


def run(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    raise SystemExit(args.func(args))
