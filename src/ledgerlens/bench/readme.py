"""Rewrite the measured-results block of README.md from benchmarks/results.json.

The README never carries hand-typed numbers: this command renders the same tables
the benchmark writes, between two HTML comment markers, and a test asserts the
block matches the committed results file.

    uv run python -m ledgerlens.bench.readme
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ledgerlens.bench.report import markdown_breakdowns, markdown_table

START = "<!-- results:start -->"
END = "<!-- results:end -->"


def render_block(results: dict[str, object]) -> str:
    """The README text between the markers for a results dictionary."""
    meta = results["meta"]  # type: ignore[index]
    header = (
        f"Measured on {meta['generated_at']} (commit {meta['git_commit']}, {meta['platform']}, "  # type: ignore[index]
        f"{meta['machine']}, Python {meta['python']}), test split of "  # type: ignore[index]
        f"{results['test']['documents']} pages, "  # type: ignore[index]
        f"calibration split of {results['calibration']['documents']} pages, OCR engine `{meta['ocr_engine']}`."  # type: ignore[index]
    )
    return f"{START}\n{header}\n\n{markdown_table(results)}\n\n{markdown_breakdowns(results)}\n{END}"


def update_readme(readme: Path, results_path: Path) -> str:
    """Replace the marked block in the README; returns the new README text."""
    text = readme.read_text()
    start, end = text.index(START), text.index(END) + len(END)
    block = render_block(json.loads(results_path.read_text()))
    updated = text[:start] + block + text[end:]
    readme.write_text(updated)
    return updated


def current_block(readme: Path) -> str:
    text = readme.read_text()
    return text[text.index(START) : text.index(END) + len(END)]


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    readme = Path(args[0]) if args else Path("README.md")
    results = Path(args[1]) if len(args) > 1 else Path("benchmarks/results.json")
    update_readme(readme, results)
    print(f"updated {readme} from {results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
