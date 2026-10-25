"""The README results block must be exactly what the committed results.json renders to."""

from pathlib import Path

import pytest

from ledgerlens.bench.readme import current_block, render_block

REPO = Path(__file__).resolve().parents[1]


def test_readme_block_matches_results_json() -> None:
    results = REPO / "benchmarks" / "results.json"
    if not results.exists():
        pytest.skip("no committed benchmark results")
    import json

    assert current_block(REPO / "README.md") == render_block(json.loads(results.read_text()))
