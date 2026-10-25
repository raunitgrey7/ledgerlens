import json
from pathlib import Path

import pytest

from ledgerlens.cli import run
from ledgerlens.generator.dataset import SplitSpec, build_document


def test_generate_and_extract_commands(tmp_path: Path, capsys, monkeypatch) -> None:
    monkeypatch.setenv("LEDGERLENS_DATA_DIR", str(Path(__file__).resolve().parents[1] / "data"))
    from ledgerlens import config

    config.get_settings.cache_clear()
    with pytest.raises(SystemExit) as exit_info:
        run(["generate", "--out", str(tmp_path), "--name", "mini", "--size", "2", "--seed", "3"])
    assert exit_info.value.code == 0
    assert "2 pages" in capsys.readouterr().out
    assert len(list((tmp_path / "mini").glob("*.png"))) == 2
    _, truth = build_document(SplitSpec("mini", 2, 3), 0)
    image = tmp_path / "mini" / f"{truth.doc_id}.png"
    with pytest.raises(SystemExit) as exit_info:
        run(["extract", str(image), "--json"])
    assert exit_info.value.code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["doc_type"] == truth.doc_type.value
    assert isinstance(payload["fields"], list) and "decision" in payload


def test_extract_prints_summary(tmp_path: Path, capsys, monkeypatch) -> None:
    monkeypatch.setenv("LEDGERLENS_DATA_DIR", str(Path(__file__).resolve().parents[1] / "data"))
    from ledgerlens import config

    config.get_settings.cache_clear()
    page, truth = build_document(SplitSpec("mini", 1, 8), 0)
    path = tmp_path / "page.png"
    page.save(path)
    with pytest.raises(SystemExit):
        run(["extract", str(path)])
    out = capsys.readouterr().out
    assert truth.doc_type.value in out and "decision=" in out


def test_atlas_command_reports_fingerprint(capsys, monkeypatch) -> None:
    monkeypatch.setenv("LEDGERLENS_DATA_DIR", str(Path(__file__).resolve().parents[1] / "data"))
    from ledgerlens import config

    config.get_settings.cache_clear()
    with pytest.raises(SystemExit):
        run(["atlas"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["templates"] > 1000 and len(payload["fingerprint"]) == 16
