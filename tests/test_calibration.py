import json
import random
from pathlib import Path

from ledgerlens.bench.metrics import Counts, DocScore
from ledgerlens.bench.runner import calibration_samples, rescore
from ledgerlens.pipeline.calibration import FieldCalibrator, fit_calibrator
from ledgerlens.schemas import ExtractedField


def test_identity_without_fit_and_round_trip(tmp_path: Path) -> None:
    identity = FieldCalibrator.load(tmp_path / "missing.json")
    assert identity.calibrate("total", 0.73) == 0.73
    fitted = FieldCalibrator({"vendor_name": (-1.0, 0.8)})
    path = tmp_path / "calibration.json"
    fitted.save(path)
    assert json.loads(path.read_text()) == {"vendor_name": [-1.0, 0.8]}
    loaded = FieldCalibrator.load(path)
    assert loaded.calibrate("vendor_name", 0.9) < 0.9 and loaded.calibrate("total", 0.9) == 0.9
    fields = {"vendor_name": ExtractedField(name="vendor_name", value="X", confidence=0.9)}
    assert loaded.apply(fields)["vendor_name"].confidence == loaded.calibrate("vendor_name", 0.9)
    assert fields["vendor_name"].confidence == 0.9


def test_fit_learns_that_an_overconfident_field_is_wrong_more_often() -> None:
    rng = random.Random(0)
    samples = [("vendor_name", c, rng.random() < c * 0.6) for c in (rng.uniform(0.5, 0.99) for _ in range(300))]
    samples += [("total", c, rng.random() < c) for c in (rng.uniform(0.5, 0.99) for _ in range(300))]
    samples += [("iban", 0.95, True)] * 40
    calibrator = fit_calibrator(samples)
    assert "iban" not in calibrator.params
    assert calibrator.calibrate("vendor_name", 0.9) < calibrator.calibrate("total", 0.9)
    assert calibrator.calibrate("vendor_name", 0.95) > calibrator.calibrate("vendor_name", 0.6)


def _score(conf: float, correct: bool, hard: bool = False) -> DocScore:
    fields = {"vendor_name": Counts(tp=1) if correct else Counts(fp=1, fn=1)}
    return DocScore(
        "d",
        "receipt",
        "t",
        "clean",
        True,
        True,
        fields,
        Counts(),
        Counts(),
        conf,
        False,
        1.0,
        0,
        1,
        predicted_fields={"vendor_name": "X"},
        field_confidences={"vendor_name": conf},
        predicted_type="receipt",
        doc_type_confidence=0.9,
        missing_fields=[],
        hard_rule_failed=hard,
    )


def test_samples_and_rescoring() -> None:
    scores = [_score(0.9, True), _score(0.8, False), _score(0.95, True, hard=True)]
    assert calibration_samples(scores) == [
        ("vendor_name", 0.9, True),
        ("vendor_name", 0.8, False),
        ("vendor_name", 0.95, True),
    ]
    rescored = rescore(scores, FieldCalibrator(), threshold=0.0)
    assert [s.accepted for s in rescored] == [True, True, False]
    assert rescored[0].confidence < 0.9
