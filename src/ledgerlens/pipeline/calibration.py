"""Per-field confidence calibration fitted on a labelled calibration split.

Raw field confidences come from the OCR character probabilities and the rule
adjustments. They rank values well within a field but are not comparable across
fields: a name has no checksum, so a raw 0.8 on a vendor name is wrong far more often
than a raw 0.8 on a total that reconciled. A one-feature logistic regression per field,
on the logit of the raw confidence, maps each field onto the probability that it is
right. The benchmark fits it on the calibration split, freezes it, and applies it to
the test split; the API loads the same file.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ledgerlens.schemas import ExtractedField

MIN_SAMPLES = 15
EPSILON = 1e-4
RIDGE = 1.0


def _logit(p: float) -> float:
    q = min(max(p, EPSILON), 1.0 - EPSILON)
    return math.log(q / (1.0 - q))


@dataclass
class FieldCalibrator:
    """Maps raw field confidence to calibrated probability; identity for fields without a fit."""

    params: dict[str, tuple[float, float]] = field(default_factory=dict)

    def calibrate(self, name: str, confidence: float) -> float:
        if name not in self.params:
            return confidence
        bias, weight = self.params[name]
        return round(1.0 / (1.0 + math.exp(-(bias + weight * _logit(confidence)))), 4)

    def apply(self, fields: dict[str, ExtractedField]) -> dict[str, ExtractedField]:
        return {
            name: f.model_copy(update={"confidence": self.calibrate(name, f.confidence)}) for name, f in fields.items()
        }

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({name: list(p) for name, p in sorted(self.params.items())}, indent=2))

    @classmethod
    def load(cls, path: Path | None) -> FieldCalibrator:
        """Identity when the file is absent, so the pipeline works before any benchmark ran."""
        if path is None or not path.exists():
            return cls()
        raw = json.loads(path.read_text())
        return cls({name: (float(p[0]), float(p[1])) for name, p in raw.items()})


def fit_calibrator(samples: list[tuple[str, float, bool]]) -> FieldCalibrator:
    """Fit one logistic per field from (field, raw confidence, value correct) samples.

    Fields with fewer than MIN_SAMPLES samples, or with only right or only wrong values,
    keep the identity map. A ridge penalty pulls the weight towards 1 and the bias
    towards 0, so small samples move the map only as far as the evidence supports.
    """
    by_field: dict[str, list[tuple[float, bool]]] = {}
    for name, confidence, correct in samples:
        by_field.setdefault(name, []).append((confidence, correct))
    params: dict[str, tuple[float, float]] = {}
    for name, pairs in by_field.items():
        targets = np.array([float(ok) for _, ok in pairs])
        if len(pairs) < MIN_SAMPLES or targets.min() == targets.max():
            continue
        features = np.array([_logit(conf) for conf, _ in pairs])
        params[name] = _fit_one(features, targets)
    return FieldCalibrator(params)


def _fit_one(features: np.ndarray, targets: np.ndarray, iterations: int = 50) -> tuple[float, float]:
    """Damped Newton's method on the penalised log-likelihood with prior (bias 0, weight 1).

    Each Newton step is halved until the objective decreases, which keeps the fit
    stable when the optimum lies far from the prior.
    """
    design = np.column_stack([np.ones(len(features)), features])
    prior = np.array([0.0, 1.0])
    weights = prior.copy()
    for _ in range(iterations):
        p = _sigmoid(design @ weights)
        gradient = design.T @ (p - targets) + RIDGE * (weights - prior)
        hessian = (design * (p * (1 - p))[:, None]).T @ design + RIDGE * np.eye(2)
        step = np.linalg.solve(hessian, gradient)
        current = _objective(design, targets, weights, prior)
        scale = 1.0
        while scale > 1e-4 and _objective(design, targets, weights - scale * step, prior) > current:
            scale /= 2.0
        weights = weights - scale * step
        if float(np.abs(scale * step).max()) < 1e-8:
            break
    return float(weights[0]), float(weights[1])


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def _objective(design: np.ndarray, targets: np.ndarray, weights: np.ndarray, prior: np.ndarray) -> float:
    """Negative log-likelihood plus the ridge penalty towards the prior."""
    p = np.clip(_sigmoid(design @ weights), 1e-12, 1.0 - 1e-12)
    nll = -float(np.sum(targets * np.log(p) + (1.0 - targets) * np.log(1.0 - p)))
    return nll + 0.5 * RIDGE * float(np.sum((weights - prior) ** 2))
