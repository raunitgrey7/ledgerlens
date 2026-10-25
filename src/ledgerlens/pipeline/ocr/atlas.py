"""Glyph-atlas OCR: nearest-template recognition over fonts rendered by the generator.

No Tesseract binary is available on the build machine, so the package ships its own
engine. It renders every character of the seen font families at several sizes and
stroke widths into a normalised 20x20 bitmap plus three geometric features (height
relative to the line's cap height, descender depth, aspect ratio), and recognises a
segmented glyph by cosine similarity against that atlas. Three extra mechanisms carry
most of the accuracy: word-level case and digit consistency (an ambiguous 'O' inside a
numeric token becomes '0'), splitting of blobs that blur fused together, and a
logistic calibration that maps similarity and margin to a probability of being right.
The calibration is fitted on synthetic lines degraded with blur and noise, never on
benchmark pages.
"""

from __future__ import annotations

import difflib
import hashlib
import importlib
import json
import random
import string
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage

from ledgerlens.generator.fonts import atlas_font_files, load_font
from ledgerlens.pipeline.ocr.base import OCRLine, OCRPage, OCRWord
from ledgerlens.pipeline.preprocess import clean_impulses, sauvola
from ledgerlens.pipeline.segment import Glyph, TextLine, segment

CHARSET: str = string.digits + string.ascii_letters + ".,:;-/()#%&@*+='\"!?$" + "£€₹"
GLYPH_SIZE = 20
TEMPLATE_SIZES: tuple[int, ...] = (16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 28, 31, 34, 38, 44, 52)
STROKE_VARIANTS: tuple[float, ...] = (0.0, 1.0)
PHASE_VARIANTS: tuple[float, ...] = (0.0, 0.5)
VECTOR_BLUR = 0.6
GEO_WEIGHTS = np.array([0.9, 0.9, 0.35], dtype=np.float32)
AMBIGUITY_MARGIN = 0.12
CONDENSE_SIMILARITY = 0.985
SPLIT_WIDE_RATIO = 1.05
SPLIT_WEAK_RATIO = 0.7
SPLIT_WEAK_SCORE = 0.92
SPLIT_ACCEPT_GAIN = 0.04
SPLIT_ACCEPT_FLOOR = 0.88
SPLIT_KEEP_SCORE = 0.95
SPLIT_MULTI_RATIO = 1.6
WIDE_LETTERS = frozenset("mw")
SPLIT_FINE_LIMIT = 8
SLIVERS = frozenset(".,:;'\"!|`()[]{}1lI")
LOGIT_LIMIT = 30.0
RARE_SYMBOLS = frozenset("#%@*+='\"!?$\u00a3\u20ac\u20b9")
RARE_PENALTY = 0.03
CURRENCY_SYMBOLS = frozenset("$\u00a3\u20ac\u20b9")
CURRENCY_PENALTY = 0.07
ATLAS_VERSION = "8"
CALIBRATION_SOURCES: tuple[str, ...] = (
    "ledgerlens.pipeline.ocr.atlas",
    "ledgerlens.pipeline.segment",
    "ledgerlens.pipeline.preprocess",
)
DIGITS, UPPER, LOWER = set(string.digits), set(string.ascii_uppercase), set(string.ascii_lowercase)


@dataclass
class Atlas:
    """Template vectors, their geometry, labels and the confidence calibration."""

    vectors: np.ndarray
    geo: np.ndarray
    labels: np.ndarray
    calibration: tuple[float, float, float]
    fingerprint: str

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            vectors=self.vectors,
            geo=self.geo,
            labels=self.labels,
            calibration=np.array(self.calibration, dtype=np.float64),
            fingerprint=self.fingerprint,
        )

    @classmethod
    def load(cls, path: Path) -> Atlas:
        with np.load(path, allow_pickle=False) as data:
            return cls(
                data["vectors"],
                data["geo"],
                data["labels"],
                tuple(float(v) for v in data["calibration"]),
                str(data["fingerprint"]),
            )


def ink_intensity(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Greyscale crop as ink intensity in [0, 1], normalised between local paper and ink levels.

    Grey values keep the information binarisation throws away: a blurred bowl that
    closes in the binary mask is still lighter than the strokes around it.
    """
    paper = float(np.percentile(gray, 90))
    ink = float(np.percentile(gray[mask], 10)) if mask.any() else float(gray.min())
    span = max(paper - ink, 8.0)
    intensity = np.clip((paper - gray.astype(np.float32)) / span, 0.0, 1.0)
    support = ndimage.binary_dilation(mask, iterations=1)
    return intensity * support


def glyph_vector(mask: np.ndarray, gray: np.ndarray | None = None) -> np.ndarray:
    """Square-pad the crop, box-resize to GLYPH_SIZE, blur slightly, L2-normalise.

    Uses the normalised grey crop when available and the binary mask otherwise. The
    blur makes the cosine similarity tolerant to the one-pixel edge shifts that
    sub-pixel glyph positioning and thresholding introduce.
    """
    source = ink_intensity(gray, mask) if gray is not None else mask.astype(np.float32)
    height, width = source.shape
    side = max(height, width)
    square = np.zeros((side, side), dtype=np.float32)
    top, left = (side - height) // 2, (side - width) // 2
    square[top : top + height, left : left + width] = source
    resized = Image.fromarray(square, mode="F").resize((GLYPH_SIZE, GLYPH_SIZE), Image.Resampling.BOX)
    smoothed = ndimage.gaussian_filter(np.asarray(resized, dtype=np.float32), VECTOR_BLUR)
    vector = smoothed.ravel()
    return vector / (np.linalg.norm(vector) + 1e-6)


def geo_features(y0: int, y1: int, width: int, baseline: float, cap_height: float) -> np.ndarray:
    """Relative height, descender depth below the baseline, and aspect ratio."""
    cap = max(cap_height, 1.0)
    return np.array([(y1 - y0) / cap, (y1 - baseline) / cap, width / max(y1 - y0, 1)], dtype=np.float32)


def _render_char(
    font_path: str, size: int, ch: str, blur: float, phase: float = 0.0
) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Binary and grey crops of one character, binarised exactly like a page, plus the baseline row.

    Using the page's own Sauvola threshold on the template render keeps stroke weights
    consistent between atlas and input; the blur variants mimic defocused captures.
    """
    font = load_font(font_path, size)
    canvas = Image.new("L", (size * 4, size * 3), 255)
    baseline = size * 2
    ImageDraw.Draw(canvas).text((size + phase, baseline), ch, font=font, fill=0, anchor="ls")
    if blur > 0:
        canvas = canvas.filter(ImageFilter.GaussianBlur(blur))
    gray = np.asarray(canvas)
    ink = clean_impulses(sauvola(gray))
    rows, cols = np.nonzero(ink)
    if rows.size == 0:
        return None
    y0, y1, x0, x1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    return ink[y0:y1, x0:x1], gray[y0:y1, x0:x1], baseline - y0


def _cap_height(font_path: str, size: int) -> float:
    rendered = _render_char(font_path, size, "0", 0.0)
    return float(rendered[0].shape[0]) if rendered else float(size) * 0.7


def render_templates() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every charset character in every seen font, size, stroke and phase variant, condensed and label-sorted."""
    vectors, geos, labels = [], [], []
    for _, path in atlas_font_files():
        for size in TEMPLATE_SIZES:
            cap = _cap_height(path, size)
            for ch in CHARSET:
                for blur in STROKE_VARIANTS:
                    for phase in PHASE_VARIANTS:
                        rendered = _render_char(path, size, ch, blur, phase)
                        if rendered is None:
                            continue
                        mask, gray, baseline_row = rendered
                        vectors.append(glyph_vector(mask, gray))
                        geos.append(geo_features(0, mask.shape[0], mask.shape[1], baseline_row, cap))
                        labels.append(ch)
    vectors_arr, geos_arr, labels_arr = condense(np.stack(vectors), np.stack(geos), np.array(labels))
    order = np.argsort(labels_arr, kind="stable")
    return vectors_arr[order], geos_arr[order], labels_arr[order]


def condense(vectors: np.ndarray, geos: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop templates nearly identical (same label, cosine above CONDENSE_SIMILARITY, close geometry)."""
    keep = np.zeros(len(labels), dtype=bool)
    for label in np.unique(labels):
        indices = np.nonzero(labels == label)[0]
        kept: list[int] = []
        for index in indices:
            if kept:
                sims = vectors[kept] @ vectors[index]
                close = (sims >= CONDENSE_SIMILARITY) & (np.abs(geos[kept] - geos[index]).max(axis=1) < 0.05)
                if close.any():
                    continue
            kept.append(int(index))
        keep[kept] = True
    return vectors[keep], geos[keep], labels[keep]


def atlas_fingerprint() -> str:
    """Identifies the atlas inputs so a stale cache is rebuilt automatically.

    The character calibration is fitted by running the recogniser, segmenter and
    preprocessor, so their source is part of the fingerprint alongside the fonts and
    template parameters: any change to that code refits the calibration, and a fresh
    clone and a long-lived checkout always use the same atlas.
    """
    digest = hashlib.sha256()
    digest.update(ATLAS_VERSION.encode())
    digest.update(CHARSET.encode())
    digest.update(str(TEMPLATE_SIZES).encode())
    digest.update(str(STROKE_VARIANTS).encode())
    for label, path in atlas_font_files():
        digest.update(label.encode())
        digest.update(Path(path).read_bytes())
    for module in CALIBRATION_SOURCES:
        digest.update(Path(str(importlib.import_module(module).__file__)).read_bytes())
    return digest.hexdigest()[:16]


class AtlasOCR:
    """The shipped engine: segmentation from ``pipeline.segment`` plus atlas matching."""

    name = "atlas"

    def __init__(self, atlas: Atlas) -> None:
        self.atlas = atlas
        self.block_labels, self.block_starts = _label_blocks(atlas.labels)
        self.block_prior = np.array([_prior(ch) for ch in self.block_labels], dtype=np.float32)

    @classmethod
    def load_or_build(cls, cache_dir: Path) -> AtlasOCR:
        """Use the cached atlas when its fingerprint matches, otherwise render, calibrate and drop stale ones."""
        cache_dir.mkdir(parents=True, exist_ok=True)
        fingerprint = atlas_fingerprint()
        path = cache_dir / f"atlas-{fingerprint}.npz"
        if path.exists():
            return cls(Atlas.load(path))
        vectors, geo, labels = render_templates()
        atlas = Atlas(vectors, geo, labels, (0.0, 1.0, 1.0), fingerprint)
        engine = cls(atlas)
        atlas.calibration = fit_calibration(engine)
        atlas.save(path)
        (cache_dir / f"atlas-{fingerprint}.json").write_text(
            json.dumps(
                {
                    "templates": len(labels),
                    "fonts": [label for label, _ in atlas_font_files()],
                    "calibration": list(atlas.calibration),
                },
                indent=2,
            )
        )
        for stale in cache_dir.glob("atlas-*"):
            if fingerprint not in stale.name:
                stale.unlink()
        return engine

    def recognize(self, gray: np.ndarray, ink: np.ndarray) -> OCRPage:
        lines, _ = segment(ink)
        out: list[OCRLine] = []
        for line in lines:
            for glyph in line.glyphs:
                glyph.gray = gray[glyph.y0 : glyph.y1, glyph.x0 : glyph.x1]
            words = [self._recognize_word(word, line) for word in line.words]
            words = [word for word in words if word.text]
            if not words:
                continue
            out.append(
                OCRLine(
                    words,
                    min(w.x0 for w in words),
                    min(w.y0 for w in words),
                    max(w.x1 for w in words),
                    max(w.y1 for w in words),
                    line.baseline,
                    line.cap_height,
                )
            )
        return OCRPage(out, ink.shape[1], ink.shape[0], self.name)

    def score_glyphs(self, glyphs: list[Glyph], line: TextLine) -> np.ndarray:
        """Best score per label for each glyph: a (glyphs x labels) matrix."""
        vectors = np.stack([glyph_vector(g.mask, g.gray) for g in glyphs])
        geo = np.stack([geo_features(g.y0, g.y1, g.width, line.baseline, line.cap_height) for g in glyphs])
        similarity = vectors @ self.atlas.vectors.T
        penalty = (np.abs(geo[:, None, :] - self.atlas.geo[None, :, :]) * GEO_WEIGHTS).sum(axis=2)
        return np.maximum.reduceat(similarity - penalty, self.block_starts, axis=1) - self.block_prior

    def _recognize_word(self, glyphs: list[Glyph], line: TextLine) -> OCRWord:
        glyphs = self._split_fused(glyphs, line)
        scores = self.score_glyphs(glyphs, line)
        candidates = [self._candidates(scores[i]) for i in range(len(glyphs))]
        classes = _class_path(candidates)
        chars, confidences, raw = [], [], []
        for options, wanted in zip(candidates, classes, strict=True):
            label, score, margin = _choose(options, wanted)
            chars.append(label)
            raw.append((score, margin))
            confidences.append(self._probability(score, margin))
        return OCRWord(
            "".join(chars),
            float(np.prod(confidences)),
            min(g.x0 for g in glyphs),
            min(g.y0 for g in glyphs),
            max(g.x1 for g in glyphs),
            max(g.y1 for g in glyphs),
            confidences,
            raw,
        )

    def _candidates(self, scores: np.ndarray) -> list[tuple[str, float]]:
        """Labels ordered by score, best first."""
        order = np.argsort(-scores)[:8]
        return [(str(self.block_labels[index]), float(scores[index])) for index in order]

    def _probability(self, score: float, margin: float) -> float:
        bias, w_score, w_margin = self.atlas.calibration
        logit = float(np.clip(bias + w_score * score + w_margin * margin, -LOGIT_LIMIT, LOGIT_LIMIT))
        return float(1.0 / (1.0 + np.exp(-logit)))

    def _split_fused(self, glyphs: list[Glyph], line: TextLine) -> list[Glyph]:
        """Cut blobs that blur or touching serifs fused, at the cut that scores best.

        A blob is a split candidate when it is wider than a cap height, or merely wide-ish
        but poorly matched. When the whole blob reads best as a wide letter and is narrower
        than 1.6 cap heights, the halves must beat it by a margin (a lowercase m must not become
        r + n); other wide blobs are split on a near tie. Cuts
        whose halves are all thin slivers, which is what a cut through an oval gives, are refused.
        """
        result: list[Glyph] = []
        for glyph in glyphs:
            result.extend(self._split_one(glyph, line, depth=0))
        return result

    def _split_one(self, glyph: Glyph, line: TextLine, depth: int) -> list[Glyph]:
        cap = line.cap_height
        if depth >= 3 or glyph.height < 4 or glyph.width < SPLIT_WEAK_RATIO * cap:
            return [glyph]
        whole = self.score_glyphs([glyph], line)[0]
        best = float(whole.max())
        wide = glyph.width >= SPLIT_WIDE_RATIO * cap
        if best >= SPLIT_KEEP_SCORE or (not wide and best >= SPLIT_WEAK_SCORE):
            return [glyph]
        found = self._best_split(glyph, line)
        if found is None:
            return [glyph]
        parts, part_best, labels = found
        if all(label in SLIVERS for label in labels):
            return [glyph]
        wide_letter = str(self.block_labels[int(whole.argmax())]) in WIDE_LETTERS
        lenient = wide and (glyph.width >= SPLIT_MULTI_RATIO * cap or not wide_letter)
        needed = best - 0.02 if lenient else max(best + SPLIT_ACCEPT_GAIN, SPLIT_ACCEPT_FLOOR)
        if part_best < needed:
            return [glyph]
        return [piece for part in parts for piece in self._split_one(part, line, depth + 1)]

    def _best_split(self, glyph: Glyph, line: TextLine) -> tuple[list[Glyph], float, list[str]] | None:
        """Search cut columns in the central band; keep the cut whose weaker half scores best.

        The thinnest column is the wrong cut for hollow letters: inside a fused "02" it
        falls in the middle of the 0. Scoring candidate halves in one batch finds the
        junction instead. Wide blobs are searched on every second column, then refined
        around the winner, which halves the cost with the same result on a smooth score.
        """
        width = glyph.width
        lo, hi = max(2, int(width * 0.2)), min(width - 2, int(np.ceil(width * 0.8)))
        step = 2 if hi - lo > SPLIT_FINE_LIMIT else 1
        found = self._score_cuts(glyph, line, list(range(lo, hi + 1, step)))
        if found is not None and step == 2:
            refined = self._score_cuts(glyph, line, [c for c in (found[0] - 1, found[0] + 1) if lo <= c <= hi])
            if refined is not None and refined[2] > found[2]:
                found = refined
        return None if found is None else (found[1], found[2], found[3])

    def _score_cuts(
        self, glyph: Glyph, line: TextLine, cuts: list[int]
    ) -> tuple[int, list[Glyph], float, list[str]] | None:
        """Best of the given cuts as (cut, halves, weaker half's score, labels of the halves)."""
        pairs = []
        for cut in cuts:
            left, right = _sub_glyph(glyph, 0, cut), _sub_glyph(glyph, cut, glyph.width)
            if left is not None and right is not None:
                pairs.append((cut, left, right))
        if not pairs:
            return None
        scores = self.score_glyphs([part for _, left, right in pairs for part in (left, right)], line)
        weakest = scores.max(axis=1).reshape(-1, 2).min(axis=1)
        index = int(np.argmax(weakest))
        labels = [str(self.block_labels[int(i)]) for i in scores[2 * index : 2 * index + 2].argmax(axis=1)]
        cut, left, right = pairs[index]
        return cut, [left, right], float(weakest[index]), labels


def _sub_glyph(glyph: Glyph, x_from: int, x_to: int) -> Glyph | None:
    mask = glyph.mask[:, x_from:x_to]
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return None
    y0, y1, x0, x1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    gray = None if glyph.gray is None else glyph.gray[y0:y1, x_from + x0 : x_from + x1]
    return Glyph(glyph.x0 + x_from + x0, glyph.y0 + y0, glyph.x0 + x_from + x1, glyph.y0 + y1, mask[y0:y1, x0:x1], gray)


def _prior(ch: str) -> float:
    """Score penalty for rare symbols; currency glyphs get more because they mimic digits."""
    if ch in CURRENCY_SYMBOLS:
        return CURRENCY_PENALTY
    return RARE_PENALTY if ch in RARE_SYMBOLS else 0.0


def _label_blocks(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distinct labels of a label-sorted atlas and the start index of each block."""
    change = np.ones(len(labels), dtype=bool)
    change[1:] = labels[1:] != labels[:-1]
    starts = np.nonzero(change)[0]
    return labels[starts], starts


def _char_class(ch: str) -> str:
    if ch in DIGITS:
        return "digit"
    if ch in UPPER:
        return "upper"
    if ch in LOWER:
        return "lower"
    return "other"


CLASSES: tuple[str, ...] = ("digit", "upper", "lower", "other")
SWITCH_COST = 0.05


def _class_path(candidates: list[list[tuple[str, float]]]) -> list[str]:
    """Viterbi over character classes: runs of digits or letters are cheaper than alternation.

    Each position emits the best score among its candidates of a class (within the
    ambiguity margin of the best overall), and switching between digit, upper and lower
    costs SWITCH_COST. Punctuation ("other") switches for free.
    """
    emissions = []
    for options in candidates:
        best = options[0][1]
        row = dict.fromkeys(CLASSES, -1.0)
        for label, score in options:
            if best - score > AMBIGUITY_MARGIN:
                break
            cls = _char_class(label)
            row[cls] = max(row[cls], score)
        emissions.append(row)
    cost: dict[str, float] = {cls: emissions[0][cls] for cls in CLASSES}
    back: list[dict[str, str]] = []
    for position, row in enumerate(emissions[1:], start=1):
        new_cost, pointers = {}, {}
        for cls in CLASSES:
            best_prev = max(CLASSES, key=lambda prev: cost[prev] - _switch(prev, cls, position))
            new_cost[cls] = cost[best_prev] - _switch(best_prev, cls, position) + row[cls]
            pointers[cls] = best_prev
        cost, back = new_cost, [*back, pointers]
    current = max(CLASSES, key=lambda cls: cost[cls])
    path = [current]
    for pointers in reversed(back):
        current = pointers[current]
        path.append(current)
    return list(reversed(path))


def _switch(previous: str, current: str, position: int) -> float:
    """Class change cost; a capital followed by lower case at the word start is free (Invoice, Ltd)."""
    if previous == current or "other" in (previous, current):
        return 0.0
    if position == 1 and previous == "upper" and current == "lower":
        return 0.0
    return SWITCH_COST


def _choose(options: list[tuple[str, float]], wanted: str) -> tuple[str, float, float]:
    """Best candidate of the chosen class within the margin, else the best overall."""
    best_label, best_score = options[0]
    for label, score in options:
        if best_score - score > AMBIGUITY_MARGIN:
            break
        if _char_class(label) == wanted:
            best_label, best_score = label, score
            break
    runner_up = max((score for label, score in options if label != best_label), default=best_score - 1.0)
    return best_label, best_score, best_score - runner_up


CALIBRATION_WORDS: tuple[str, ...] = (
    "Invoice",
    "Total",
    "1,234.56",
    "GSTIN",
    "07ABCDE1234F1Z5",
    "2026-03-14",
    "Subtotal",
    "INR",
    "Qty",
    "Meridian",
    "Logistics",
    "Pvt",
    "Ltd",
    "GB82WEST12345698765432",
    "14/03/2026",
    "Amount",
    "Due",
    "Balance",
    "3,45,678.00",
    "SALARY",
    "CREDIT",
    "Rate",
    "Bill",
    "To:",
    "Date:",
    "No:",
    "Grand",
    "Tax",
    "18%",
    "x",
    "2",
    "Cloud",
    "hosting",
    "(monthly)",
    "Copper",
    "wire",
    "2.5",
    "sqmm",
    "(100m)",
    "Thank",
    "you",
    "DE89 3704",
)


def _calibration_lines(seed: int) -> list[tuple[str, str, int, float, float]]:
    """(text, font path, size, blur, noise) for the calibration render; deterministic."""
    rng = random.Random(seed)
    fonts = [path for _, path in atlas_font_files()]
    lines = []
    for _ in range(160):
        text = " ".join(rng.choice(CALIBRATION_WORDS) for _ in range(rng.randint(2, 5)))
        lines.append(
            (text, rng.choice(fonts), rng.choice((19, 20, 22, 24)), rng.uniform(0.0, 1.2), rng.uniform(0.0, 14.0))
        )
    return lines


def _render_line(text: str, font_path: str, size: int, blur: float, noise: float, seed: int) -> np.ndarray:
    font = load_font(font_path, size)
    width = int(font.getlength(text)) + 2 * size
    canvas = Image.new("L", (width, size * 3), 245)
    ImageDraw.Draw(canvas).text((size, size * 2), text, font=font, fill=0, anchor="ls")
    if blur > 0:
        canvas = canvas.filter(ImageFilter.GaussianBlur(blur))
    array = np.asarray(canvas).astype(np.float32)
    if noise > 0:
        array = array + np.random.default_rng(seed).normal(0.0, noise, size=array.shape)
    return np.clip(array, 0, 255).astype(np.uint8)


def fit_calibration(engine: AtlasOCR, seed: int = 2024) -> tuple[float, float, float]:
    """Fit P(correct | score, margin) by logistic regression on aligned calibration lines."""
    features, targets = [], []
    for index, (text, path, size, blur, noise) in enumerate(_calibration_lines(seed)):
        gray = _render_line(text, path, size, blur, noise, seed + index)
        ink = clean_impulses(sauvola(gray))
        page = engine.recognize(gray, ink)
        predicted = "".join(word.text for line in page.lines for word in line.words)
        raw = [pair for line in page.lines for word in line.words for pair in word.char_scores]
        matcher = difflib.SequenceMatcher(a=predicted, b=text.replace(" ", ""), autojunk=False)
        correct = np.zeros(len(predicted), dtype=bool)
        for block in matcher.get_matching_blocks():
            correct[block.a : block.a + block.size] = True
        for feature, label in zip(raw, correct, strict=True):
            features.append(feature)
            targets.append(float(label))
    return _logistic_fit(np.array(features, dtype=np.float64), np.array(targets, dtype=np.float64))


def _logistic_fit(features: np.ndarray, targets: np.ndarray, iterations: int = 40) -> tuple[float, float, float]:
    """Newton's method for a two-feature logistic regression with light L2 regularisation."""
    design = np.column_stack([np.ones(len(features)), features])
    weights = np.zeros(design.shape[1])
    ridge = 1e-3 * np.eye(design.shape[1])
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-design @ weights))
        gradient = design.T @ (p - targets) + ridge @ weights
        hessian = (design * (p * (1 - p))[:, None]).T @ design + ridge
        weights = weights - np.linalg.solve(hessian, gradient)
    return float(weights[0]), float(weights[1]), float(weights[2])
