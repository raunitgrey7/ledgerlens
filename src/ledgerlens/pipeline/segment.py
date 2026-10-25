"""Connected components to glyphs, glyphs to text lines, lines to words.

Lines are formed by vertical overlap rather than by a projection profile so that
multi-column headers keep their rows separate. Diacritics and the pieces of ':' or
'%' are merged by horizontal overlap inside a line. Word breaks use a per-line gap
threshold: the gaps inside a line are bimodal (letter gaps, word gaps) and the
largest relative jump in the sorted gap list marks the boundary, which keeps
monospace and proportional fonts on the same code path.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

MIN_AREA = 4
RULE_ASPECT = 8.0
RULE_THICKNESS = 5
BAND_SLACK = 0.3
TINY_SLACK = 0.7
WORD_GAP_RATIO = 0.34
SEGMENT_GAP = 2.5
LINE_GAP = 2.5


@dataclass
class Glyph:
    """One connected (possibly merged) ink blob, with its greyscale crop once a recogniser attaches it."""

    x0: int
    y0: int
    x1: int
    y1: int
    mask: np.ndarray
    gray: np.ndarray | None = None

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0


@dataclass
class Rule:
    """A long thin component: a table rule or a box edge, kept for layout analysis."""

    x0: int
    y0: int
    x1: int
    y1: int
    horizontal: bool


@dataclass
class TextLine:
    """Glyphs sharing a baseline band, with the metrics the recogniser needs."""

    glyphs: list[Glyph] = field(default_factory=list)
    y0: int = 0
    y1: int = 0
    baseline: float = 0.0
    cap_height: float = 1.0
    words: list[list[Glyph]] = field(default_factory=list)
    x1: int = 0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


def extract_components(ink: np.ndarray) -> tuple[list[Glyph], list[Rule]]:
    """Label the ink mask; split blobs into glyph candidates and ruling lines."""
    labels, _count = ndimage.label(ink, structure=np.ones((3, 3), dtype=int))
    glyphs: list[Glyph] = []
    rules: list[Rule] = []
    for index, slices in enumerate(ndimage.find_objects(labels), start=1):
        if slices is None:
            continue
        ys, xs = slices
        mask = labels[ys, xs] == index
        area = int(mask.sum())
        if area < MIN_AREA:
            continue
        width, height = xs.stop - xs.start, ys.stop - ys.start
        if (width >= RULE_ASPECT * height and height <= RULE_THICKNESS) or (width > 200 and height <= RULE_THICKNESS):
            rules.append(Rule(xs.start, ys.start, xs.stop, ys.stop, True))
        elif (height >= RULE_ASPECT * width and width <= RULE_THICKNESS) or (height > 200 and width <= RULE_THICKNESS):
            rules.append(Rule(xs.start, ys.start, xs.stop, ys.stop, False))
        elif width > 120 and height > 120 and area < 0.25 * width * height:
            rules.extend(_box_edges(xs.start, ys.start, xs.stop, ys.stop))
        elif width > 140 and height > 80:
            continue
        else:
            glyphs.append(Glyph(xs.start, ys.start, xs.stop, ys.stop, mask))
    return glyphs, rules


def _box_edges(x0: int, y0: int, x1: int, y1: int) -> list[Rule]:
    return [
        Rule(x0, y0, x1, y0 + 1, True),
        Rule(x0, y1 - 1, x1, y1, True),
        Rule(x0, y0, x0 + 1, y1, False),
        Rule(x1 - 1, y0, x1, y1, False),
    ]


def _overlap(glyph: Glyph, line: TextLine) -> float:
    """Fraction of the glyph inside the line band widened by a slack; dots and commas get more slack."""
    tiny = glyph.height < 0.5 * line.height
    slack = (TINY_SLACK if tiny else BAND_SLACK) * max(line.height, glyph.height)
    top, bottom = max(glyph.y0, line.y0 - slack), min(glyph.y1, line.y1 + slack)
    return (bottom - top) / max(1.0, float(glyph.height))


def group_lines(glyphs: list[Glyph]) -> list[TextLine]:
    """Assign glyphs to lines by vertical band overlap and horizontal proximity, scanning left to right.

    Proximity keeps a tall title from swallowing the small lines printed beside it:
    a glyph may only extend a line whose right edge is within LINE_GAP heights.
    """
    lines: list[TextLine] = []
    for glyph in sorted(glyphs, key=lambda g: (g.x0, -g.height)):
        best, best_overlap = None, 0.5
        for line in lines:
            if glyph.x0 - line.x1 > LINE_GAP * max(line.height, glyph.height, 6):
                continue
            overlap = _overlap(glyph, line)
            if overlap > best_overlap and _compatible(glyph, line):
                best, best_overlap = line, overlap
        if best is None:
            best = TextLine([], glyph.y0, glyph.y1, x1=glyph.x1)
            lines.append(best)
        best.glyphs.append(glyph)
        best.x1 = max(best.x1, glyph.x1)
        if glyph.height >= 0.5 * best.height or glyph.height >= 6:
            best.y0, best.y1 = min(best.y0, glyph.y0), max(best.y1, glyph.y1)
    return [line for line in lines if line.glyphs]


def _compatible(glyph: Glyph, line: TextLine) -> bool:
    """Reject a glyph far taller than the line it would join; small glyphs (punctuation) always may."""
    return glyph.height <= 2.2 * max(line.height, 6) or line.height < 6


def merge_vertical_pieces(line: TextLine) -> None:
    """Merge glyphs that overlap horizontally (dots over i/j, ':' and '%' pieces)."""
    glyphs = sorted(line.glyphs, key=lambda g: g.x0)
    merged: list[Glyph] = []
    for glyph in glyphs:
        if merged:
            last = merged[-1]
            overlap = min(last.x1, glyph.x1) - max(last.x0, glyph.x0)
            if overlap > 0.5 * min(last.width, glyph.width):
                merged[-1] = _union(last, glyph)
                continue
        merged.append(glyph)
    line.glyphs = merged


def _union(a: Glyph, b: Glyph) -> Glyph:
    x0, y0, x1, y1 = min(a.x0, b.x0), min(a.y0, b.y0), max(a.x1, b.x1), max(a.y1, b.y1)
    mask = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    mask[a.y0 - y0 : a.y1 - y0, a.x0 - x0 : a.x1 - x0] |= a.mask
    mask[b.y0 - y0 : b.y1 - y0, b.x0 - x0 : b.x1 - x0] |= b.mask
    return Glyph(x0, y0, x1, y1, mask)


def compute_metrics(line: TextLine) -> None:
    """Baseline from the dominant bottom edge, cap height from the upper part of the height distribution.

    The 80th percentile of non-punctuation heights lands on capitals, digits and ascenders
    in ordinary mixed-case text and ignores a merged i-dot or a bracket that overshoots.
    """
    heights = np.array([g.height for g in line.glyphs], dtype=float)
    body = heights >= 0.3 * heights.max()
    line.cap_height = float(np.percentile(heights[body], 80)) if body.any() else float(heights.max())
    bottoms = np.array([g.y1 for g in line.glyphs], dtype=float)
    base_glyphs = heights >= 0.5 * line.cap_height
    line.baseline = float(np.median(bottoms[base_glyphs])) if base_glyphs.any() else float(np.median(bottoms))


def drop_rule_fragments(line: TextLine) -> None:
    """Remove flat slivers sitting on or under the baseline: pieces of a deskewed ruling line.

    A hyphen is just as flat but sits mid x-height, so vertical position separates them.
    """
    cap, base = line.cap_height, line.baseline
    kept = [
        g
        for g in line.glyphs
        if not (g.height <= 3 and g.width >= 2 * g.height and (g.cy > base - 0.15 * cap or g.cy < base - 1.05 * cap))
    ]
    if kept:
        line.glyphs = kept


def split_words(line: TextLine) -> None:
    """Break the glyph sequence at word gaps.

    Monospace lines are detected from the regular pitch of glyph centres and split where
    the centre distance exceeds 1.5 cells; proportional lines split at ink gaps wider
    than a fraction of the cap height.
    """
    glyphs = sorted(line.glyphs, key=lambda g: g.x0)
    if len(glyphs) == 1:
        line.words = [glyphs]
        return
    gaps = np.array([glyphs[i + 1].x0 - glyphs[i].x1 for i in range(len(glyphs) - 1)], dtype=float)
    centres = np.array([(g.x0 + g.x1) / 2.0 for g in glyphs])
    pitch = monospace_pitch(np.diff(centres), line.cap_height)
    breaks = np.diff(centres) > 1.5 * pitch if pitch is not None else gaps > WORD_GAP_RATIO * line.cap_height
    words: list[list[Glyph]] = [[glyphs[0]]]
    for is_break, glyph in zip(breaks, glyphs[1:], strict=True):
        if is_break:
            words.append([glyph])
        else:
            words[-1].append(glyph)
    line.words = words


def monospace_pitch(distances: np.ndarray, cap_height: float) -> float | None:
    """Cell width when centre distances sit on a lattice of 1, 2 or 3 cells; else None."""
    if len(distances) < 4:
        return None
    base = distances[distances <= np.percentile(distances, 60)]
    pitch = float(np.median(base))
    if not 0.55 * cap_height <= pitch <= 1.2 * cap_height:
        return None
    multiples = np.round(distances / pitch)
    on_lattice = (np.abs(distances - multiples * pitch) <= 0.15 * pitch) & (multiples >= 1) & (multiples <= 3)
    return pitch if on_lattice.mean() >= 0.9 else None


def split_segments(line: TextLine) -> list[TextLine]:
    """Cut a page-wide row into block-local segments at gaps wider than SEGMENT_GAP cap heights.

    A bold vendor name and a small meta line can share a baseline; separate segments give
    each its own cap height and baseline so the recogniser sees consistent geometry.
    """
    glyphs = sorted(line.glyphs, key=lambda g: g.x0)
    segments: list[list[Glyph]] = [[glyphs[0]]]
    for previous, glyph in itertools.pairwise(glyphs):
        if glyph.x0 - previous.x1 > SEGMENT_GAP * line.cap_height:
            segments.append([glyph])
        else:
            segments[-1].append(glyph)
    result = []
    for group in segments:
        piece = TextLine(group, min(g.y0 for g in group), max(g.y1 for g in group))
        compute_metrics(piece)
        result.append(piece)
    return result


def segment(ink: np.ndarray) -> tuple[list[TextLine], list[Rule]]:
    """Full segmentation: components, rows, merged glyphs, block-local segments, metrics, words."""
    glyphs, rules = extract_components(ink)
    lines: list[TextLine] = []
    for row in group_lines(glyphs):
        merge_vertical_pieces(row)
        compute_metrics(row)
        drop_rule_fragments(row)
        for piece in split_segments(row):
            split_words(piece)
            lines.append(piece)
    lines.sort(key=lambda line: (line.y0, line.glyphs[0].x0))
    return lines, rules
