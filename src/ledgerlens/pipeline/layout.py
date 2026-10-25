"""Spatial reasoning over OCR output: rows, fuzzy labels, values to the right or below, tables.

Extraction never trusts exact strings. Labels are matched with an edit-distance ratio
so that "Tota1" still anchors a total, and values are located geometrically (same row
to the right, or the next row below for stacked layouts) because that is how humans
read a form. Tables are found from a header row whose words match column keywords and
whose x positions define the columns.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

from ledgerlens.pipeline.ocr.base import OCRLine, OCRPage, OCRWord

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
TEXT_COLUMNS = frozenset({"description", "date"})
SEPARATOR_CHARS = ":;-|.,'\"_~=*"
TEXT_COLUMN_MARGIN = 24.0
SHORT_KEYWORD_THRESHOLD = 0.6


def normalise(text: str) -> str:
    """Lower-case alphanumerics only, for label comparison."""
    return _NON_ALNUM.sub("", text.lower())


def similarity(a: str, b: str) -> float:
    """Edit-distance ratio between normalised strings."""
    return difflib.SequenceMatcher(a=normalise(a), b=normalise(b), autojunk=False).ratio()


@dataclass
class Row:
    """OCR lines that share a baseline band across the page, left to right."""

    lines: list[OCRLine]
    y0: int
    y1: int

    @property
    def words(self) -> list[OCRWord]:
        return sorted((w for line in self.lines for w in line.words), key=lambda w: w.x0)

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


def build_rows(page: OCRPage) -> list[Row]:
    """Merge block-local OCR lines into page-wide rows by vertical overlap."""
    rows: list[Row] = []
    for line in sorted(page.lines, key=lambda line: (line.y0, line.x0)):
        for row in rows:
            overlap = min(row.y1, line.y1) - max(row.y0, line.y0)
            if overlap > 0.5 * min(row.height, line.y1 - line.y0):
                row.lines.append(line)
                row.y0, row.y1 = min(row.y0, line.y0), max(row.y1, line.y1)
                break
        else:
            rows.append(Row([line], line.y0, line.y1))
    rows.sort(key=lambda row: row.y0)
    return rows


@dataclass
class LabelMatch:
    """Where a label was found and the words that follow it on the same row."""

    row_index: int
    words: list[OCRWord]
    score: float
    rest: list[OCRWord] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        return min((w.confidence for w in self.words), default=0.0)


def find_label(
    rows: list[Row], phrases: tuple[str, ...], threshold: float = 0.8, exclude: tuple[str, ...] = ()
) -> list[LabelMatch]:
    """All rows where a phrase appears as a run of words at any position.

    Matches are ordered best score first. An ``exclude`` phrase vetoes a match when a
    run of the exclude phrase's own length that overlaps the match fits it at least as
    well as the label fits ("Sub Total" is not a "Total"). Comparing the bare label with
    a longer phrase would wrongly veto "Invoice" because it resembles "Tax Invoice".
    """
    matches: list[LabelMatch] = []
    for index, row in enumerate(rows):
        words = row.words
        for phrase in phrases:
            length = len(phrase.split())
            for start in range(max(0, len(words) - length + 1)):
                run = words[start : start + length]
                score = similarity(" ".join(w.text for w in run), phrase)
                if score < threshold or _excluded(words, start, length, exclude, max(score, threshold)):
                    continue
                matches.append(LabelMatch(index, run, score, words[start + length :]))
    matches.sort(key=lambda m: (-m.score, m.row_index))
    return matches


def _excluded(words: list[OCRWord], start: int, length: int, exclude: tuple[str, ...], bar: float) -> bool:
    """True when some exclude phrase fits a same-length run of words that overlaps the match."""
    for phrase in exclude:
        size = len(phrase.split())
        for first in range(max(0, start - size + 1), min(start + length, len(words) - size + 1)):
            if similarity(" ".join(w.text for w in words[first : first + size]), phrase) >= bar:
                return True
    return False


def row_below(rows: list[Row], index: int, max_gap: float = 2.5) -> Row | None:
    """The next row whose top is within ``max_gap`` row heights below the given row."""
    if index + 1 >= len(rows):
        return None
    current, candidate = rows[index], rows[index + 1]
    if candidate.y0 - current.y1 <= max_gap * max(current.height, 8):
        return candidate
    return None


def words_below(rows: list[Row], index: int, x0: float, x1: float, max_gap: float = 2.5) -> list[OCRWord]:
    """Words of the next row that overlap the horizontal span [x0, x1] of a label."""
    below = row_below(rows, index, max_gap)
    if below is None:
        return []
    span = max(x1 - x0, 40.0)
    return [w for w in below.words if w.x1 > x0 - 0.5 * span and w.x0 < x1 + 1.5 * span]


def line_below(rows: list[Row], index: int, x0: float, x1: float, max_gap: float = 2.5) -> list[OCRWord]:
    """Words of the block-local line in the next row that starts under the label's span.

    Used for names and addresses, where the value is a whole line and a neighbouring
    block (a Ship To column) must not leak in.
    """
    below = row_below(rows, index, max_gap)
    if below is None:
        return []
    span = max(x1 - x0, 40.0)
    candidates = [line for line in below.lines if line.x1 > x0 - 0.5 * span and line.x0 < x1 + 1.5 * span]
    if not candidates:
        return []
    best = min(candidates, key=lambda line: abs(line.x0 - x0))
    return list(best.words)


def strip_separators(words: list[OCRWord]) -> list[OCRWord]:
    """Drop lone punctuation tokens such as ':' or '-' that follow a label; '&' is part of names."""
    return [w for w in words if w.text.strip(SEPARATOR_CHARS)]


@dataclass
class Column:
    """A table column named by its header word and bounded by x coordinates."""

    name: str
    x0: float
    x1: float
    header: OCRWord

    def contains(self, word: OCRWord) -> bool:
        return self.x0 <= word.cx < self.x1


@dataclass
class TableHeader:
    """Detected header row and the columns derived from it."""

    row_index: int
    columns: list[Column]

    def column_for(self, word: OCRWord, right_aligned: bool = False) -> Column | None:
        """The column holding a word: by its centre, or by its right edge for right-aligned values.

        Amounts are printed right-aligned under their header, so a wide value can start
        left of its column's boundary; its right edge never does.
        """
        x = word.x1 - 1 if right_aligned else word.cx
        for column in self.columns:
            if column.x0 <= x < column.x1:
                return column
        return None


def find_table_header(
    rows: list[Row], keywords: dict[str, tuple[str, ...]], page_width: int, minimum: int = 2, threshold: float = 0.8
) -> TableHeader | None:
    """First row with at least ``minimum`` distinct column keywords; columns span between headers.

    Boundaries come from ``_boundary``; the first column starts at the left page edge
    and the last runs to the right edge.
    """
    for index, row in enumerate(rows):
        found = _header_columns(row, keywords, threshold)
        if len(found) >= minimum:
            found.sort(key=lambda item: item[1].x0)
            columns = []
            for position, (name, word) in enumerate(found):
                left = 0.0 if position == 0 else columns[-1].x1
                right = (
                    float(page_width) if position == len(found) - 1 else _boundary(name, word, found[position + 1][1])
                )
                columns.append(Column(name, left, right, word))
            return TableHeader(index, columns)
    return None


def _boundary(name: str, word: OCRWord, next_word: OCRWord) -> float:
    """Right edge of a column: text columns run up to the next header, numeric ones split the gap.

    Numeric values are right-aligned under their headers, so the gap between two
    numeric headers is free space; a text column's values are left-aligned and may run
    almost up to the next header.
    """
    if name in TEXT_COLUMNS:
        return next_word.x0 - TEXT_COLUMN_MARGIN
    return (word.x1 + next_word.x0) / 2.0


def _header_columns(row: Row, keywords: dict[str, tuple[str, ...]], threshold: float) -> list[tuple[str, OCRWord]]:
    found: list[tuple[str, OCRWord]] = []
    taken: set[str] = set()
    words = row.words
    for position, word in enumerate(words):
        for name, phrases in keywords.items():
            if name in taken:
                continue
            for phrase in phrases:
                length = len(phrase.split())
                run = words[position : position + length]
                if len(run) == length and _keyword_matches(" ".join(w.text for w in run), phrase, threshold):
                    found.append((name, word))
                    taken.add(name)
                    break
    return found


def _keyword_matches(text: str, phrase: str, threshold: float) -> bool:
    """Fuzzy keyword test that tolerates one wrong character in short keywords of the same length.

    One misread letter costs a three-letter keyword like "Qty" a third of its similarity,
    so short keywords of equal length only need SHORT_KEYWORD_THRESHOLD.
    """
    a, b = normalise(text), normalise(phrase)
    limit = SHORT_KEYWORD_THRESHOLD if len(b) <= 4 and len(a) == len(b) else threshold
    return similarity(text, phrase) >= limit
