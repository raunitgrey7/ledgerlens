"""A drawing surface that records the ink box of every word it writes.

Templates only ever call ``Canvas.text`` and the ruling helpers, so the word-level
ground truth comes for free and stays exact: boxes are measured from the same font
metrics Pillow uses to rasterise, not estimated afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from PIL import Image, ImageDraw, ImageFont

from ledgerlens.schemas import WordBox

Color = tuple[int, int, int]


@dataclass
class Canvas:
    """RGB page plus the list of word boxes drawn so far."""

    width: int
    height: int
    background: Color = (255, 255, 255)
    words: list[WordBox] = field(default_factory=list)
    image: Image.Image = field(init=False)
    draw: ImageDraw.ImageDraw = field(init=False)
    _line_id: int = 0

    def __post_init__(self) -> None:
        self.image = Image.new("RGB", (self.width, self.height), self.background)
        self.draw = ImageDraw.Draw(self.image)

    def text(
        self,
        x: float,
        baseline: float,
        content: str,
        font: ImageFont.FreeTypeFont,
        fill: Color = (0, 0, 0),
        align: str = "left",
    ) -> float:
        """Draw one line of text at a baseline; returns the x coordinate of its right edge."""
        content = " ".join(content.split())
        if not content:
            return x
        total = font.getlength(content)
        if align == "right":
            x -= total
        elif align == "center":
            x -= total / 2
        self.draw.text((x, baseline), content, font=font, fill=fill, anchor="ls")
        self._record_words(x, baseline, content, font)
        self._line_id += 1
        return x + total

    def _record_words(self, x: float, baseline: float, content: str, font: ImageFont.FreeTypeFont) -> None:
        tokens = content.split(" ")
        for index, word in enumerate(tokens):
            prefix = " ".join(tokens[: index + 1])
            box = self.draw.textbbox((x, baseline), word, font=font, anchor="ls")
            shift = font.getlength(prefix) - font.getlength(word)
            self.words.append(
                WordBox(
                    text=word,
                    x0=int(box[0] + shift),
                    y0=int(box[1]),
                    x1=int(box[2] + shift) + 1,
                    y1=int(box[3]) + 1,
                    line_id=self._line_id,
                )
            )

    def lines(
        self,
        x: float,
        baseline: float,
        rows: list[str],
        font: ImageFont.FreeTypeFont,
        leading: float,
        fill: Color = (0, 0, 0),
        align: str = "left",
    ) -> float:
        """Draw several lines; returns the baseline following the last one."""
        for row in rows:
            self.text(x, baseline, row, font, fill, align)
            baseline += leading
        return baseline

    def hline(self, x0: float, x1: float, y: float, width: int = 1, fill: Color = (0, 0, 0)) -> None:
        self.draw.line([(x0, y), (x1, y)], fill=fill, width=width)

    def vline(self, x: float, y0: float, y1: float, width: int = 1, fill: Color = (0, 0, 0)) -> None:
        self.draw.line([(x, y0), (x, y1)], fill=fill, width=width)

    def rect(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        fill: Color | None = None,
        outline: Color | None = None,
        width: int = 1,
    ) -> None:
        self.draw.rectangle([x0, y0, x1, y1], fill=fill, outline=outline, width=width)

    def dashes(self, x0: float, x1: float, y: float, font: ImageFont.FreeTypeFont, fill: Color = (0, 0, 0)) -> None:
        """A separator made of hyphen glyphs, as thermal printers produce; not recorded as words."""
        unit = max(font.getlength("-"), 1.0)
        count = int((x1 - x0) / unit)
        self.draw.text((x0, y), "-" * count, font=font, fill=fill, anchor="ls")
