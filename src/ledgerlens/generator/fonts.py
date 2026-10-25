"""Font registry: which font files the generator may use and which the OCR atlas has seen.

The split between "seen" and "unseen" fonts is the honesty lever of the benchmark.
The glyph-atlas OCR is trained on the seen families, so pages rendered in an unseen
face measure how far the engine generalises instead of how well it memorises.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path

from PIL import ImageFont

FONT_DIR: Path = Path(str(resources.files("ledgerlens").joinpath("assets", "fonts")))


@dataclass(frozen=True)
class FontFamily:
    """A regular/bold pair with the capabilities the templates need to know about."""

    name: str
    regular: str | None
    bold: str | None
    mono: bool
    seen: bool
    has_rupee: bool

    def path(self, bold: bool) -> str | None:
        """File path for the requested weight; None means Pillow's embedded Aileron."""
        if bold and self.bold is not None:
            return self.bold
        return self.regular


SEEN_FAMILIES: tuple[FontFamily, ...] = (
    FontFamily("DejaVuSans", "DejaVuSans.ttf", "DejaVuSans-Bold.ttf", False, True, True),
    FontFamily("DejaVuSerif", "DejaVuSerif.ttf", "DejaVuSerif-Bold.ttf", False, True, True),
    FontFamily("DejaVuSansMono", "DejaVuSansMono.ttf", "DejaVuSansMono-Bold.ttf", True, True, True),
    FontFamily("STIXGeneral", "STIXGeneral.ttf", "STIXGeneralBol.ttf", False, True, False),
)
UNSEEN_FAMILIES: tuple[FontFamily, ...] = (
    FontFamily("Aileron", None, None, False, False, False),
    FontFamily("DejaVuSerifItalic", "DejaVuSerif-Italic.ttf", "DejaVuSerif-Italic.ttf", False, False, True),
)
ALL_FAMILIES: dict[str, FontFamily] = {f.name: f for f in SEEN_FAMILIES + UNSEEN_FAMILIES}


def atlas_font_files() -> list[tuple[str, str]]:
    """(label, path) for every weight of every seen family; the OCR atlas trains on these."""
    files: list[tuple[str, str]] = []
    for family in SEEN_FAMILIES:
        for weight, name in (("regular", family.regular), ("bold", family.bold)):
            if name is not None:
                files.append((f"{family.name}-{weight}", str(FONT_DIR / name)))
    return files


@lru_cache(maxsize=256)
def load_font(path: str | None, size: int) -> ImageFont.FreeTypeFont:
    """Load a TrueType font at ``size`` pixels, falling back to Pillow's embedded face."""
    if path is None:
        return ImageFont.load_default(size=size)  # type: ignore[return-value]
    full = path if Path(path).is_absolute() else str(FONT_DIR / path)
    return ImageFont.truetype(full, size=size)


def family_font(family: FontFamily, size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Convenience wrapper: a family at a size and weight."""
    return load_font(family.path(bold), size)
