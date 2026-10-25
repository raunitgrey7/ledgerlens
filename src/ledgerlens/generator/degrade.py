"""Physically motivated page degradations: skew, defocus, sensor noise, dust, lighting, stamps.

Each step mimics a real capture defect. Rotation is feeder or hand-held skew; Gaussian
blur is defocus or a soft point spread function; additive Gaussian noise is sensor read
noise; salt-and-pepper is dust and dead pixels; the multiplicative low-frequency field is
uneven lighting or a vignetting lens; stamps are coloured ink pressed over content; JPEG
is the compression of a phone capture. Parameters are drawn per severity level and
recorded so that failures can be traced back to a cause.
"""

from __future__ import annotations

import io
import math
import random
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ledgerlens.generator.fonts import load_font
from ledgerlens.schemas import Degradation, Severity

STAMP_TEXTS: tuple[str, ...] = ("PAID", "RECEIVED", "APPROVED", "COPY", "POSTED", "VERIFIED")
STAMP_COLORS: tuple[tuple[int, int, int], ...] = ((200, 30, 30), (30, 60, 200), (120, 30, 160), (20, 120, 60))


@dataclass(frozen=True)
class SeverityProfile:
    """Parameter ranges for one severity level."""

    rotation: float
    blur: tuple[float, float]
    noise: tuple[float, float]
    salt_pepper: tuple[float, float]
    illumination: tuple[float, float]
    stamp_probability: float
    jpeg: tuple[int, int] | None
    jpeg_probability: float
    tint: tuple[int, int]


PROFILES: dict[Severity, SeverityProfile] = {
    Severity.CLEAN: SeverityProfile(0.0, (0.0, 0.0), (0.0, 1.5), (0.0, 0.0), (0.0, 0.0), 0.0, None, 0.0, (250, 255)),
    Severity.LIGHT: SeverityProfile(
        1.5, (0.3, 0.7), (3.0, 8.0), (0.0, 0.001), (0.03, 0.12), 0.3, (70, 90), 0.5, (238, 252)
    ),
    Severity.HEAVY: SeverityProfile(
        4.0, (0.7, 1.3), (8.0, 18.0), (0.001, 0.004), (0.10, 0.30), 0.6, (35, 65), 0.8, (225, 248)
    ),
}


def _uniform(rng: random.Random, bounds: tuple[float, float]) -> float:
    return rng.uniform(bounds[0], bounds[1])


def apply_stamp(image: Image.Image, rng: random.Random, severity: Severity) -> Image.Image:
    """Press a rotated, semi-transparent coloured stamp onto the page."""
    width, height = image.size
    size = int(min(width, height) * rng.uniform(0.09, 0.14))
    font = load_font("DejaVuSans-Bold.ttf", size)
    text = rng.choice(STAMP_TEXTS)
    pad = size // 2
    text_width = int(font.getlength(text))
    layer = Image.new("RGBA", (text_width + 2 * pad, size + 2 * pad), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    color = rng.choice(STAMP_COLORS)
    alpha = int(255 * rng.uniform(0.45, 0.8))
    draw.rounded_rectangle(
        [4, 4, layer.width - 5, layer.height - 5], radius=size // 4, outline=(*color, alpha), width=max(3, size // 12)
    )
    draw.text((layer.width / 2, layer.height / 2), text, font=font, fill=(*color, alpha), anchor="mm")
    layer = layer.rotate(rng.uniform(-30, 30), resample=Image.Resampling.BICUBIC, expand=True)
    if severity is Severity.HEAVY:
        x = rng.randint(0, max(1, width - layer.width))
        y = rng.randint(int(height * 0.25), max(int(height * 0.25) + 1, height - layer.height))
    else:
        x = rng.randint(int(width * 0.5), max(int(width * 0.5) + 1, width - layer.width))
        y = rng.randint(int(height * 0.55), max(int(height * 0.55) + 1, height - layer.height))
    base = image.convert("RGBA")
    base.alpha_composite(layer, dest=(x, y))
    return base.convert("RGB")


def apply_illumination(array: np.ndarray, rng: random.Random, strength: float) -> np.ndarray:
    """Multiply by a smooth field that darkens towards one side or corner (vignetting)."""
    height, width = array.shape[:2]
    cy, cx = rng.uniform(0.2, 0.8) * height, rng.uniform(0.2, 0.8) * width
    yy, xx = np.mgrid[0:height, 0:width]
    distance = np.sqrt(((yy - cy) / height) ** 2 + ((xx - cx) / width) ** 2)
    field = 1.0 - strength * (distance / distance.max())
    return array * field[:, :, None]


def apply_noise(array: np.ndarray, rng: random.Random, sigma: float, salt_pepper: float) -> np.ndarray:
    """Luma Gaussian noise plus a sprinkling of pure black and white pixels."""
    generator = np.random.default_rng(rng.getrandbits(32))
    if sigma > 0:
        array = array + generator.normal(0.0, sigma, size=array.shape[:2])[:, :, None]
    if salt_pepper > 0:
        mask = generator.random(array.shape[:2])
        array[mask < salt_pepper / 2] = 0.0
        array[mask > 1.0 - salt_pepper / 2] = 255.0
    return array


def recompress_jpeg(image: Image.Image, quality: int) -> Image.Image:
    """Round-trip through JPEG to add block and ringing artefacts."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def degrade(image: Image.Image, severity: Severity, rng: random.Random) -> tuple[Image.Image, Degradation]:
    """Apply the severity profile to a clean render; returns the page and what was done."""
    profile = PROFILES[severity]
    tint = rng.randint(*profile.tint)
    record = Degradation(paper_tint=tint)
    page = image.convert("RGB")
    if rng.random() < profile.stamp_probability:
        page = apply_stamp(page, rng, severity)
        record.stamp = True
    if profile.rotation > 0:
        record.rotation_deg = round(rng.uniform(-profile.rotation, profile.rotation), 3)
        page = page.rotate(record.rotation_deg, resample=Image.Resampling.BICUBIC, fillcolor=(255, 255, 255))
    record.blur_sigma = round(_uniform(rng, profile.blur), 3)
    if record.blur_sigma > 0:
        page = page.filter(ImageFilter.GaussianBlur(record.blur_sigma))
    array = np.asarray(page).astype(np.float32)
    if severity is Severity.HEAVY and rng.random() < 0.3:
        record.motion_px = rng.randint(3, 5)
        array = apply_motion_blur(array, motion_kernel(record.motion_px, rng.uniform(0.0, 180.0)))
    array = array * (tint / 255.0)
    record.illumination = round(_uniform(rng, profile.illumination), 3)
    if record.illumination > 0:
        array = apply_illumination(array, rng, record.illumination)
    record.noise_sigma = round(_uniform(rng, profile.noise), 3)
    record.salt_pepper = round(_uniform(rng, profile.salt_pepper), 5)
    array = apply_noise(array, rng, record.noise_sigma, record.salt_pepper)
    page = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGB")
    if profile.jpeg is not None and rng.random() < profile.jpeg_probability:
        record.jpeg_quality = rng.randint(*profile.jpeg)
        page = recompress_jpeg(page, record.jpeg_quality)
    return page, record


def rotation_bounds(severity: Severity) -> float:
    """Largest skew the profile can produce, used by tests and the deskew search range."""
    return PROFILES[severity].rotation


def apply_motion_blur(array: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve each channel with a line kernel: the smear left by a moving hand-held camera."""
    from scipy.ndimage import convolve

    return np.stack([convolve(array[:, :, c], kernel, mode="nearest") for c in range(array.shape[2])], axis=2)


def motion_kernel(length: int, angle_deg: float) -> np.ndarray:
    """A normalised line kernel for camera-shake blur at a given direction."""
    kernel = np.zeros((length, length), dtype=np.float32)
    centre = length // 2
    for step in range(length):
        offset = step - centre
        x = round(centre + offset * math.cos(math.radians(angle_deg)))
        y = round(centre + offset * math.sin(math.radians(angle_deg)))
        kernel[min(max(y, 0), length - 1), min(max(x, 0), length - 1)] = 1.0
    return kernel / kernel.sum()
