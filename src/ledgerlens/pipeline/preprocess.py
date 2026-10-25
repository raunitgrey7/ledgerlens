"""Turn a captured page into a clean, upright binary image.

Order matters and each step undoes one capture defect: coloured ink (stamps) is
suppressed before greyscale conversion because stamps carry chroma that text lacks;
the skew angle is found by maximising the variance of the horizontal projection
profile, which peaks when text rows are horizontal; a median filter removes dust
before thresholding; Sauvola's local threshold copes with uneven lighting where a
single global threshold would swallow the dark corner of a vignetted page.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage

SATURATION_LIMIT = 48
SKEW_RANGE_DEG = 5.0
SKEW_COARSE_STEP = 0.5
SKEW_FINE_STEP = 0.1
SKEW_DOWNSAMPLE = 4
NOISE_FLOOR = 2.5
MAX_SMOOTH_SIGMA = 1.0


@dataclass
class Preprocessed:
    """Deskewed greyscale page and its ink mask."""

    gray: np.ndarray
    ink: np.ndarray
    skew_deg: float
    noise_sigma: float = 0.0


def suppress_colour_ink(rgb: np.ndarray) -> np.ndarray:
    """Greyscale where saturated (coloured) pixels take their brightest channel.

    Under a red stamp, paper keeps a high red channel while black text stays dark in
    every channel, so the brightest channel removes the stamp and keeps the text.
    """
    channels = rgb.astype(np.int16)
    saturation = channels.max(axis=2) - channels.min(axis=2)
    gray = (0.299 * channels[:, :, 0] + 0.587 * channels[:, :, 1] + 0.114 * channels[:, :, 2]).astype(np.float32)
    coloured = saturation > SATURATION_LIMIT
    if coloured.any():
        gray = np.where(coloured, channels.max(axis=2).astype(np.float32), gray)
    return np.clip(gray, 0, 255).astype(np.uint8)


def _profile_score(small: np.ndarray, angle: float) -> float:
    rotated = ndimage.rotate(small, angle, reshape=False, order=1, mode="constant", cval=0.0)
    return float(rotated.sum(axis=1).var())


def estimate_skew(ink: np.ndarray) -> float:
    """Page skew in degrees, counter-clockwise positive as Pillow rotates; 0 when upright."""
    small = ink[::SKEW_DOWNSAMPLE, ::SKEW_DOWNSAMPLE].astype(np.float32)
    if small.sum() < 10:
        return 0.0
    coarse = np.arange(-SKEW_RANGE_DEG, SKEW_RANGE_DEG + 1e-9, SKEW_COARSE_STEP)
    best = max(coarse, key=lambda angle: _profile_score(small, float(angle)))
    fine = np.arange(best - SKEW_COARSE_STEP, best + SKEW_COARSE_STEP + 1e-9, SKEW_FINE_STEP)
    best = max(fine, key=lambda angle: _profile_score(small, float(angle)))
    return round(-float(best), 2)


def sauvola(gray: np.ndarray, window: int = 31, k: float = 0.2, dynamic_range: float = 128.0) -> np.ndarray:
    """Local threshold T = m (1 + k (s / R - 1)); returns the ink mask."""
    values = gray.astype(np.float32)
    mean = ndimage.uniform_filter(values, window, mode="reflect")
    mean_sq = ndimage.uniform_filter(values * values, window, mode="reflect")
    std = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))
    threshold = mean * (1.0 + k * (std / dynamic_range - 1.0))
    return (values < threshold) & (values < 200)


def quick_ink(gray: np.ndarray) -> np.ndarray:
    """Cheap global mask used only to estimate skew before the real binarisation."""
    threshold = max(60.0, float(np.percentile(gray, 50)) * 0.7)
    return gray < threshold


def clean_impulses(ink: np.ndarray) -> np.ndarray:
    """Drop ink pixels with no ink neighbour and fill paper pixels surrounded by ink.

    This removes dust and sensor spikes without the stroke thinning a grey median filter
    causes on hairline serifs; the end pixel of a 1 px hyphen keeps its one neighbour.
    """
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    neighbours = ndimage.convolve(ink.astype(np.uint8), kernel, mode="constant", cval=0)
    kept = ink & (neighbours >= 1)
    filled = ~ink & (neighbours >= 7)
    return kept | filled


def estimate_noise(gray: np.ndarray) -> float:
    """Robust sensor-noise estimate: scaled median absolute deviation from a 3x3 median."""
    sample = gray[::3, ::3].astype(np.float32)
    residual = np.abs(sample - ndimage.median_filter(sample, size=3))
    return float(1.4826 * np.median(residual))


def denoise(gray: np.ndarray, noise_sigma: float) -> np.ndarray:
    """Gaussian smoothing proportional to the measured noise; clean pages are left alone.

    Smoothing also widens 1 px hairlines into connected 2 to 3 px strokes, which keeps
    thin serif glyphs in one piece after thresholding.
    """
    if noise_sigma < NOISE_FLOOR:
        return gray
    sigma = min(MAX_SMOOTH_SIGMA, 0.35 + 0.05 * noise_sigma)
    return ndimage.gaussian_filter(gray.astype(np.float32), sigma).astype(np.uint8)


def preprocess(image: Image.Image) -> Preprocessed:
    """Colour suppression, deskew, noise-adaptive smoothing, Sauvola binarisation, cleanup."""
    rgb = np.asarray(image.convert("RGB"))
    gray = suppress_colour_ink(rgb)
    skew = estimate_skew(quick_ink(gray))
    if abs(skew) >= SKEW_FINE_STEP:
        gray = ndimage.rotate(gray, -skew, reshape=False, order=1, mode="constant", cval=255).astype(np.uint8)
    noise = estimate_noise(gray)
    smoothed = denoise(gray, noise)
    ink = clean_impulses(sauvola(smoothed))
    return Preprocessed(gray=smoothed, ink=ink, skew_deg=skew, noise_sigma=round(noise, 2))
