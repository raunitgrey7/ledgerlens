"""Seeded synthetic document generator: content, layout templates, rendering, degradation.

The generator exists because public invoice datasets are small, licence-restricted and
rarely carry line-item ground truth. Rendering our own pages gives exact field, line
item and word-box labels for every pixel we score against, at any volume, with a seed.
"""
