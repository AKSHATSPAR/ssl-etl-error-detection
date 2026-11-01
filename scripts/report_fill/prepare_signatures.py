"""Turn photographed signatures into clean cutouts for the declaration page.

A phone photo of a signature carries the paper with it, so dropping the file
straight into the document puts a grey rectangle on the page. This reads the
ink out of the photo instead: paper becomes fully transparent, ink keeps its
own colour, and the result is cropped to the writing.

Source images live in artifacts/signatures/raw and are not committed, since
they are personal. Run this after putting them there:

    python scripts/report_fill/prepare_signatures.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "artifacts/signatures/raw"
OUT = ROOT / "artifacts/signatures"

# Which photo belongs to whom, and what the cutout should be called.
SOURCES = {
    "anuprakash.png": "anuprakash_raw.jpg",
    "akshat.png": "akshat_raw.jpg",
}


def cutout(path: Path, pad: int = 12) -> Image.Image:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32)
    lum = arr @ np.array([0.299, 0.587, 0.114], dtype=np.float32)

    # The paper is whatever most of the photo is; the ink is the dark tail.
    paper = float(np.percentile(lum, 85))
    ink = float(np.percentile(lum, 1))
    spread = max(paper - ink, 1.0)

    # Hold the darkest fifth fully opaque so strokes stay solid, and let the
    # rest fall away, which keeps the stroke edges smooth rather than jagged.
    alpha = np.clip((paper - lum) / spread, 0.0, 1.0)
    alpha = np.clip((alpha - 0.30) / 0.40, 0.0, 1.0)

    # Rough paper photographs as grain, which survives the threshold as specks
    # around the strokes. A median pass removes them and leaves the ink.
    alpha = np.asarray(
        Image.fromarray((alpha * 255).astype(np.uint8))
        .filter(ImageFilter.MedianFilter(5))
    ).astype(np.float32) / 255.0

    solid = alpha > 0.75
    if not solid.any():
        raise SystemExit(f"no ink found in {path.name}")
    colour = np.median(arr[solid], axis=0)
    # Deepen it a little; a photo lightens ink and the page should look inked.
    colour = np.clip(colour * 0.85, 0, 255)

    h, w = alpha.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = colour.astype(np.uint8)
    rgba[..., 3] = (alpha * 255).astype(np.uint8)

    rows = np.where(solid.any(axis=1))[0]
    cols = np.where(solid.any(axis=0))[0]
    top, bot = max(rows[0] - pad, 0), min(rows[-1] + pad + 1, h)
    left, right = max(cols[0] - pad, 0), min(cols[-1] + pad + 1, w)
    return Image.fromarray(rgba[top:bot, left:right])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, src in SOURCES.items():
        path = RAW / src
        if not path.exists():
            print(f"  missing {path.relative_to(ROOT)}, skipped")
            continue
        img = cutout(path)
        img.save(OUT / name)
        print(f"  {name}  {img.width}x{img.height}")


if __name__ == "__main__":
    main()
