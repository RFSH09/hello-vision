"""Download the YuNet face-detection model (MIT, OpenCV Zoo) and verify its checksum.

Usage: ``uv run python tools/fetch_face_model.py --out /var/lib/hello-vision/models``
The file is git-ignored (``*.onnx``); it is fetched at deployment time, never committed.
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from collections.abc import Sequence
from pathlib import Path

URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)
SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"


def main(argv: Sequence[str] | None = None) -> int:
    """Fetch the model into ``--out`` and fail if the checksum differs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / URL.rsplit("/", 1)[1]
    with urllib.request.urlopen(URL, timeout=60) as r:
        data = r.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"checksum mismatch: {digest}")
    dest.write_bytes(data)
    print(f"saved {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
