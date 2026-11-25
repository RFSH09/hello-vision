"""Frame sources: video file, image folder, and live camera (USB index or RTSP URL)."""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from hello_vision.capture.types import Frame, Image

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _rgb(bgr: NDArray[Any]) -> Image:  # numpy dtype varies by OpenCV stub
    """Convert an OpenCV BGR array to an RGB ``uint8`` image."""
    return np.asarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)


class FrameSource(Protocol):
    """Anything that yields frames in time order."""

    camera_id: str

    def __iter__(self) -> Iterator[Frame]:
        """Iterate over frames until the source is exhausted or closed."""
        ...


class VideoFileSource:
    """Replay a video file.

    Args:
        path: Video path.
        camera_id: Id stamped on every frame.
    """

    def __init__(self, path: Path, camera_id: str = "cam0") -> None:
        self.path = path
        self.camera_id = camera_id

    def __iter__(self) -> Iterator[Frame]:
        """Yield RGB frames with ``timestamp = index / fps``."""
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open video: {self.path}")
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            i = 0
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                yield Frame(self.camera_id, i, i / fps, _rgb(bgr))
                i += 1
        finally:
            cap.release()


class ImageFolderSource:
    """Replay a folder of images in name order at a fixed rate.

    Args:
        folder: Directory with image files.
        fps: Assumed capture rate used to build timestamps.
        camera_id: Id stamped on every frame.
    """

    def __init__(self, folder: Path, fps: float = 15.0, camera_id: str = "cam0") -> None:
        self.folder = folder
        self.fps = fps
        self.camera_id = camera_id

    def __iter__(self) -> Iterator[Frame]:
        """Yield RGB frames; unreadable files raise ``ValueError``."""
        files = sorted(p for p in self.folder.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES)
        for i, p in enumerate(files):
            bgr = cv2.imread(str(p))
            if bgr is None:
                raise ValueError(f"cannot read image: {p}")
            yield Frame(self.camera_id, i, i / self.fps, _rgb(bgr))


class CameraSource:
    """Live camera: a USB device index or an RTSP/HTTP URL.

    NOTE: not exercised by tests or real hardware yet (camera model and connection are
    still open in docs/HARDWARE_QUESTIONS.md). Timestamps use the wall clock.

    Args:
        uri: Device index as a string (``"0"``) or a stream URL.
        camera_id: Id stamped on every frame.
        max_frames: Optional cap, mainly for tests and benchmarks.
    """

    def __init__(self, uri: str, camera_id: str = "cam0", max_frames: int | None = None) -> None:
        self.uri = uri
        self.camera_id = camera_id
        self.max_frames = max_frames

    def __iter__(self) -> Iterator[Frame]:
        """Yield frames until the stream ends, fails, or ``max_frames`` is reached."""
        cap = cv2.VideoCapture(int(self.uri) if self.uri.isdigit() else self.uri)
        if not cap.isOpened():
            raise ConnectionError(f"cannot open camera: {self.uri}")
        try:
            i = 0
            while self.max_frames is None or i < self.max_frames:
                ok, bgr = cap.read()
                if not ok:
                    break
                yield Frame(self.camera_id, i, time.time(), _rgb(bgr))
                i += 1
        finally:
            cap.release()
