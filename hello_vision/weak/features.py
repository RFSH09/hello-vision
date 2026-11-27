"""Preprocess a session once: keep frames with motion in the door zone, extract frozen features.

For each kept frame two views are embedded with a frozen DINOv2 backbone: the whole frame and
a crop around the moving region in the door zone (a cheap stand-in for a hand crop until a
small hand detector is chosen). Per view: the class token plus a 4x4 grid of pooled patch
tokens, stored as int8 with a per-array scale. One ``.npz`` per order.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from hello_vision.capture.types import Image

DOOR_LINE = 0.45  # the door zone is below this fraction of the height
GRID = 4  # patch tokens are pooled to GRID x GRID


@dataclass
class MotionFrame:
    """A sampled frame with motion, and the box (pixels) around the motion in the door zone."""

    time_s: float
    image: Image
    box: tuple[int, int, int, int]
    motion: float


class FrameEncoder(Protocol):
    """Frozen backbone: images -> ``(N, 1 + GRID*GRID, dim)`` tokens."""

    def tokens(self, images: Sequence[Image]) -> NDArray[np.float32]:
        """Encode a batch."""
        ...


def motion_box(
    mask: NDArray[np.bool_], scale_x: float, scale_y: float, offset_y: int, size: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Square-ish crop around the largest moving blob, in original pixels."""
    width, height = size
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return 0, offset_y, width, height
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = (int(v) for v in stats[best, :4])
    cx, cy = (x + w / 2) * scale_x, (y + h / 2) * scale_y + offset_y
    side = max(w * scale_x, h * scale_y, 0.3 * height) * 1.2
    left = int(max(0, min(cx - side / 2, width - side)))
    top = int(max(0, min(cy - side / 2, height - side)))
    return left, top, int(min(width, left + side)), int(min(height, top + side))


def motion_frames(
    video: Path,
    fps: float = 5.0,
    threshold: float = 0.012,
    pixel_delta: int = 18,
    max_frames: int = 96,
) -> list[MotionFrame]:
    """Sample at ``fps`` and keep frames whose door zone changed since the previous sample.

    Args:
        video: Input video.
        fps: Sampling rate.
        threshold: Minimum share of changed pixels in the door zone.
        pixel_delta: Grey-level difference that counts as a change.
        max_frames: Keep at most this many (the ones with the most motion), in time order.
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video}")
    src = cap.get(cv2.CAP_PROP_FPS) or 25.0
    previous: NDArray[np.uint8] | None = None
    kept: list[MotionFrame] = []
    index, next_t = -1, 0.0
    try:
        while cap.grab():
            index += 1
            t = index / src
            if t + 1e-9 < next_t:
                continue
            next_t += 1.0 / fps
            ok, bgr = cap.retrieve()
            if not ok:
                continue
            h, w = bgr.shape[:2]
            top = int(DOOR_LINE * h)
            small = np.asarray(
                cv2.GaussianBlur(
                    cv2.resize(cv2.cvtColor(bgr[top:], cv2.COLOR_BGR2GRAY), (160, 56)), (5, 5), 0
                ),
                dtype=np.uint8,
            )
            if previous is not None:
                mask = cv2.absdiff(small, previous) > pixel_delta
                amount = float(mask.mean())
                if amount >= threshold:
                    box = motion_box(mask, w / 160, (h - top) / 56, top, (w, h))
                    rgb = np.asarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)
                    kept.append(MotionFrame(round(t, 2), rgb, box, amount))
                    if len(kept) > 2 * max_frames:  # bound memory on long videos
                        kept = _strongest(kept, max_frames)
            previous = small
    finally:
        cap.release()
    return _strongest(kept, max_frames)


def _strongest(frames: list[MotionFrame], n: int) -> list[MotionFrame]:
    if len(frames) <= n:
        return frames
    best = sorted(frames, key=lambda f: f.motion, reverse=True)[:n]
    return sorted(best, key=lambda f: f.time_s)


class DinoTokenEncoder:
    """DINOv2 (Apache-2.0) class token + pooled patch grid, no gradients."""

    def __init__(
        self,
        model_id: str = "facebook/dinov2-small",
        cache_dir: Path = Path("models"),
        device: str | None = None,
        size: int = 224,
        batch_size: int = 64,
    ) -> None:
        import torch
        from transformers import AutoModel

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # use the saved copy first: even a cached model makes an online check, which can hang
        # for good on a bad network
        model: Any
        try:
            model = AutoModel.from_pretrained(
                model_id, cache_dir=str(cache_dir), local_files_only=True
            )
        except OSError:  # not downloaded yet
            model = AutoModel.from_pretrained(model_id, cache_dir=str(cache_dir))
        self.model = model.to(self.device).eval()
        self.size = size
        self.batch_size = batch_size

    def tokens(self, images: Sequence[Image]) -> NDArray[np.float32]:
        """Encode images to ``(N, 17, dim)``."""
        import torch

        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        out: list[NDArray[np.float32]] = []
        for i in range(0, len(images), self.batch_size):
            batch = np.stack(
                [cv2.resize(im, (self.size, self.size)) for im in images[i : i + self.batch_size]]
            )
            x = torch.from_numpy(batch).to(self.device).permute(0, 3, 1, 2).float() / 255.0
            with (
                torch.inference_mode(),
                torch.autocast("cuda", dtype=torch.float16, enabled=self.device == "cuda"),
            ):
                t = self.model(pixel_values=(x - mean) / std).last_hidden_state.float()
            n, patches = t.shape[0], t[:, 1:]
            side = int(patches.shape[1] ** 0.5)
            grid = patches.reshape(n, side, side, -1).permute(0, 3, 1, 2)
            pooled = torch.nn.functional.adaptive_avg_pool2d(grid, GRID).flatten(2).transpose(1, 2)
            out.append(torch.cat([t[:, :1], pooled], dim=1).cpu().numpy())
        return np.concatenate(out)


def quantise(tokens: NDArray[np.float32]) -> tuple[NDArray[np.int8], np.float32]:
    """Store tokens as int8 with one scale per array (about half the size of float16)."""
    peak = float(np.abs(tokens).max()) if tokens.size else 0.0
    scale = np.float32(peak / 127.0 if peak > 0 else 1.0)
    return np.asarray(np.round(tokens / scale), dtype=np.int8), scale


def dequantise(stored: NDArray[np.int8], scale: np.float32) -> NDArray[np.float32]:
    """Inverse of :func:`quantise` (up to rounding)."""
    return np.asarray(stored.astype(np.float32) * scale, dtype=np.float32)


def extract_session(videos: Sequence[Path], encoder: FrameEncoder, dest: Path) -> int:
    """Write the features of one order (all its cameras) to ``dest``; return the frame count."""
    full, crops, cams, times, boxes = [], [], [], [], []
    for cam, video in enumerate(videos):
        frames = motion_frames(video)
        if not frames:
            continue
        full.append(encoder.tokens([f.image for f in frames]))
        crops.append(
            encoder.tokens([f.image[f.box[1] : f.box[3], f.box[0] : f.box[2]] for f in frames])
        )
        cams += [cam] * len(frames)
        times += [f.time_s for f in frames]
        boxes += [f.box for f in frames]
    dest.parent.mkdir(parents=True, exist_ok=True)
    empty = np.zeros((0, 1 + GRID * GRID, 1), np.float32)
    full_q, full_scale = quantise(np.concatenate(full) if full else empty)
    crop_q, crop_scale = quantise(np.concatenate(crops) if crops else empty)
    partial = dest.with_suffix(".partial.npz")
    np.savez(
        partial,
        full=full_q,
        crop=crop_q,
        full_scale=full_scale,
        crop_scale=crop_scale,
        cam=np.array(cams, np.int8),
        time=np.array(times, np.float32),
        box=np.array(boxes, np.int16).reshape(-1, 4),
    )
    partial.replace(dest)
    return len(times)
