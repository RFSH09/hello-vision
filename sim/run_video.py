"""Zero-shot demo on a real video: annotated clip, per-track JSON and a contact sheet.

    uv run hv-demo --video data/videos/X.mp4 --out out/X.mp4
    uv run hv-batch --videos data/videos --orders data/orders.csv --out out --limit 10

The "taken" guess is deliberately naive (a hand touched the item, it moved far and then
disappeared). It is a first look, not billing logic.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from hello_vision.capture.types import Image
from hello_vision.detect.types import Detection, Detector
from hello_vision.identify.index import GalleryIndex, Match, crop
from hello_vision.privacy.blur import Box, FaceBlurrer, FaceDetector, YuNetFaceDetector
from hello_vision.track.simple import ByteSimpleTracker, SimpleTrack
from sim.orders import OrderScore, count_items, fmt, merge_cameras, order_id, report

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".ts", ".dav"}
KNOWN, UNKNOWN, HAND = (40, 200, 40), (255, 165, 0), (230, 40, 200)  # RGB
FACE_MODEL = "face_detection_yunet_2023mar.onnx"
SHELF_LINE = 0.45  # fraction of the frame height; products above it are treated as stock
DARK = 14.0  # mean brightness below this = door closed, nothing to detect


class RotatedFaces:
    """Run a face detector on the frame turned 0/90/180/270 degrees (top-down views)."""

    def __init__(self, inner: FaceDetector) -> None:
        self.inner = inner

    def detect(self, image: Image) -> list[Box]:
        """Return faces found at any rotation, in the original frame coordinates."""
        h, w = image.shape[:2]
        out: list[Box] = []
        for k in range(4):
            turned = np.ascontiguousarray(np.rot90(image, k))
            for x, y, bw, bh in self.inner.detect(turned):
                if k == 0:
                    out.append((x, y, bw, bh))
                elif k == 1:
                    out.append((w - y - bh, x, bh, bw))
                elif k == 2:
                    out.append((w - x - bw, h - y - bh, bw, bh))
                else:
                    out.append((y, h - x - bw, bh, bw))
        return out


def blur_heads(image: Image, heads: Sequence[Detection]) -> Image:
    """Pixelate and smooth every head box."""
    regions = [
        (d.x1 / image.shape[1], d.y1 / image.shape[0], d.x2 / image.shape[1], d.y2 / image.shape[0])
        for d in heads
    ]
    return FaceBlurrer(None, regions, margin=0.1).blur(image) if regions else image


@dataclass
class Models:
    """Everything loaded once and reused across videos."""

    detector: Detector
    embedder: Any  # has embed_batch(list of RGB crops)
    index: GalleryIndex
    name: str
    blurrer: FaceBlurrer | None = None


@dataclass
class VideoResult:
    """Output of one processed video."""

    video: str
    annotated: str
    sheet: str
    tracks: list[dict[str, Any]]
    cart: dict[str, int]
    processed_frames: int
    seconds: float
    video_seconds: float
    notes: list[str] = field(default_factory=list)


def load_models(
    detector_name: str = "gdino",
    gallery_dir: Path = Path("data/gallery"),
    models_dir: Path = Path("models"),
    det_threshold: float | None = None,
    id_threshold: float = 0.35,
    margin: float = 0.02,
    device: str | None = None,
    views: int = 50,
) -> Models:
    """Load the zero-shot detector, DINOv2 and the (cached) SKU gallery."""
    from hello_vision.detect.zeroshot import ZeroShotDetector
    from hello_vision.identify.gallery import DinoEmbedder, build_gallery, load_packshots

    detector = ZeroShotDetector(detector_name, det_threshold, device, models_dir)
    embedder = DinoEmbedder(device, models_dir)
    photos = sorted(p for p in gallery_dir.glob("*.*"))
    if not photos:
        raise FileNotFoundError(f"no product photos in {gallery_dir}")
    key = hashlib.sha256(
        "|".join([f"v1:{views}", *(f"{p.name}:{p.stat().st_size}" for p in photos)]).encode()
    ).hexdigest()[:16]
    cache = models_dir / f"gallery_{key}.npz"
    if cache.exists():
        data = np.load(cache, allow_pickle=False)
        index = GalleryIndex(id_threshold, margin)
        for sku, vec in zip(data["skus"], data["vectors"], strict=True):
            index.add(str(sku), vec)
    else:
        index = build_gallery(load_packshots(gallery_dir), embedder, views, id_threshold, margin)
        models_dir.mkdir(parents=True, exist_ok=True)
        skus, vectors = index.export()
        np.savez_compressed(cache, skus=np.array(skus), vectors=np.stack(vectors))
    face_model = models_dir / FACE_MODEL
    if not face_model.is_file():
        raise FileNotFoundError(
            f"{face_model} missing: run `uv run python tools/fetch_face_model.py --out models`"
        )
    blurrer = FaceBlurrer(RotatedFaces(YuNetFaceDetector(face_model, score_threshold=0.75)))
    return Models(detector, embedder, index, detector_name, blurrer)


def _text(
    img: Image, text: str, x: int, y: int, colour: tuple[int, int, int], scale: float
) -> None:
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    y = max(th + 4, y)
    x = max(0, min(x, img.shape[1] - tw - 4))
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + base - 1), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


def annotate(
    image: Image, dets: Sequence[Detection], tracks: dict[int, SimpleTrack], seconds: float
) -> Image:
    """Draw boxes, SKU names, scores, track ids and a legend on a copy of ``image``."""
    out = image.copy()
    scale = max(0.4, out.shape[1] / 2200)
    for i, d in enumerate(dets):
        track = tracks.get(i)
        if d.label == "head":
            continue
        if d.label == "hand":
            colour, label = HAND, f"hand #{track.id}" if track else "hand"
        else:
            sku, conf, guess = track.sku_vote() if track else (None, 0.0, None)
            colour = KNOWN if sku else UNKNOWN
            name = f"{sku} {conf:.2f}" if sku else f"unknown ({guess}?)" if guess else "unknown"
            label = f"#{track.id} {name}" if track else f"product {d.score:.2f}"
        cv2.rectangle(out, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), colour, 2)
        _text(out, label, int(d.x1), int(d.y1) - 2, colour, scale)
    legend = [("identified product", KNOWN), ("unknown product", UNKNOWN), ("hand", HAND)]
    for k, (name, colour) in enumerate(legend):
        _text(out, name, 8, 22 + k * int(40 * scale), colour, scale)
    _text(out, f"t={seconds:5.1f}s", out.shape[1] - 110, 22, (255, 255, 255), scale)
    return out


def _open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    """MPEG-4 writer (this OpenCV build has no H.264 encoder)."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"cannot open a video writer for {path}")
    return writer


def summarise_tracks(
    products: Sequence[SimpleTrack],
    hands: Sequence[SimpleTrack],
    times: Sequence[float],
    size: tuple[int, int],
    min_hits: int = 2,
    shelf_line: float | None = None,
) -> list[dict[str, Any]]:
    """Per-track report including the naive "taken" guess."""
    width, height = size
    diag = (width**2 + height**2) ** 0.5
    hand_boxes: dict[int, list[Detection]] = {}
    for h in hands:
        for f, d in zip(h.frames, h.dets, strict=True):
            hand_boxes.setdefault(f, []).append(d)
    out: list[dict[str, Any]] = []
    for t in products:
        if len(t.dets) < min_hits:
            continue
        held = [
            d
            for f, d in zip(t.frames, t.dets, strict=True)
            if any(
                not (d.x2 + 20 < h.x1 or h.x2 + 20 < d.x1 or d.y2 + 20 < h.y1 or h.y2 + 20 < d.y1)
                for h in hand_boxes.get(f, [])
            )
        ]
        touched = bool(held)
        (x0, y0) = t.dets[0].center
        travel = max(((d.center[0] - x0) ** 2 + (d.center[1] - y0) ** 2) ** 0.5 for d in t.dets)
        end = t.dets[-1]
        near_edge = (
            end.x1 < 0.04 * width
            or end.x2 > 0.96 * width
            or end.y1 < 0.04 * height
            or end.y2 > 0.96 * height
        )
        gone = t.frames[-1] < len(times) - 3
        moved = travel > 0.12 * diag
        sku, conf, guess = t.sku_vote()
        if shelf_line is None:
            taken = bool(touched and moved and (gone or near_edge))
        else:  # fridge cameras: shelf on top, a taken item shows up in a hand below it
            taken = sum(d.center[1] > shelf_line * height for d in held) >= 2
        out.append(
            {
                "track_id": t.id,
                "sku": sku,
                "confidence": round(conf, 3),
                "best_guess": guess,
                "first_seen_s": round(times[t.frames[0]], 2),
                "last_seen_s": round(times[t.frames[-1]], 2),
                "frames": len(t.dets),
                "hand_touched": touched,
                "travel_fraction": round(travel / diag, 3),
                "taken_guess": taken,
                "best_frame": int(t.frames[int(np.argmax([d.score for d in t.dets]))]),
                # [time s, x1, y1, x2, y2, score, held by a hand] to extract real crops later
                "boxes": [
                    [
                        round(times[f], 2),
                        round(d.x1),
                        round(d.y1),
                        round(d.x2),
                        round(d.y2),
                        round(d.score, 3),
                        int(d in held),
                    ]
                    for f, d in zip(t.frames, t.dets, strict=True)
                ],
            }
        )
    return out


def contact_sheet(frames: Sequence[Image], busy: Sequence[int], path: Path, n: int = 8) -> None:
    """Save ``n`` key frames (preferring frames with hands) as a 4-column grid."""
    pool = list(busy) if len(busy) >= n else list(range(len(frames)))
    picks = [pool[int(i)] for i in np.linspace(0, len(pool) - 1, min(n, len(pool))).round()]
    tiles = [frames[i] for i in picks]
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i : i + 4]) for i in range(0, len(tiles), 4)]
    cv2.imwrite(str(path), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))


def process_video(
    video: Path, out: Path, models: Models, fps: float = 5.0, max_seconds: float | None = None
) -> VideoResult:
    """Run the pipeline on one video; write the annotated clip, JSON and contact sheet."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if not 1 <= src_fps <= 240:
        src_fps = 25.0
    out.parent.mkdir(parents=True, exist_ok=True)
    threshold = float(getattr(models.detector, "threshold", 0.2))
    products = ByteSimpleTracker("product", high_score=threshold + 0.05)
    hands = ByteSimpleTracker("hand", high_score=threshold, max_distance=1.5)
    writer: cv2.VideoWriter | None = None
    times: list[float] = []
    thumbs: list[Image] = []
    busy: list[int] = []
    size = (0, 0)
    index = -1
    next_t = 0.0
    recent_heads: list[list[Detection]] = []
    start = time.perf_counter()
    try:
        while True:
            if not cap.grab():
                break
            index += 1
            t = index / src_fps
            if max_seconds is not None and t > max_seconds:
                break
            if t + 1e-9 < next_t:
                continue
            next_t += 1.0 / fps
            ok, bgr = cap.retrieve()
            if not ok:
                continue
            image = np.asarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)
            size = (image.shape[1], image.shape[0])
            if not times:
                models.detector.start_session(image)
            dark = float(image.mean()) < DARK
            dets = [] if dark else models.detector.detect(image)
            prod_idx = [i for i, d in enumerate(dets) if d.label == "product"]
            vectors = models.embedder.embed_batch(
                [crop(image, (dets[i].x1, dets[i].y1, dets[i].x2, dets[i].y2)) for i in prod_idx]
            )
            matches: list[Match | None] = [None] * len(dets)
            for i, v in zip(prod_idx, vectors, strict=True):
                matches[i] = models.index.query(v)
            frame_no = len(times)
            times.append(t)
            assigned = products.update(frame_no, dets, matches)
            hand_assigned = hands.update(frame_no, dets)
            # faces are blurred in everything that is written to disk (not before detection)
            shown = models.blurrer.blur(image) if models.blurrer and not dark else image
            recent_heads = [*recent_heads[-2:], [d for d in dets if d.label == "head"]]
            shown = blur_heads(shown, [h for frame in recent_heads for h in frame])
            drawn = annotate(shown, dets, {**assigned, **hand_assigned}, t)
            if writer is None:
                scale = min(1.0, 1280 / size[0])
                out_size = (int(size[0] * scale) // 2 * 2, int(size[1] * scale) // 2 * 2)
                writer = _open_writer(out, fps, out_size)
            writer.write(cv2.cvtColor(cv2.resize(drawn, out_size), cv2.COLOR_RGB2BGR))
            thumbs.append(
                np.asarray(cv2.resize(drawn, (480, int(480 * size[1] / size[0]))), dtype=np.uint8)
            )
            if hand_assigned:
                busy.append(frame_no)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if not times:
        raise ValueError(f"no frames could be read from {video}")
    seconds = time.perf_counter() - start
    line = SHELF_LINE if order_id(video.stem) != video.stem else None  # *_camN fridge videos
    tracks = summarise_tracks(products.tracks, hands.tracks, times, size, shelf_line=line)
    cart = count_items(tracks)
    sheet = out.with_name(out.stem + "_sheet.jpg")
    contact_sheet(thumbs, busy, sheet)
    result = VideoResult(
        str(video), str(out), str(sheet), tracks, dict(cart), len(times), round(seconds, 1),
        round(times[-1], 1),
    )  # fmt: skip
    report = {**result.__dict__, "detector": models.name, "sample_fps": fps}
    out.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return result


def speed_line(r: VideoResult) -> str:
    """One-line speed summary."""
    fps = r.processed_frames / max(r.seconds, 1e-9)
    return (
        f"{r.processed_frames} frames in {r.seconds:.1f}s = {fps:.1f} processed fps "
        f"({r.seconds / max(r.video_seconds, 1e-9):.2f}x video length)"
    )


def read_orders(path: Path) -> dict[str, Counter[str]]:
    """Read ``video_id,sku,qty`` rows."""
    orders: dict[str, Counter[str]] = {}
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            video = Path(row["video_id"].strip()).stem
            orders.setdefault(video, Counter())[row["sku"].strip()] += int(row.get("qty") or 1)
    return orders


def _parser(batch: bool) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    if batch:
        p.add_argument("--videos", type=Path, default=Path("data/videos"))
        p.add_argument("--orders", type=Path, default=Path("data/orders.csv"))
        p.add_argument("--out", type=Path, default=Path("out"))
        p.add_argument("--limit", type=int, default=1000)
    else:
        p.add_argument("--video", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
    p.add_argument("--backend", choices=["owlv2", "gdino"], default="gdino")
    p.add_argument("--fps", type=float, default=5.0)
    p.add_argument("--det-threshold", type=float)
    p.add_argument("--id-threshold", type=float, default=0.35)
    p.add_argument("--margin", type=float, default=0.02)
    p.add_argument("--gallery", type=Path, default=Path("data/gallery"))
    p.add_argument("--device", choices=["cuda", "cpu"])
    p.add_argument("--max-seconds", type=float)
    return p


def _models(a: argparse.Namespace) -> Models:
    return load_models(
        a.backend, a.gallery, Path("models"), a.det_threshold, a.id_threshold, a.margin, a.device
    )


def main(argv: Sequence[str] | None = None) -> int:
    """``hv-demo``: one video."""
    a = _parser(batch=False).parse_args(argv)
    models = _models(a)
    r = process_video(a.video, a.out, models, a.fps, a.max_seconds)
    print(
        f"annotated: {r.annotated}\nsheet:     {r.sheet}\njson:      {a.out.with_suffix('.json')}"
    )
    print(f"speed:     {speed_line(r)}")
    print(f"naive cart: {fmt(Counter(r.cart))}")
    return 0


def _stack_sheets(sheets: Sequence[str], dest: Path) -> None:
    """Stack the camera contact sheets of one order vertically."""
    images = [img for img in (cv2.imread(x) for x in sheets) if img is not None]
    if not images:
        return
    width = images[0].shape[1]
    rows = [cv2.resize(i, (width, int(i.shape[0] * width / i.shape[1]))) for i in images]
    cv2.imwrite(str(dest), np.vstack(rows))


def batch(argv: Sequence[str] | None = None) -> int:
    """``hv-batch``: all videos grouped into orders, with a comparison against orders.csv."""
    a = _parser(batch=True).parse_args(argv)
    videos = sorted(p for p in a.videos.rglob("*") if p.suffix.lower() in VIDEO_SUFFIXES)
    groups: dict[str, list[Path]] = {}
    for v in videos:
        groups.setdefault(order_id(v.stem), []).append(v)
    if not groups:
        print(f"no videos found in {a.videos}")
        return 1
    orders = read_orders(a.orders) if a.orders.exists() else {}
    models = _models(a)
    scores: list[OrderScore] = []
    other: list[str] = []
    errors: list[str] = []
    frames = seconds = video_seconds = 0.0
    for n, (order, files) in enumerate(sorted(groups.items())[: a.limit], start=1):
        carts, seen, sheets = [], set[str](), []
        try:
            for v in files:
                r = process_video(v, a.out / f"{v.stem}.mp4", models, a.fps, a.max_seconds)
                print(
                    f"[{n}/{len(groups)}] {v.stem}: {speed_line(r)} cart: {fmt(Counter(r.cart))}",
                    flush=True,
                )
                carts.append(Counter(r.cart))
                seen.update(t["sku"] for t in r.tracks if t["sku"])
                sheets.append(r.sheet)
                frames += r.processed_frames
                seconds += r.seconds
                video_seconds += r.video_seconds
        except (OSError, ValueError, RuntimeError) as exc:
            errors.append(f"{order}: {exc}")
            continue
        if len(files) > 1:
            _stack_sheets(sheets, a.out / f"{order}_sheet.jpg")
        cart = merge_cameras(carts)
        unknown = cart.pop("unknown", 0)
        if order in orders:
            scores.append(OrderScore(order, cart, orders[order], seen, unknown))
            (a.out / "summary.md").write_text(report(scores), encoding="utf-8")
        else:
            other.append(f"{order}: {fmt(cart)} (no order data)")
    text = report(scores) if scores else ""
    if other:
        text += "\nWithout order data:\n" + "\n".join(f"- {o}" for o in other) + "\n"
    if errors:
        text += "\nErrors:\n" + "\n".join(f"- {e}" for e in errors) + "\n"
    text += (
        f"\nSpeed: {frames / max(seconds, 1e-9):.1f} processed fps; "
        f"{seconds:.0f}s for {video_seconds:.0f}s of video at {a.fps} fps sampling.\n"
    )
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
