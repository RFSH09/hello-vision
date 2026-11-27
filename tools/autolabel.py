"""Pre-label real frames for detector training (correct them in CVAT or Label Studio).

    uv run --extra vision python tools/autolabel.py --results out --out out/autolabel --count 500

Frames are chosen where the model is least sure and hands are involved (read from the
``hv-batch`` JSON), then detected again, face-blurred and exported as COCO (CVAT) and Label
Studio pre-annotations. Everything written here is private.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from hello_vision.datatools.labeling import (
    LabeledBox,
    LabeledFrame,
    export_coco,
)
from hello_vision.identify.index import crop
from sim.run_video import blur_heads, load_models


def frame_uncertainty(tracks: Sequence[dict[str, Any]]) -> dict[float, float]:
    """Time -> how interesting the frame is: low detector scores and held, unnamed items."""
    scores: dict[float, list[float]] = {}
    for t in tracks:
        for b in t["boxes"]:
            value = (1.0 - b[5]) + (1.0 if b[6] else 0.0) + (0.5 if b[6] and not t["sku"] else 0)
            scores.setdefault(b[0], []).append(value)
    return {time: float(np.mean(v)) + 0.1 * len(v) ** 0.5 for time, v in scores.items()}


def select(
    per_video: dict[str, dict[float, float]], count: int, min_gap_s: float = 0.6
) -> dict[str, list[float]]:
    """Pick ``count`` frames, round-robin over videos, most uncertain first, spread in time."""
    queues = {v: sorted(s, key=s.__getitem__, reverse=True) for v, s in per_video.items()}
    chosen: dict[str, list[float]] = {v: [] for v in queues}
    total = 0
    while total < count and any(queues.values()):
        for video, queue in sorted(queues.items()):
            while queue and total < count:
                t = queue.pop(0)
                if all(abs(t - o) >= min_gap_s for o in chosen[video]):
                    chosen[video].append(t)
                    total += 1
                    break
    return {v: sorted(t) for v, t in chosen.items() if t}


def label_studio_tasks(frames: Sequence[LabeledFrame], prefix: str) -> list[dict[str, Any]]:
    """Label Studio tasks with rectangle predictions (``product:<sku>`` or ``hand`` labels)."""
    tasks: list[dict[str, Any]] = []
    for f in frames:
        results = []
        for i, b in enumerate(f.boxes):
            name = f"product:{b.sku}" if b.sku else b.label
            results.append(
                {
                    "id": f"b{i}",
                    "type": "rectanglelabels",
                    "from_name": "label",
                    "to_name": "image",
                    "original_width": f.width,
                    "original_height": f.height,
                    "value": {
                        "x": 100 * b.x1 / f.width,
                        "y": 100 * b.y1 / f.height,
                        "width": 100 * (b.x2 - b.x1) / f.width,
                        "height": 100 * (b.y2 - b.y1) / f.height,
                        "rectanglelabels": [name],
                    },
                }
            )
        tasks.append(
            {"data": {"image": prefix + f.file_name}, "predictions": [{"result": results}]}
        )
    return tasks


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("out"))
    parser.add_argument("--videos", type=Path, default=Path("data/videos"))
    parser.add_argument("--out", type=Path, default=Path("out/autolabel"))
    parser.add_argument("--count", type=int, default=500)
    args = parser.parse_args(argv)

    per_video = {
        r.stem: frame_uncertainty(json.loads(r.read_text(encoding="utf-8"))["tracks"])
        for r in sorted(args.results.glob("order-*.json"))
    }
    chosen = select(per_video, args.count)
    models = load_models()
    images = args.out / "images"
    images.mkdir(parents=True, exist_ok=True)
    frames: list[LabeledFrame] = []
    for n, (name, times) in enumerate(sorted(chosen.items()), start=1):
        cap = cv2.VideoCapture(str(args.videos / f"{name}.mp4"))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        wanted = {round(t, 2) for t in times}
        index = -1
        while wanted and cap.grab():
            index += 1
            key = round(index / fps, 2)
            if key not in wanted:
                continue
            wanted.discard(key)
            ok, bgr = cap.retrieve()
            if not ok:
                continue
            image = np.asarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)
            dets = models.detector.detect(image)
            boxes: list[LabeledBox] = []
            for d in dets:
                if d.label == "head" or d.x2 <= d.x1 or d.y2 <= d.y1:
                    continue
                sku = None
                if d.label == "product":
                    patch = crop(image, (d.x1, d.y1, d.x2, d.y2))
                    sku = models.index.query(models.embedder.embed_batch([patch])[0]).sku
                boxes.append(
                    LabeledBox(
                        label="hand" if d.label == "hand" else "product",
                        sku=sku,
                        x1=d.x1,
                        y1=d.y1,
                        x2=d.x2,
                        y2=d.y2,
                    )
                )
            items = [(d.x1, d.y1, d.x2, d.y2) for d in dets if d.label == "product"]
            shown = models.blurrer.blur(image, items) if models.blurrer else image
            shown = blur_heads(shown, [d for d in dets if d.label == "head"])
            file_name = f"{name}_{index:05d}.jpg"
            bgr_out = cv2.cvtColor(shown, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(images / file_name), bgr_out, [cv2.IMWRITE_JPEG_QUALITY, 92])
            frames.append(
                LabeledFrame(
                    file_name=file_name,
                    width=image.shape[1],
                    height=image.shape[0],
                    boxes=boxes,
                )
            )
        cap.release()
        print(f"[{n}/{len(chosen)}] {name}: {len(frames)} frames so far", flush=True)
    export_coco(frames, args.out / "cvat_coco.json")
    tasks = label_studio_tasks(frames, "/data/local-files/?d=images/")
    (args.out / "label_studio.json").write_text(json.dumps(tasks), encoding="utf-8")
    with zipfile.ZipFile(args.out / "cvat_images.zip", "w") as z:
        for f in frames:
            z.write(images / f.file_name, f.file_name)
    print(f"{len(frames)} frames, {sum(len(f.boxes) for f in frames)} boxes -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
