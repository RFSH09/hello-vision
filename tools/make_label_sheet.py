"""Build the local labelling sheet from the results of ``hv-batch``.

    uv run --extra vision python tools/make_label_sheet.py --results out --out out/labeling

Open ``out/labeling/index.html`` in a browser. The folder contains private footage crops.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image as PILImage

from hello_vision.capture.types import Image
from hello_vision.datatools.crops import pick, track_candidates
from hello_vision.datatools.label_sheet import render
from hello_vision.privacy.blur import FaceBlurrer, YuNetFaceDetector
from sim.orders import order_id
from sim.run_video import FACE_MODEL, RotatedFaces, load_models, read_orders


def write_refs(gallery: Path, dest: Path) -> list[str]:
    """Small reference images of every product; returns the SKU ids."""
    dest.mkdir(parents=True, exist_ok=True)
    skus: list[str] = []
    for p in sorted(gallery.glob("*.png")):
        with PILImage.open(p) as im:
            rgba = im.convert("RGBA")
            white = PILImage.new("RGBA", rgba.size, (255, 255, 255, 255))
            white.alpha_composite(rgba)
            white.convert("RGB").resize((220, 220)).save(dest / f"{p.stem}.jpg", quality=85)
        skus.append(p.stem)
    return skus


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("out"))
    parser.add_argument("--videos", type=Path, default=Path("data/videos"))
    parser.add_argument("--orders", type=Path, default=Path("data/orders.csv"))
    parser.add_argument("--gallery", type=Path, default=Path("data/gallery"))
    parser.add_argument("--out", type=Path, default=Path("out/labeling"))
    parser.add_argument("--min-boxes", type=int, default=2)
    parser.add_argument(
        "--update",
        action="store_true",
        help="keep the existing items and ids; re-render the page and re-cut sharp crops",
    )
    args = parser.parse_args(argv)
    if args.update:
        return update(args.results, args.videos, args.out, args.min_boxes)

    models = load_models(gallery_dir=args.gallery)  # also checks the face model is present
    assert models.blurrer is not None, f"face model {FACE_MODEL} required"  # noqa: S101
    blurrer = models.blurrer
    orders = read_orders(args.orders) if args.orders.exists() else {}
    shutil.rmtree(args.out / "crops", ignore_errors=True)  # regenerated below
    skus = write_refs(args.gallery, args.out / "ref")
    tasks: list[dict[str, Any]] = []
    for report in sorted(args.results.glob("order-*.json")):
        video = args.videos / f"{report.stem}.mp4"
        tracks = json.loads(report.read_text(encoding="utf-8"))["tracks"]
        found = track_candidates(video, tracks, args.min_boxes, prepare=blurrer.blur)
        order = order_id(report.stem)
        items = dict(orders.get(order, {}))
        for track_id, candidates in sorted(found.items()):
            chosen = pick(candidates)
            folder = args.out / "crops" / report.stem
            folder.mkdir(parents=True, exist_ok=True)
            paths: list[str] = []
            for k, c in enumerate(chosen):
                path = folder / f"t{track_id}_{k}.jpg"
                bgr = cv2.cvtColor(c.crop, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                paths.append(path.relative_to(args.out).as_posix())
            crops: list[Image] = [c.crop for c in chosen]
            tasks.append(
                {
                    "id": f"{report.stem}:{track_id}",
                    "order": order,
                    "video": report.stem,
                    "crops": paths,
                    "order_items": items,
                    "suggested": suggest(models.embedder, models.index, crops, list(items)),
                    "quality": float(np.max([c.quality for c in chosen])),
                }
            )
        print(f"{report.stem}: {len(found)} items", flush=True)
    tasks.sort(key=lambda t: (t["order"], -t["quality"]))
    (args.out / "tasks.json").write_text(json.dumps(tasks, indent=1), encoding="utf-8")
    page = render(tasks, skus, sheet_id=f"{len(tasks)}-{tasks[0]['id'] if tasks else ''}")
    (args.out / "index.html").write_text(page, encoding="utf-8")
    print(f"{len(tasks)} items -> {args.out / 'index.html'}")
    return 0


def update(results: Path, videos: Path, out: Path, min_boxes: int) -> int:
    """Re-cut the crops of an existing sheet from the original frames and rebuild the page.

    ``results`` must be the run the sheet was made from. Task ids and the sheet id do not
    change, so saved or exported labels still apply. Crops are face-blurred for display, but
    never where a "face" is really product packaging.
    """
    # only the face detector is needed here (CPU); no detection or embedding models
    blurrer = FaceBlurrer(
        RotatedFaces(YuNetFaceDetector(Path("models") / FACE_MODEL, score_threshold=0.75))
    )
    tasks = json.loads((out / "tasks.json").read_text(encoding="utf-8"))
    page = (out / "index.html").read_text(encoding="utf-8")
    found_id = re.search(r'"id": "(\d+-[^"]+)"\};', page)
    sheet_id = found_id.group(1) if found_id else f"{len(tasks)}-{tasks[0]['id']}"
    by_video: dict[str, set[int]] = {}
    for t in tasks:
        by_video.setdefault(t["video"], set()).add(int(t["id"].rsplit(":", 1)[1]))
    for n, (stem, ids) in enumerate(sorted(by_video.items()), start=1):
        report = json.loads((results / f"{stem}.json").read_text(encoding="utf-8"))
        tracks = [t for t in report["tracks"] if int(t["track_id"]) in ids]
        found = track_candidates(videos / f"{stem}.mp4", tracks, min_boxes, pad=0.0)
        for t in (t for t in tasks if t["video"] == stem):
            chosen = pick(found.get(int(t["id"].rsplit(":", 1)[1]), []))
            for path, c in zip(t["crops"], chosen, strict=False):
                h, w = c.crop.shape[:2]
                shown = blurrer.blur(c.crop, protect=[(0, 0, w, h)])
                bgr = cv2.cvtColor(shown, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(out / path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"[{n}/{len(by_video)}] {stem}", flush=True)
    skus = write_refs(Path("data/gallery"), out / "ref")
    (out / "index.html").write_text(render(tasks, skus, sheet_id), encoding="utf-8")
    print(f"updated {len(tasks)} items -> {out / 'index.html'}")
    return 0


def suggest(embedder: Any, index: Any, crops: list[Image], allowed: list[str]) -> str | None:  # noqa: ANN401
    """Which of the order's products looks most like these crops (a hint, often wrong)."""
    if not allowed or not crops:
        return None
    if len(allowed) == 1:
        return allowed[0]
    refs, vectors = index.export()
    matrix = np.stack(vectors)
    names = np.array(refs)
    scores = {s: 0.0 for s in allowed}
    for v in embedder.embed_batch(crops):
        sims = matrix @ v
        for s in allowed:
            if (names == s).any():
                scores[s] += float(sims[names == s].max())
    return max(scores, key=scores.__getitem__)


if __name__ == "__main__":
    raise SystemExit(main())
