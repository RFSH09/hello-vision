"""Generate a self-contained local viewer of the model's results (``uv run hv-viewer``).

Writes ``out/viewer/index.html`` (data inlined, works from ``file://`` without a server),
``thumbs/`` and ``crops/``. The page plays the ORIGINAL videos and draws boxes on a canvas.
Everything it reads and writes is private and must stay out of git.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image as PILImage

from hello_vision.capture.types import Image
from hello_vision.datatools.crops import pad_crop
from sim.orders import OrderScore, confused_pairs, count_items, merge_cameras, order_id
from sim.run_video import DARK, FACE_MODEL, SHELF_LINE, read_orders

DETECTORS = {"gdino": "Grounding DINO", "owlv2": "OWLv2"}


def video_info(path: Path) -> dict[str, float]:
    """Frame size, rate and duration of a video."""
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        return {
            "w": cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            "h": cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            "fps": round(fps, 3),
            "duration": round(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps, 2),
        }
    finally:
        cap.release()


def read_results(folder: Path) -> dict[str, dict[str, Any]]:
    """Video stem -> result JSON written by ``hv-demo`` / ``hv-batch`` in ``folder``."""
    out: dict[str, dict[str, Any]] = {}
    for p in sorted(folder.glob("*.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "tracks" in data:
            out[p.stem] = data
    return out


def find_runs(out: Path) -> dict[str, dict[str, Any]]:
    """The result sets to switch between: current, before the fixes, and detector trials."""
    runs: dict[str, dict[str, Any]] = {}
    for folder in (out, out / "baseline", out / "cmp_gdino", out / "cmp_owlv2"):
        results = read_results(folder) if folder.is_dir() else {}
        if not results:
            continue
        first = next(iter(results.values()))
        detector = str(first.get("detector") or folder.name.removeprefix("cmp_"))
        key = "before" if folder.name == "baseline" else detector
        if key in runs:
            continue
        has_boxes = all("boxes" in t for r in results.values() for t in r["tracks"])
        fixed = any("rejected_detections" in r for r in results.values())
        name = DETECTORS.get(detector, detector)
        if key == "before":
            name = "Before fixes"
        elif fixed:
            name = f"After fixes ({name})"
        runs[key] = {"name": name, "results": results, "boxes": has_boxes, "fixed": fixed}
    return runs


def session_run(path: Path, stems: set[str]) -> dict[str, Any]:
    """A run of the session model (carts per order), shaped like the other runs.

    ``path`` holds ``{name, note, orders: {id: {predicted, actual, flagged, videos}}}``, where
    ``actual`` is the cart mapped to the model's classes and ``videos`` maps a camera video
    to the frames the model weighted most.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    videos: dict[str, Any] = {}
    views: dict[str, Any] = {}
    scores: dict[str, OrderScore] = {}
    for order, item in data["orders"].items():
        score = OrderScore(order, Counter(item["predicted"]), Counter(item["actual"]))
        scores[order] = score
        status = "right" if score.exact else "partly" if score.correct else "wrong"
        if item["flagged"]:
            status = "review"
        views[order] = {
            "predicted": item["predicted"],
            "actual": item["actual"],
            "probabilities": item.get("probabilities", {}),
            "unknown": 0,
            "status": status,
        }
        views[order].update(correct=score.correct, wrong=score.wrong, missed=score.missed)
        for stem, tracks in item["videos"].items():
            if stem in stems:
                videos[stem] = {"tracks": tracks, "hands": None}
    cells: dict[str, dict[str, Any]] = {}
    for order, s in scores.items():
        pairs = Counter({(sku, sku): n for sku, n in (s.predicted & s.actual).items()})
        pairs.update(confused_pairs([s]))
        if not s.predicted - s.actual:
            pairs.update({(sku, ""): 1 for sku in s.actual - s.predicted})
        for (truth, pred), n in pairs.items():
            cell = cells.setdefault(f"{truth}|{pred}", {"n": 0, "orders": []})
            cell["n"] += n
            cell["orders"].append(order)
    correct = sum(s.correct for s in scores.values())
    actual = sum(sum(s.actual.values()) for s in scores.values())
    return {
        "name": data["name"],
        "note": data["note"],
        "about_orders": {
            o: {"fridge": item.get("fridge", ""), "items": item.get("items", item["actual"])}
            for o, item in data["orders"].items()
        },
        "boxes": True,
        "videos": videos,
        "orders": views,
        "summary": {
            "orders": len(scores),
            "exact": sum(views[o]["status"] == "right" for o in scores),
            "correct": correct,
            "wrong": sum(s.wrong for s in scores.values()),
            "missed": actual - correct,
            "actual": actual,
            "fps": "-",
            "review": sum(v["status"] == "review" for v in views.values()),
            "confusion": cells,
        },
        "mistakes": [],
        "flagged": [],
    }


def best_box(track: dict[str, Any]) -> list[float] | None:
    """The box used for the inspector crop: highest detector score, held ones first."""
    boxes = track.get("boxes") or []
    if not boxes:
        return None
    held = [b for b in boxes if b[6]] or boxes
    return list(max(held, key=lambda b: b[5]))


def save_crops(
    video: Path,
    tracks: Sequence[dict[str, Any]],
    dest: Path,
    stem: str,
    blurrer: Any,  # noqa: ANN401
) -> dict[int, Image]:
    """Re-read the video once and save each track's best crop as a small JPG.

    Returns the RAW crops (for identification); only the saved JPG is face-blurred.
    """
    wanted: dict[float, list[tuple[int, list[float]]]] = {}
    for t in tracks:
        box = best_box(t)
        if box:
            wanted.setdefault(round(box[0], 2), []).append((int(t["track_id"]), box))
    crops: dict[int, Image] = {}
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    index = -1
    try:
        while wanted and cap.grab():
            index += 1
            key = round(index / fps, 2)
            if key not in wanted:
                continue
            ok, bgr = cap.retrieve()
            if not ok:
                continue
            frame = np.asarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)
            for track_id, box in wanted.pop(key):
                crops[track_id] = pad_crop(frame, box[1:5], 0.0)
                shown = pad_crop(frame, box[1:5])
                if blurrer:
                    h, w = shown.shape[:2]
                    shown = blurrer.blur(shown, protect=[(0, 0, w, h)])
                scale = min(1.0, 320 / max(shown.shape[:2]))
                small = cv2.resize(shown, None, fx=scale, fy=scale)
                cv2.imwrite(
                    str(dest / f"{stem}_t{track_id}.jpg"),
                    cv2.cvtColor(small, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 85],
                )
    finally:
        cap.release()
    return crops


def top_matches(vector: Any, skus: np.ndarray[Any, Any], matrix: Any, n: int = 3) -> list[Any]:  # noqa: ANN401
    """The ``n`` most similar SKUs (best reference view each) with their cosine similarity."""
    sims = matrix @ vector
    best = {str(s): float(sims[skus == s].max()) for s in np.unique(skus)}
    return [[s, round(v, 3)] for s, v in sorted(best.items(), key=lambda kv: -kv[1])[:n]]


def order_view(
    order: str,
    results: dict[str, dict[str, Any]],
    actual: Counter[str] | None,
) -> dict[str, Any]:
    """Cart comparison of one order for one run."""
    carts = [count_items(r["tracks"]) for r in results.values()]
    cart = merge_cameras(carts)
    unknown = cart.pop("unknown", 0)
    view: dict[str, Any] = {"predicted": dict(cart), "unknown": unknown, "status": "nodata"}
    if actual is not None:
        s = OrderScore(order, cart, actual)
        view["status"] = "right" if s.exact else "partly" if s.correct else "wrong"
        view.update(correct=s.correct, wrong=s.wrong, missed=s.missed)
    return view


def summary(run: dict[str, Any], orders: dict[str, Counter[str]]) -> dict[str, Any]:
    """Top-bar numbers and the confusion grid of one run."""
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for stem, r in run["results"].items():
        grouped.setdefault(order_id(stem), {})[stem] = r
    scores: list[OrderScore] = []
    for order, results in sorted(grouped.items()):
        if order in orders:
            cart = merge_cameras([count_items(r["tracks"]) for r in results.values()])
            cart.pop("unknown", 0)
            scores.append(OrderScore(order, cart, orders[order]))
    cells: dict[str, dict[str, Any]] = {}

    def add(truth: str, pred: str, count: int, order: str) -> None:
        cell = cells.setdefault(f"{truth}|{pred}", {"n": 0, "orders": []})
        cell["n"] += count
        cell["orders"].append(order)

    for s in scores:
        for sku, n in (s.predicted & s.actual).items():
            add(sku, sku, n, s.order)
        for (truth, pred), n in confused_pairs([s]).items():
            add(truth, pred, n, s.order)
        if not s.predicted - s.actual:
            for sku in s.actual - s.predicted:
                add(sku, "", 1, s.order)
    correct = sum(s.correct for s in scores)
    wrong = sum(s.wrong for s in scores)
    actual = sum(sum(s.actual.values()) for s in scores)
    frames = sum(r.get("processed_frames", 0) for r in run["results"].values())
    seconds = sum(r.get("seconds", 0) for r in run["results"].values())
    return {
        "orders": len(scores),
        "exact": sum(s.exact for s in scores),
        "correct": correct,
        "wrong": wrong,
        "missed": actual - correct,
        "actual": actual,
        "fps": round(frames / seconds, 2) if seconds else 0,
        "confusion": cells,
    }


def compute_hands(videos: Sequence[Path], dest: Path, backend: str, fps: float = 5.0) -> None:
    """Detect hands on the same sampled frames and store them next to the results (overlay only)."""
    from hello_vision.detect.zeroshot import ZeroShotDetector

    detector = ZeroShotDetector(backend)
    dest.mkdir(parents=True, exist_ok=True)
    for n, video in enumerate(videos, start=1):
        target = dest / f"{video.stem}.json"
        if target.exists():
            continue
        cap = cv2.VideoCapture(str(video))
        src = cap.get(cv2.CAP_PROP_FPS) or 25.0
        index, next_t, hands = -1, 0.0, []
        while cap.grab():
            index += 1
            t = index / src
            if t + 1e-9 < next_t:
                continue
            next_t += 1.0 / fps
            ok, bgr = cap.retrieve()
            if not ok or float(bgr.mean()) < DARK:
                continue
            for d in detector.detect(
                np.asarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)
            ):
                if d.label == "hand":
                    box = [round(d.x1), round(d.y1), round(d.x2), round(d.y2)]
                    hands.append([round(t, 2), *box, round(d.score, 3)])
        cap.release()
        target.write_text(json.dumps(hands), encoding="utf-8")
        print(f"hands [{n}/{len(videos)}] {video.stem}: {len(hands)} boxes", flush=True)


def mistakes(run: dict[str, Any], orders: dict[str, Counter[str]], n: int = 10) -> list[Any]:
    """Examples of remaining errors, alternating wrong and missed items, one per order first."""
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for stem, r in run["results"].items():
        grouped.setdefault(order_id(stem), {})[stem] = r
    wrong: list[dict[str, Any]] = []
    lost: list[dict[str, Any]] = []
    for order, results in sorted(grouped.items()):
        if order not in orders:
            continue
        cart = merge_cameras([count_items(r["tracks"]) for r in results.values()])
        cart.pop("unknown", 0)
        tracks = [(stem, t) for stem, r in results.items() for t in r["tracks"]]
        for sku in sorted(cart - orders[order]):
            hits = [x for x in tracks if x[1]["taken_guess"] and x[1]["sku"] == sku]
            stem, t = max(hits, key=lambda x: x[1]["frames"])
            wrong.append(
                {"order": order, "kind": "wrong", "sku": sku, "stem": stem,
                 "track": t["track_id"],
                 "why": f"counted {sku}, which was not bought ({t['frames']} frames)"}
            )  # fmt: skip
        for sku in sorted(orders[order] - cart):
            guessed = [x for x in tracks if x[1]["hand_touched"] and x[1]["best_guess"] == sku]
            named = [x for x in tracks if x[1]["sku"] == sku]
            unnamed = [x for x in tracks if x[1]["taken_guess"] and not x[1]["sku"]]
            pool = named or guessed or unnamed
            if named:
                why = "named correctly on a track, but it was not counted as taken"
            elif guessed:
                why = "closest guess on a track was right, but below the threshold (unknown)"
            elif unnamed:
                why = "taken items stayed unknown; this may be one of them"
            else:
                why = "no held track at all: not detected or rejected as not-product"
            item: dict[str, Any] = {"order": order, "kind": "missed", "sku": sku, "why": why}
            if pool:
                stem, t = max(pool, key=lambda x: x[1]["frames"])
                item.update(stem=stem, track=t["track_id"])
            lost.append(item)
    chosen: list[dict[str, Any]] = []
    seen: set[str] = set()
    queues = [lost, wrong]
    for unique in (True, False):
        k = 0
        while len(chosen) < n and any(queues):
            queue = queues[k % 2]
            k += 1
            pick_one = next((m for m in queue if not unique or m["order"] not in seen), None)
            if pick_one is not None:
                queue.remove(pick_one)
                chosen.append(pick_one)
                seen.add(pick_one["order"])
            elif unique and all(not any(m["order"] not in seen for m in q) for q in queues):
                break
    return chosen


def build(data: Path, out: Path, use_models: bool = True, labels: Path | None = None) -> Path:
    """Write the viewer and return the path of ``index.html``."""
    videos = data / "videos"
    manifest_path = videos / "manifest.json"
    manifest = (
        {m["video_id"]: m for m in json.loads(manifest_path.read_text(encoding="utf-8"))}
        if manifest_path.exists()
        else {}
    )
    orders_path = data / "orders.csv"
    try:
        orders = read_orders(orders_path) if orders_path.exists() else {}
    except (KeyError, ValueError, csv.Error) as exc:
        raise SystemExit(f"cannot read {orders_path}: {exc}") from exc
    runs = find_runs(out)
    if not runs:
        raise SystemExit(f"no result JSON files in {out}; run hv-batch first")
    viewer = out / "viewer"
    (viewer / "thumbs").mkdir(parents=True, exist_ok=True)
    (viewer / "crops").mkdir(parents=True, exist_ok=True)
    skus: list[str] = []
    for p in sorted((data / "gallery").glob("*.png")):
        with PILImage.open(p) as im:
            rgba = im.convert("RGBA")
            white = PILImage.new("RGBA", rgba.size, (255, 255, 255, 255))
            white.alpha_composite(rgba)
            white.thumbnail((120, 120))
            white.convert("RGB").save(viewer / "thumbs" / f"{p.stem}.jpg", quality=88)
        skus.append(p.stem)

    blurrer = embedder = base = real = None
    models_dir = Path("models")
    if use_models and (models_dir / FACE_MODEL).is_file():
        from sim.run_video import load_models

        try:
            models = load_models(gallery_dir=data / "gallery")
            blurrer, embedder, base = models.blurrer, models.embedder, models.index
            if labels is not None and labels.exists():
                from hello_vision.datatools.real_crops import load_or_build

                real = load_or_build(labels, out / "baseline", videos, embedder, models_dir)
        except ImportError as exc:
            print(f"top-3 matches skipped (install the vision extra): {exc}")
    if blurrer is None:
        print("WARNING: models unavailable, so crops are NOT face-blurred and have no top-3.")

    payload_runs: dict[str, Any] = {}
    stems: set[str] = set()
    for key, run in runs.items():
        entry: dict[str, Any] = {"name": run["name"], "boxes": run["boxes"], "videos": {}}
        for n, (stem, r) in enumerate(run["results"].items(), start=1):
            video = videos / f"{stem}.mp4"
            if not video.is_file():
                continue
            stems.add(stem)
            crops: dict[int, Image] = {}
            if run["boxes"]:
                crops = save_crops(video, r["tracks"], viewer / "crops", f"{key}_{stem}", blurrer)
            tops: dict[int, list[Any]] = {}
            if embedder is not None and base is not None and crops:
                # the gallery that run used for this order (never containing its own crops)
                gallery = base
                if real is not None and run["fixed"]:
                    gallery = real.gallery(base, order_id(stem))
                names, matrix_rows = gallery.export()
                refs, matrix = np.array(names), np.stack(matrix_rows)
                ids = list(crops)
                for track_id, v in zip(
                    ids, embedder.embed_batch([crops[i] for i in ids]), strict=True
                ):
                    tops[track_id] = top_matches(v, refs, matrix)
            hands = r.get("hands")
            side = out / "hands" / f"{stem}.json"
            if hands is None and key in ("gdino", "before") and side.exists():
                hands = json.loads(side.read_text(encoding="utf-8"))
            tracks = []
            for t in r["tracks"]:
                tid = int(t["track_id"])
                tracks.append(
                    {
                        **t,
                        "crop": f"crops/{key}_{stem}_t{tid}.jpg" if tid in crops else None,
                        "top": tops.get(tid),
                        "best": best_box(t),
                    }
                )
            entry["videos"][stem] = {"tracks": tracks, "hands": hands}
            print(f"{run['name']} [{n}/{len(run['results'])}] {stem}", flush=True)
        grouped: dict[str, dict[str, Any]] = {}
        for stem, r in run["results"].items():
            grouped.setdefault(order_id(stem), {})[stem] = r
        entry["orders"] = {o: order_view(o, g, orders.get(o)) for o, g in grouped.items()}
        entry["summary"] = summary(run, orders)
        entry["mistakes"] = mistakes(run, orders) if run["boxes"] else []
        entry["flagged"] = sorted(
            {
                order_id(s)
                for s, r in run["results"].items()
                for t in r["tracks"]
                if t.get("multiple")
            }
        )
        payload_runs[key] = entry

    # the session model may cover many more orders, whose videos sit with the order history
    files = {p.stem: p for p in (data / "history" / "videos").glob("*.mp4")}
    files.update({p.stem: p for p in videos.glob("*.mp4")})
    session_keys: list[str] = []
    extra: dict[str, dict[str, Any]] = {}  # fridge and items of orders orders.csv lacks
    for path in sorted((out / "session").glob("*.json"), reverse=True):
        run = session_run(path, set(files))
        extra.update(run.pop("about_orders"))
        stems.update(run["videos"])
        payload_runs[path.stem] = run
        session_keys.append(path.stem)
    if session_keys:
        with PILImage.new("RGB", (120, 120), (90, 90, 90)) as blank:
            blank.save(viewer / "thumbs" / "other.jpg")

    order_list: dict[str, dict[str, Any]] = {}
    for stem in sorted(stems):
        o = order_id(stem)
        known = extra.get(o, {})
        item = order_list.setdefault(
            o,
            {
                "id": o,
                "fridge": manifest.get(o, {}).get("fridge_key", "") or known.get("fridge", ""),
                "actual": dict(orders[o]) if o in orders else known.get("items"),
                "videos": [],
            },
        )
        source = files.get(stem, videos / f"{stem}.mp4")
        relative = Path(os.path.relpath(source, viewer)).as_posix()  # works from file://
        item["videos"].append({"stem": stem, "src": relative, **video_info(source)})
    payload = {
        "runs": payload_runs,
        "orders": sorted(order_list.values(), key=lambda x: (x["actual"] is None, x["id"])),
        "skus": skus,
        "shelfLine": SHELF_LINE,
        "idThreshold": 0.35,
        "order": session_keys + [k for k in ("gdino", "before", "owlv2") if k in payload_runs],
        "idMargin": 0.02,
    }
    template = (Path(__file__).parent / "viewer_template.html").read_text(encoding="utf-8")
    page = template.replace("__DATA__", json.dumps(payload).replace("<", "\\u003c"))
    index = viewer / "index.html"
    index.write_text(page, encoding="utf-8")
    return index


def main(argv: Sequence[str] | None = None) -> int:
    """``hv-viewer``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("out"))
    parser.add_argument("--no-models", action="store_true", help="no blur/top-3 (fast)")
    parser.add_argument(
        "--compute-hands", action="store_true", help="first detect hand boxes (slow, needs GPU)"
    )
    args = parser.parse_args(argv)
    if args.compute_hands:
        videos = sorted(args.data / "videos" / f"{stem}.mp4" for stem in read_results(args.out))
        compute_hands(videos, args.out / "hands", "gdino")
    index = build(args.data, args.out, not args.no_models, args.data / "labels.json")
    print(f"\nOpen: {index.resolve()}")
    print(
        "1. Top bar: exact carts, item recall/precision, missed, wrong, speed; detector switch.\n"
        "2. Left: orders with right/partly/wrong badges, filters and product search.\n"
        "3. Both cameras in sync with box overlays, layer toggles, frame stepping and a timeline.\n"
        "4. Cart: model vs actual; click a track to see its crop and top-3 gallery matches.\n"
        "5. Confusion tab: true vs predicted products; notes per order are saved in the browser."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
