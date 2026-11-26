"""Local Gradio demo: pick or upload a video, see the annotated clip, tracks and guessed cart.

    uv run --extra vision --extra app hv-app

Runs on 127.0.0.1 only, with Gradio sharing and analytics switched off: the videos are private.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from hello_vision.capture.types import Image
from hello_vision.datatools.crops import pad_crop
from sim.orders import count_items, fmt, order_id
from sim.run_video import VIDEO_SUFFIXES, Models, load_models, process_video, speed_line

COLUMNS = ["track", "product", "best guess", "confidence", "first s", "last s", "held", "taken?"]


def track_rows(tracks: Sequence[dict[str, Any]]) -> list[list[Any]]:
    """Table rows, items held by a hand first."""
    ordered = sorted(tracks, key=lambda t: (not t["taken_guess"], not t["hand_touched"]))
    return [
        [
            t["track_id"],
            t["sku"] or "unknown",
            t["best_guess"] or "",
            t["confidence"],
            t["first_seen_s"],
            t["last_seen_s"],
            "yes" if t["hand_touched"] else "",
            "yes" if t["taken_guess"] else "",
        ]
        for t in ordered
    ]


def thumbnails(
    video: Path, tracks: Sequence[dict[str, Any]], models: Models, limit: int = 24
) -> list[tuple[Image, str]]:
    """One face-blurred crop (highest detector score) for each held track."""
    held = [t for t in tracks if t["hand_touched"]]
    held.sort(key=lambda t: (not t["taken_guess"], -t["frames"]))
    cap = cv2.VideoCapture(str(video))
    out: list[tuple[Image, str]] = []
    try:
        for t in held[:limit]:
            boxes = [b for b in t["boxes"] if b[6]] or t["boxes"]
            best = max(boxes, key=lambda b: b[5])
            cap.set(cv2.CAP_PROP_POS_MSEC, best[0] * 1000)
            ok, bgr = cap.read()
            if not ok:
                continue
            frame = np.asarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)
            if models.blurrer:
                frame = models.blurrer.blur(frame)
            label = f"#{t['track_id']} {t['sku'] or 'unknown'}"
            out.append((pad_crop(frame, best[1:5]), label))
    finally:
        cap.release()
    return out


def cart_text(tracks: Sequence[dict[str, Any]]) -> str:
    """Human readable cart guess."""
    cart: Counter[str] = count_items(tracks)
    unknown = cart.pop("unknown", 0)
    text = f"Guessed cart (naive, NOT reliable): {fmt(cart)}"
    return text + (f"\nHeld but unidentified: {unknown}" if unknown else "")


def analyse(
    video: Path, models: Models, out_dir: Path, fps: float
) -> tuple[str, list[list[Any]], list[tuple[Image, str]], str]:
    """Process one video; returns ``(annotated video, rows, thumbnails, summary text)``."""
    # WebM so that the browser can play it (this OpenCV build cannot write H.264)
    result = process_video(video, out_dir / f"{video.stem}.webm", models, fps)
    summary = f"{cart_text(result.tracks)}\n{speed_line(result)}"
    if order_id(video.stem) != video.stem:
        summary += "\nThis is one camera of an order; the other camera may see more."
    return (
        result.annotated,
        track_rows(result.tracks),
        thumbnails(video, result.tracks, models),
        summary,
    )


def build_app(videos: Path, out_dir: Path, models_factory: Any = load_models) -> Any:  # noqa: ANN401
    """Assemble the Gradio interface; models load on the first run."""
    import gradio as gr

    cache: dict[str, Models] = {}
    choices = sorted(p.name for p in videos.glob("*") if p.suffix.lower() in VIDEO_SUFFIXES)

    def run(name: str | None, upload: str | None, fps: float) -> tuple[Any, ...]:
        path = Path(upload) if upload else (videos / name if name else None)
        if path is None:
            raise gr.Error("Pick a video or upload one.")
        if "m" not in cache:
            cache["m"] = models_factory()
        return analyse(path, cache["m"], out_dir, fps)

    with gr.Blocks(title="hello-vision demo", analytics_enabled=False) as app:
        gr.Markdown(
            "## hello-vision demo\nUntrained zero-shot baseline. Expect wrong and missing items."
        )
        with gr.Row():
            name = gr.Dropdown(choices, label="Video from data/videos", value=None)
            upload = gr.Video(label="...or upload a video", sources=["upload"])
        fps = gr.Slider(1, 10, value=5, step=1, label="Frames per second to process")
        button = gr.Button("Run", variant="primary")
        summary = gr.Textbox(label="Result", lines=4)
        video = gr.Video(label="Annotated video")
        gallery = gr.Gallery(label="Items held by a hand", columns=6, height="auto")
        table = gr.Dataframe(headers=COLUMNS, label="Tracks", interactive=False)
        button.click(run, [name, upload, fps], [video, table, gallery, summary])
    return app


def main(argv: Sequence[str] | None = None) -> int:
    """``hv-app``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, default=Path("data/videos"))
    parser.add_argument("--out", type=Path, default=Path("out/app"))
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args(argv)
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    app = build_app(args.videos, args.out)
    app.launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=False,
        allowed_paths=[str(args.out.resolve())],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
