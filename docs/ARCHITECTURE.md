# Architecture

Status: design (M0). Nothing below is implemented yet.

One **shopping session** = door_open -> frames from all cameras -> door_close.

1. **Session events / capture** - adapters: live camera (RTSP/USB), video file, image folder
   (replay and tests). A ring buffer holds recent frames; door events delimit the session.
2. **Detection** - class-agnostic `product` detector and `hand` detector per frame (ONNX).
3. **Identification (two-stage)** - crop each product, embed it, match against an SKU gallery
   (packshots + real crops) by cosine similarity. New SKUs are added by adding gallery images,
   without retraining the detector. Below the similarity threshold the outcome is `unknown`.
4. **Tracking + action logic** - track products and hands across frames; decide TAKE or
   RETURN per item from zones (shelf vs door/outside) and hand-product association.
5. **Shelf-state cross-check** - before/after shelf snapshots give per-shelf SKU counts; the
   difference must agree with step 4. Both signals are fused into the final cart.
6. **Decision** - `cart = [{sku, qty, confidence}]` plus session confidence. Low confidence,
   conflicting evidence or an unknown product sets `needs_review = true`: a human checks the
   clip and the customer is not charged a guessed item.
7. **Output** - POST to the backend ([API.md](API.md)), store an evidence clip (faces blurred,
   detections overlaid), keep metrics.

## Package layout

`hello_vision/{catalog,capture,detect,identify,track,session,privacy,service,review,edge}`,
plus `training/`, `eval/`, `tools/`, `sim/`, `configs/`, `docker/`.

## Principles

- Hardware-agnostic until [HARDWARE_QUESTIONS.md](HARDWARE_QUESTIONS.md) is answered; lock and
  door sensor sit behind an interface with a mock. No real lock control before pilot approval.
- Never guess a charge: uncertainty becomes review, not money.
- Data location is configuration; everything must be able to run fully in Uzbekistan.
- Permissive licenses only ([LICENSES.md](LICENSES.md)).

## Go/no-go gate (proposed, to be confirmed by the owner)

On shadow data: >= 99% exact cart accuracy for 2 weeks, false-charge rate < 0.2%,
review rate < 3%, result < 5 s after door close on the target edge device.
