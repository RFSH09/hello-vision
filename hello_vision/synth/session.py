"""Scripted synthetic shopping sessions with ground truth (demo video, simulator tests).

A hand enters from below the frame (the door side), takes a product off a shelf slot and
leaves, or brings one back. Everything is drawn from packshot-style sprites, so no real
footage is involved.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from pydantic import BaseModel

from hello_vision.synth.compose import Box, apply_lighting, draw_hand, make_background, place_sprite

WIDTH, HEIGHT, FPS = 960, 540, 15
NUM_SLOTS = 6
SLOT_CY = 250
PRODUCT_HEIGHT = 170
PRODUCT_MAX_WIDTH = 120
LEAD, GAP, TAIL = 8, 6, 8  # idle frames before / between / after actions
ACTION_LEN, CONTACT = 28, 13  # frames per action; frame at which product is grabbed/placed
SHELF_ZONE_Y_MAX = 0.7  # below this fraction of the height is the door zone


@dataclass(frozen=True)
class ScenarioSpec:
    """One scripted scenario.

    Attributes:
        actions: ``(kind, slot)`` per action, kind ``take`` or ``return``.
        gain: Lighting gain (below 1 = darker).
        gamma: Lighting gamma (above 1 = darker midtones).
        action_len: Frames per action (shorter = faster hand).
        concurrent: All actions happen at the same time with one hand each (two hands).
        held_under_hand: The carried product is drawn BEHIND the hand (heavy occlusion).
    """

    actions: tuple[tuple[str, int], ...]
    gain: float = 1.0
    gamma: float = 1.0
    action_len: int = ACTION_LEN
    concurrent: bool = False
    held_under_hand: bool = False


SCENARIOS: dict[str, ScenarioSpec] = {
    "take_one": ScenarioSpec((("take", 2),)),
    "take_two": ScenarioSpec((("take", 1), ("take", 4))),
    "take_and_return": ScenarioSpec((("take", 3), ("return", 3))),
    "take_nothing": ScenarioSpec(()),
    "swap": ScenarioSpec((("take", 1), ("return", 1), ("take", 4))),
    "take_one_low_light": ScenarioSpec((("take", 0),), gain=0.45, gamma=1.3),
    "fast_hand": ScenarioSpec((("take", 2),), action_len=14),
    "two_hands": ScenarioSpec((("take", 1), ("take", 4)), concurrent=True),
    "occluded": ScenarioSpec((("take", 3),), held_under_hand=True),
}


class CartItem(BaseModel):
    """Ground-truth cart line."""

    sku: str
    qty: int


class ActionTruth(BaseModel):
    """One TAKE or RETURN with its frame range and the frame of contact with the shelf."""

    kind: str
    sku: str
    slot: int
    start_frame: int
    contact_frame: int
    end_frame: int


class SessionTruth(BaseModel):
    """Ground truth of a synthetic session (what the recogniser should output)."""

    scenario: str
    seed: int
    session_id: str
    fridge_id: str
    fps: int
    width: int
    height: int
    num_frames: int
    door_open_frame: int
    door_close_frame: int
    shelf_zone_y_max: float
    slot_skus: dict[int, str]
    actions: list[ActionTruth]
    cart: list[CartItem]


@dataclass
class SessionClip:
    """Rendered session: RGB frames, per-frame boxes and ground truth."""

    frames: list[NDArray[np.uint8]]
    boxes: list[list[Box]]
    truth: SessionTruth


def _slot_x(slot: int) -> float:
    return 120.0 + slot * 144.0


def _contact_frame(action_len: int) -> int:
    """Frame (within an action) at which the product is grabbed or placed."""
    return round(action_len * CONTACT / ACTION_LEN)


def _hand_pos(slot: int, t: int, action_len: int = ACTION_LEN) -> tuple[float, float]:
    contact = _contact_frame(action_len)
    rest = (_slot_x(slot), float(HEIGHT + 260))
    target = (_slot_x(slot), SLOT_CY + 60.0)
    f = t / contact if t < contact else 1.0 - (t - contact) / max(1, action_len - 1 - contact)
    f = min(1.0, max(0.0, f))
    return (rest[0] + (target[0] - rest[0]) * f, rest[1] + (target[1] - rest[1]) * f)


def _draw_carried(
    img: Image.Image,
    boxes: list[Box],
    carried: list[tuple[int, tuple[float, float]]],
    sprites: dict[str, Image.Image],
    slot_sku: dict[int, str],
) -> None:
    """Draw every product currently held by a hand and record its box."""
    for slot, pos in carried:
        bb = place_sprite(img, sprites[slot_sku[slot]], pos, PRODUCT_HEIGHT, 0.0, PRODUCT_MAX_WIDTH)
        if bb:
            boxes.append(Box("product", slot_sku[slot], *bb))


def render_session(
    scenario: str,
    sprites: dict[str, Image.Image],
    seed: int = 0,
    fridge_id: str = "demo-fridge",
) -> SessionClip:
    """Render one scripted session.

    Args:
        scenario: Key of :data:`SCENARIOS`.
        sprites: SKU id -> RGBA cut-out; the first ``NUM_SLOTS`` fill the shelf slots.
        seed: Seed for backgrounds, jitter and noise (same seed -> same frames).
        fridge_id: Fridge id written to the truth.

    Returns:
        Frames, boxes and ground truth.

    Raises:
        KeyError: Unknown scenario.
        ValueError: Fewer sprites than shelf slots.
    """
    spec = SCENARIOS[scenario]
    plan, gain, gamma, alen = spec.actions, spec.gain, spec.gamma, spec.action_len
    contact = _contact_frame(alen)
    skus = list(sprites)
    if len(skus) < NUM_SLOTS:
        raise ValueError(f"need at least {NUM_SLOTS} sprites, got {len(skus)}")
    rng = np.random.default_rng(seed)
    slot_sku = {i: skus[i] for i in range(NUM_SLOTS)}
    angles = {i: float(rng.uniform(-4, 4)) for i in range(NUM_SLOTS)}
    bg = make_background((WIDTH, HEIGHT), rng)

    starts: list[int] = []
    cursor = LEAD
    for _ in plan:
        starts.append(cursor)
        if not spec.concurrent:
            cursor += alen + GAP
    if spec.concurrent and plan:
        cursor += alen + GAP
    num_frames = cursor + TAIL - GAP if plan else LEAD + TAIL + 10

    present = dict.fromkeys(range(NUM_SLOTS), True)
    actions: list[ActionTruth] = []
    cart: dict[str, int] = {}
    frames: list[NDArray[np.uint8]] = []
    all_boxes: list[list[Box]] = []

    for f in range(num_frames):
        img = bg.copy()
        boxes: list[Box] = []
        draw_slots = dict(present)
        carried: list[tuple[int, tuple[float, float]]] = []
        hands_at: list[tuple[float, float]] = []
        for idx, s0 in enumerate(starts):
            if not s0 <= f < s0 + alen:
                continue
            kind, slot = plan[idx]
            t = f - s0
            hand_at = _hand_pos(slot, t, alen)
            hands_at.append(hand_at)
            carry_pos = (hand_at[0], hand_at[1] - 40)
            if kind == "take":
                if not present[slot] and t == 0:
                    raise ValueError("take from an empty slot")
                draw_slots[slot] = t < contact
                if t >= contact:
                    carried.append((slot, carry_pos))
            else:
                if present[slot] and t == 0:
                    raise ValueError("return into an occupied slot")
                draw_slots[slot] = t >= contact
                if t < contact:
                    carried.append((slot, carry_pos))

        for slot, here in draw_slots.items():
            if here:
                bb = place_sprite(
                    img,
                    sprites[slot_sku[slot]],
                    (_slot_x(slot), SLOT_CY),
                    PRODUCT_HEIGHT,
                    angles[slot],
                    PRODUCT_MAX_WIDTH,
                )
                if bb:
                    boxes.append(Box("product", slot_sku[slot], *bb))

        # a grasped product is held in front of the fingers (drawn after the hand), unless the
        # scenario asks for heavy occlusion (drawn before the hand, i.e. behind it)
        if spec.held_under_hand:
            _draw_carried(img, boxes, carried, sprites, slot_sku)
        for hand_at in hands_at:
            hb = draw_hand(img, hand_at, rng, 1.0)
            if hb:
                boxes.append(Box("hand", None, *hb))
        if not spec.held_under_hand:
            _draw_carried(img, boxes, carried, sprites, slot_sku)

        img = apply_lighting(img, rng, gain, gamma, noise_sigma=2.0)
        frames.append(np.asarray(img, dtype=np.uint8))
        all_boxes.append(boxes)

        # finalise an action on its last frame
        for (kind, slot), s0 in zip(plan, starts, strict=True):
            if f == s0 + alen - 1:
                sku = slot_sku[slot]
                present[slot] = kind == "return"
                cart[sku] = cart.get(sku, 0) + (1 if kind == "take" else -1)
                actions.append(
                    ActionTruth(
                        kind=kind,
                        sku=sku,
                        slot=slot,
                        start_frame=s0,
                        contact_frame=s0 + contact,
                        end_frame=f,
                    )
                )

    truth = SessionTruth(
        scenario=scenario,
        seed=seed,
        session_id=f"{scenario}-{seed}",
        fridge_id=fridge_id,
        fps=FPS,
        width=WIDTH,
        height=HEIGHT,
        num_frames=num_frames,
        door_open_frame=0,
        door_close_frame=num_frames - 1,
        shelf_zone_y_max=SHELF_ZONE_Y_MAX,
        slot_skus=slot_sku,
        actions=actions,
        cart=[CartItem(sku=k, qty=v) for k, v in sorted(cart.items()) if v > 0],
    )
    return SessionClip(frames=frames, boxes=all_boxes, truth=truth)


def write_video(frames: list[NDArray[np.uint8]], path: Path, fps: int = FPS) -> None:
    """Write RGB frames to an MP4 file (created with parent folders).

    Raises:
        RuntimeError: If the video writer cannot be opened.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer for {path}")
    try:
        for fr in frames:
            writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
