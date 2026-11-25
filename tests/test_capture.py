from pathlib import Path

import cv2
import numpy as np
import pytest

from hello_vision.capture.door import MockDoorSensor, MockLock
from hello_vision.capture.recorder import RingBuffer, SessionBuilder
from hello_vision.capture.sources import ImageFolderSource, VideoFileSource
from hello_vision.capture.types import DoorEvent, DoorEventType, Frame
from hello_vision.synth.session import write_video


def _frame(i: int, cam: str = "c") -> Frame:
    return Frame(cam, i, i / 10, np.zeros((4, 4, 3), np.uint8))


def test_ring_buffer_keeps_latest() -> None:
    rb = RingBuffer(3)
    for i in range(5):
        rb.push(_frame(i))
    assert [f.index for f in rb.snapshot()] == [2, 3, 4] and len(rb) == 3
    with pytest.raises(ValueError):
        RingBuffer(-1)


def test_session_builder_with_pre_roll() -> None:
    b = SessionBuilder(pre_roll_frames=2)
    for i in range(4):
        b.push_frame(_frame(i))
    assert b.on_door(DoorEvent(DoorEventType.OPEN, 0.4, "s1")) is None and b.is_open
    b.on_door(DoorEvent(DoorEventType.OPEN, 0.5, "other"))  # repeated open ignored
    for i in range(4, 7):
        b.push_frame(_frame(i))
    s = b.on_door(DoorEvent(DoorEventType.CLOSE, 0.7))
    assert s is not None and s.session_id == "s1"
    assert [f.index for f in s.frames] == [2, 3, 4, 5, 6]
    assert s.duration == pytest.approx(0.3) and s.camera_ids == ["c"]
    assert not b.is_open


def test_close_without_open_ignored_and_generated_id() -> None:
    b = SessionBuilder()
    assert b.on_door(DoorEvent(DoorEventType.CLOSE, 1.0)) is None
    b.on_door(DoorEvent(DoorEventType.OPEN, 1.0))
    s = b.on_door(DoorEvent(DoorEventType.CLOSE, 2.0))
    assert s is not None and len(s.session_id) == 32


def test_mocks() -> None:
    ev = [DoorEvent(DoorEventType.OPEN, 0.0), DoorEvent(DoorEventType.CLOSE, 1.0)]
    assert list(MockDoorSensor(ev).events()) == ev
    lock = MockLock()
    lock.unlock()
    lock.lock()
    assert lock.calls == ["unlock", "lock"]


def test_video_and_folder_sources(tmp_path: Path) -> None:
    frames = [np.full((32, 48, 3), i * 20, np.uint8) for i in range(5)]
    write_video(frames, tmp_path / "v.mp4", fps=10)
    got = list(VideoFileSource(tmp_path / "v.mp4", "camX"))
    assert len(got) == 5 and got[0].camera_id == "camX" and got[0].image.shape == (32, 48, 3)
    assert got[3].timestamp == pytest.approx(0.3)

    folder = tmp_path / "imgs"
    folder.mkdir()
    for i, fr in enumerate(frames):
        cv2.imwrite(str(folder / f"{i:03d}.png"), fr)
    got2 = list(ImageFolderSource(folder, fps=5))
    assert [f.index for f in got2] == [0, 1, 2, 3, 4] and got2[2].timestamp == pytest.approx(0.4)


def test_missing_video_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list(VideoFileSource(tmp_path / "nope.mp4"))
