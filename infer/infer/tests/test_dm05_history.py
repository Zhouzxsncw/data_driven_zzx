"""头相机历史按 1 Hz 回看，当前帧不进历史槽。"""

from __future__ import annotations

import numpy as np

from infer.policies.dm05_history import HeadHistoryBuffer, head_index


def _rgb(value: int) -> np.ndarray:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[..., 0] = value
    return image


def test_head_index_prefers_mid_over_list_order() -> None:
    keys = [
        "observation.cam_left",
        "observation.cam_mid",
        "observation.cam_right",
    ]
    assert head_index(keys) == 1


def test_one_hertz_slots_exclude_current_frame() -> None:
    buf = HeadHistoryBuffer(
        slots=32,
        hz=1.0,
        tolerance_sec=0.2,
        store_min_interval_sec=0.0,
    )
    for t in range(0, 33):
        assert buf.record(float(t), _rgb(t))
    sample = buf.sample(32.0)
    assert sample.valid_slots == 32
    assert sample.newest_age_s == 1.0
    assert sample.oldest_age_s == 32.0


def test_gap_does_not_shift_newer_slots() -> None:
    buf = HeadHistoryBuffer(
        slots=4,
        hz=1.0,
        tolerance_sec=0.2,
        store_min_interval_sec=0.0,
    )
    buf.record(0.0, _rgb(0))
    buf.record(2.0, _rgb(2))
    sample = buf.sample(2.0)
    assert sample.valid_slots == 0


def test_prefix_before_start_is_dropped() -> None:
    buf = HeadHistoryBuffer(
        slots=32,
        hz=1.0,
        tolerance_sec=0.2,
        store_min_interval_sec=0.0,
    )
    for t in range(0, 6):
        buf.record(float(t), _rgb(t))
    sample = buf.sample(5.0)
    assert sample.valid_slots == 5
    assert sample.newest_age_s == 1.0
    assert sample.oldest_age_s == 5.0


def test_exact_tenth_second_frames_are_kept() -> None:
    buf = HeadHistoryBuffer(slots=4, hz=1.0, tolerance_sec=0.2, store_min_interval_sec=0.1)
    kept = 0
    for i in range(21):
        if buf.record(i * 0.1, _rgb(i)):
            kept += 1
    assert kept == 21


def test_dense_camera_is_thinned_before_sampling() -> None:
    buf = HeadHistoryBuffer(slots=2, hz=1.0, tolerance_sec=0.2, store_min_interval_sec=0.1)
    kept = 0
    for i in range(30):
        if buf.record(i / 30.0, _rgb(i)):
            kept += 1
    assert kept < 30
    assert kept >= 3
