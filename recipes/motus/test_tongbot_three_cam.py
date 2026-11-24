#!/usr/bin/env python3
"""TongBot front/left/right must resolve to Motus three_cam, not front-only."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from camera_views import (  # noqa: E402
    resolve_image_source,
    require_three_cam_features,
    stitch_three_views_hwc,
)

TONGBOT = {
    "observation.images.left": {"dtype": "video", "shape": [3, 480, 640]},
    "observation.images.front": {"dtype": "video", "shape": [3, 480, 640]},
    "observation.images.right": {"dtype": "video", "shape": [3, 480, 640]},
}


def _resize_crop(img: np.ndarray, size) -> np.ndarray:
    th, tw = size
    out = np.zeros((th, tw, img.shape[2]), dtype=img.dtype)
    h = min(th, img.shape[0])
    w = min(tw, img.shape[1])
    out[:h, :w] = img[:h, :w]
    return out


def test_tongbot_keys_are_three_cam() -> None:
    mode, keys = resolve_image_source(TONGBOT)
    assert mode == "three_cam"
    assert keys == (
        "observation.images.front",
        "observation.images.left",
        "observation.images.right",
    )
    require_three_cam_features(TONGBOT)


def test_old_sorted_fallback_would_pick_front() -> None:
    visual = [k for k in TONGBOT]
    assert sorted(visual)[0] == "observation.images.front"


def test_agilex_and_concat_still_win() -> None:
    agilex = {
        "observation.images.cam_high": {"dtype": "video"},
        "observation.images.cam_left_wrist": {"dtype": "video"},
        "observation.images.cam_right_wrist": {"dtype": "video"},
    }
    mode, keys = resolve_image_source(agilex)
    assert mode == "three_cam"
    assert keys[0].endswith("cam_high")
    concat = {"observation.images.cam_concatenated": {"dtype": "video"}, **TONGBOT}
    mode, keys = resolve_image_source(concat)
    assert mode == "concatenated"
    assert keys == ("observation.images.cam_concatenated",)


def test_single_front_stays_single_view() -> None:
    mode, keys = resolve_image_source(
        {"observation.images.front": {"dtype": "video"}}
    )
    assert mode == "single_view"
    assert keys == ("observation.images.front",)


def test_t_stitch_layout() -> None:
    h, w = 40, 80
    front = np.full((h, w, 3), 10, dtype=np.uint8)
    left = np.full((h, w, 3), 80, dtype=np.uint8)
    right = np.full((h, w, 3), 200, dtype=np.uint8)
    front[..., 0] = 255
    left[..., 1] = 255
    right[..., 2] = 255
    out = stitch_three_views_hwc(front, left, right, _resize_crop)
    assert out.shape == (80, 80, 3)
    assert int(out[5, 40, 0]) == 255
    assert int(out[45, 10, 1]) == 255
    assert int(out[45, 70, 2]) == 255


def test_require_rejects_unmapped_triple() -> None:
    feats = {
        "observation.images.cam_a": {"dtype": "video"},
        "observation.images.cam_b": {"dtype": "video"},
        "observation.images.cam_c": {"dtype": "video"},
    }
    try:
        require_three_cam_features(feats)
    except ValueError as e:
        assert "three_cam" in str(e)
        return
    raise AssertionError("expected ValueError")


if __name__ == "__main__":
    test_tongbot_keys_are_three_cam()
    test_old_sorted_fallback_would_pick_front()
    test_agilex_and_concat_still_win()
    test_single_front_stays_single_view()
    test_t_stitch_layout()
    test_require_rejects_unmapped_triple()
    print("ok")
