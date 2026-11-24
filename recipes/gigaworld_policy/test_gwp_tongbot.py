"""GWP TongBot 16D 契约：不加载模型。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gwp_tongbot import (
    ACTION_CHUNK,
    CAM_FRONT,
    DELTA_MASK,
    TONGBOT_DIM,
    delta_to_abs,
    lookup_cam,
    obs_image_to_hwc_uint8,
    quantile_denormalize,
    quantile_normalize,
    resolve_transformer_dir,
    state16,
    views_from_obs,
)


def test_state16_pads_and_truncates() -> None:
    assert state16([1, 2, 3]).shape == (16,)
    assert state16(np.arange(20)).tolist() == list(range(16))


def test_quantile_roundtrip() -> None:
    feat = {
        "q01": [-2.0] * 16,
        "q99": [2.0] * 16,
    }
    x = np.zeros(16, dtype=np.float32)
    n = quantile_normalize(x, feat)
    assert np.allclose(n, 0.0, atol=1e-5)
    y = quantile_denormalize(n, feat)
    assert np.allclose(y, x, atol=1e-5)


def test_delta_to_abs_adds_joints_keeps_grippers() -> None:
    state = np.arange(16, dtype=np.float32)
    pred = np.ones((ACTION_CHUNK, TONGBOT_DIM), dtype=np.float32)
    out = delta_to_abs(pred, state)
    assert out.shape == (ACTION_CHUNK, 16)
    assert np.allclose(out[:, :14], 1.0 + state[:14])
    assert np.allclose(out[:, 14:], 1.0)
    assert DELTA_MASK.tolist() == [True] * 14 + [False] * 2


def test_cam_aliases_and_chw() -> None:
    img = np.zeros((3, 8, 10), dtype=np.float32)
    obs = {"observation.images.cam_high": img, CAM_FRONT.replace("front", "left"): img}
    # front via agilex alias
    got = lookup_cam({"observation.images.cam_high": img}, CAM_FRONT)
    hwc = obs_image_to_hwc_uint8(got)
    assert hwc.shape == (8, 10, 3)
    assert hwc.dtype == np.uint8


def test_views_from_obs_uint8() -> None:
    front = np.full((4, 6, 3), 10, dtype=np.uint8)
    left = np.full((4, 6, 3), 20, dtype=np.uint8)
    right = np.full((4, 6, 3), 30, dtype=np.uint8)
    f, l, r = views_from_obs(
        {
            "observation.images.front": front,
            "observation.images.left": left,
            "observation.images.right": right,
        }
    )
    assert f[0, 0, 0] == 10 and l[0, 0, 0] == 20 and r[0, 0, 0] == 30


def test_resolve_transformer_dir(tmp_path: Path) -> None:
    trans = tmp_path / "checkpoint_epoch_22_step_50000" / "transformer"
    trans.mkdir(parents=True)
    (trans / "config.json").write_text("{}", encoding="utf-8")
    (trans / "diffusion_pytorch_model.bin").write_bytes(b"x")
    assert resolve_transformer_dir(tmp_path / "checkpoint_epoch_22_step_50000") == trans.resolve()
    assert resolve_transformer_dir(trans) == trans.resolve()


def test_local_norm_stats_are_16d() -> None:
    path = Path("/home/xk/VLA/h100_ckpts/gwp05_tongbot_00073/norm_stats_00073.json")
    if not path.is_file():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    st = data["norm_stats"]["observation.state"]["q01"]
    act = data["norm_stats"]["action"]["q01"]
    assert len(st) == 16 and len(act) == 16


if __name__ == "__main__":
    test_state16_pads_and_truncates()
    test_quantile_roundtrip()
    test_delta_to_abs_adds_joints_keeps_grippers()
    test_cam_aliases_and_chw()
    test_views_from_obs_uint8()
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        test_resolve_transformer_dir(Path(d))
    test_local_norm_stats_are_16d()
    print("ok")
