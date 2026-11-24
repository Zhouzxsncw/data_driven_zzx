"""GR00T 00036 观测构造：不连服务器，只验证与训练模态对齐。"""

from __future__ import annotations

import numpy as np

from infer.core.types import PolicyType
from infer.policies.gr00t_runtime import (
    DEFAULT_LANG_KEY,
    DEFAULT_STATE_KEY,
    DEFAULT_VIDEO_KEYS,
    Gr00tZmqRuntime,
    _squeeze_action,
    _to_hwc_uint8,
)


def test_policy_type_gr00t_aliases() -> None:
    from infer.config_loader import _parse_policy_type

    assert _parse_policy_type("gr00t") is PolicyType.GR00T
    assert _parse_policy_type("groot") is PolicyType.GR00T
    assert _parse_policy_type("gr00t_n17") is PolicyType.GR00T


def test_to_hwc_uint8_chw_float() -> None:
    chw = np.zeros((3, 4, 5), dtype=np.float32)
    chw[0, 0, 0] = 1.0
    out = _to_hwc_uint8(chw)
    assert out.shape == (4, 5, 3)
    assert out.dtype == np.uint8
    assert out[0, 0, 0] == 255


def test_build_observation_matches_00036_config() -> None:
    rt = Gr00tZmqRuntime()
    images = [
        np.zeros((480, 640, 3), dtype=np.uint8),
        np.ones((480, 640, 3), dtype=np.uint8) * 10,
        np.ones((480, 640, 3), dtype=np.uint8) * 20,
    ]
    state = np.arange(16, dtype=np.float32)
    prompt = (
        "With both hands, pick up the yellow banana and the yellow lemon "
        "and put them into the black box; then take the green vegetables "
        "and place them into the blue box."
    )
    obs = rt.build_observation(
        images,
        state,
        prompt,
        camera_keys=[
            "observation.cam_left",
            "observation.cam_mid",
            "observation.cam_right",
        ],
    )
    assert set(obs.keys()) == {"video", "state", "language"}
    assert list(obs["video"].keys()) == DEFAULT_VIDEO_KEYS
    for key in DEFAULT_VIDEO_KEYS:
        arr = obs["video"][key]
        assert arr.shape == (1, 1, 480, 640, 3)
        assert arr.dtype == np.uint8
    st = obs["state"][DEFAULT_STATE_KEY]
    assert st.shape == (1, 1, 16)
    assert st.dtype == np.float32
    assert np.allclose(st[0, 0], state)
    assert obs["language"][DEFAULT_LANG_KEY] == [[prompt]]


def test_load_00036_yaml_policy_fields() -> None:
    from pathlib import Path

    from infer.config_loader import load_infer_config, validate_infer_config

    path = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "lingbot"
        / "gr00t_tongbot_16d_00036.yaml"
    )
    cfg = load_infer_config(path)
    assert cfg.policy.type is PolicyType.GR00T
    assert cfg.policy.gr00t_port == 5555
    assert cfg.policy.gr00t_language_key == "annotation.human.task_description"
    assert cfg.policy.gr00t_video_keys == [
        "observation.images.cam_left",
        "observation.images.cam_mid",
        "observation.images.cam_right",
    ]
    assert cfg.policy.keep_dims == list(range(16))
    validate_infer_config(cfg)

