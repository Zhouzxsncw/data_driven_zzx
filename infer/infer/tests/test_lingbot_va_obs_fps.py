"""LingBot-VA 真机 obs 应按训练 5fps（200ms）组 frame_chunk_size=2。"""

from __future__ import annotations

import numpy as np

from infer.policies.lingbot_va_runtime import LingbotVaWebsocketRuntime


def _rgb(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)


def test_record_camera_frame_downsamples_to_5fps() -> None:
    rt = LingbotVaWebsocketRuntime(host="127.0.0.1", port=1234, obs_fps=5.0)
    cams = ["observation.images.front", "observation.images.left", "observation.images.right"]
    rt.camera_keys = cams

    kept = 0
    for i in range(15):
        imgs = [_rgb(i), _rgb(100 + i), _rgb(200 + i)]
        if rt.record_camera_frame(imgs, timestamp=i / 15.0):
            kept += 1
    # 15Hz × 1s → 5fps 大约留下 6 帧（含 t=0）
    assert 5 <= kept <= 7
    assert len(rt._fps5_frames) == kept


def test_obs_chunk_uses_200ms_pair() -> None:
    rt = LingbotVaWebsocketRuntime(host="127.0.0.1", port=1234, obs_fps=5.0)
    rt.camera_keys = [
        "observation.images.front",
        "observation.images.left",
        "observation.images.right",
    ]
    t0 = 10.0
    for i in range(4):
        imgs = [_rgb(i), _rgb(10 + i), _rgb(20 + i)]
        rt.record_camera_frame(imgs, timestamp=t0 + i * 0.2)

    now = [_rgb(99), _rgb(199), _rgb(299)]
    fallback = rt._images_to_dict(now)
    obs_list, dt = rt._obs_pair_from_buffer(fallback)
    assert len(obs_list) == 2
    assert dt is not None
    assert abs(dt - 0.2) < 1e-6

    state = np.zeros(16, dtype=np.float32)
    obs = rt.build_observation(now, state)
    assert len(obs["obs"]) == 2
    assert "n_exec" not in obs
    rt.n_exec = 12
    obs = rt.build_observation(now, state)
    assert obs["n_exec"] == 12
    # 推理时刻那张 now 不能替代 200ms 前的 5fps 对
    np.testing.assert_array_equal(obs["obs"][0][rt.camera_keys[0]], rt._fps5_frames[-2][1][rt.camera_keys[0]])
    np.testing.assert_array_equal(obs["obs"][1][rt.camera_keys[0]], rt._fps5_frames[-1][1][rt.camera_keys[0]])
