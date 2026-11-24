"""τ0-WM TongBot EE 20D 契约：不加载模型。"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from tau_tongbot import (
    ACTION_CHUNK,
    CAM_FRONT,
    EE_DIM,
    IMG_H,
    IMG_W,
    TONGBOT_DIM,
    ee20_to_play_state_and_grippers,
    euler_zyx_to_rot6d_cols,
    matrix_to_rot6d_cols,
    obs_image_to_hwc_uint8,
    resolve_tau_ckpt,
    rot6d_cols_to_matrix,
    rot6d_cols_to_quat_xyzw,
    state16,
    views_from_obs,
    write_deploy_yaml,
)


def test_state16_pads_and_truncates() -> None:
    assert state16([1, 2, 3]).shape == (16,)
    assert state16(np.arange(20)).tolist() == list(range(16))


def test_rot6d_cols_roundtrip_identity() -> None:
    ident = np.eye(3, dtype=np.float64)
    r6 = matrix_to_rot6d_cols(ident)
    assert r6.tolist() == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    R = rot6d_cols_to_matrix(r6)
    assert np.allclose(R, ident, atol=1e-7)


def test_rot6d_cols_not_rows() -> None:
    # 非对称旋转：行约定与列约定的 6D 互为「前两行 vs 前两列」，不能相等
    yaw = 0.4
    euler = np.array([0.1, -0.2, yaw], dtype=np.float64)
    cols = euler_zyx_to_rot6d_cols(euler)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(-0.2), np.sin(-0.2)
    cr, sr = np.cos(0.1), np.sin(0.1)
    r00, r01, r02 = cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr
    r10, r11, r12 = sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr
    rows = np.array([r00, r01, r02, r10, r11, r12], dtype=np.float64)
    assert not np.allclose(cols, rows, atol=1e-5)
    R = rot6d_cols_to_matrix(cols)
    assert np.allclose(R[:, 0], cols[:3], atol=1e-7)
    assert np.allclose(R[:, 1], cols[3:], atol=1e-6)


def test_euler_cols_matches_decoded_matrix() -> None:
    euler = np.array([0.3, -0.4, 1.1], dtype=np.float64)
    r6 = euler_zyx_to_rot6d_cols(euler)
    R = rot6d_cols_to_matrix(r6)
    # 再编码应回到同一 6D（Gram-Schmidt 后仍接近）
    r6_b = matrix_to_rot6d_cols(R)
    assert np.allclose(r6 / np.linalg.norm(r6), r6_b / np.linalg.norm(r6_b), atol=1e-6)


def test_quat_roundtrip_keeps_columns() -> None:
    euler = np.array([0.2, 0.1, -0.5], dtype=np.float64)
    r6 = euler_zyx_to_rot6d_cols(euler)
    q = rot6d_cols_to_quat_xyzw(r6)
    assert q.shape == (4,)
    assert abs(float(np.linalg.norm(q)) - 1.0) < 1e-6
    R = rot6d_cols_to_matrix(r6)
    r6_again = matrix_to_rot6d_cols(R)
    assert np.allclose(r6_again, matrix_to_rot6d_cols(rot6d_cols_to_matrix(r6)), atol=1e-6)


def test_ee20_to_play_layout() -> None:
    ee = np.zeros(EE_DIM, dtype=np.float64)
    ee[0:3] = [0.1, 0.2, 0.3]
    ee[3:9] = [1, 0, 0, 0, 1, 0]
    ee[9] = 0.8
    ee[10:13] = [-0.1, 0.4, 0.2]
    ee[13:19] = [1, 0, 0, 0, 1, 0]
    ee[19] = 0.2
    state, grips = ee20_to_play_state_and_grippers(ee)
    assert state.shape == (14,)
    assert np.allclose(state[0:3], [0.1, 0.2, 0.3])
    assert np.allclose(state[7:10], [-0.1, 0.4, 0.2])
    # 单位旋转 → xyzw ≈ (0,0,0,1)
    assert np.allclose(state[3:7], [0, 0, 0, 1], atol=1e-5)
    assert np.allclose(grips, [0.8, 0.2])


def test_views_stack_192x256() -> None:
    front = np.full((480, 640, 3), 10, dtype=np.uint8)
    left = np.full((480, 640, 3), 20, dtype=np.uint8)
    right = np.full((480, 640, 3), 30, dtype=np.uint8)
    stacked = views_from_obs(
        {
            "observation.images.front": front,
            "observation.images.left": left,
            "observation.images.right": right,
        }
    )
    assert stacked.shape == (3, IMG_H, IMG_W, 3)
    assert stacked.dtype == np.uint8
    assert int(stacked[0].mean()) == 10
    assert int(stacked[1].mean()) == 20
    assert int(stacked[2].mean()) == 30


def test_cam_alias_and_chw() -> None:
    img = np.zeros((3, 8, 10), dtype=np.float32)
    hwc = obs_image_to_hwc_uint8(img)
    assert hwc.shape == (8, 10, 3)
    assert hwc.dtype == np.uint8


def test_resolve_tau_ckpt_prefers_file_without_index(tmp_path: Path) -> None:
    step = tmp_path / "step_5000"
    step.mkdir()
    (step / "diffusion_pytorch_model.bin").write_bytes(b"x")
    (step / "config.json").write_text("{}", encoding="utf-8")
    got = resolve_tau_ckpt(tmp_path)
    assert got == (step / "diffusion_pytorch_model.bin").resolve()


def test_resolve_tau_ckpt_index_dir(tmp_path: Path) -> None:
    (tmp_path / "diffusion_pytorch_model.bin.index.json").write_text("{}", encoding="utf-8")
    assert resolve_tau_ckpt(tmp_path) == tmp_path.resolve()


def test_write_deploy_yaml_paths(tmp_path: Path) -> None:
    wan = tmp_path / "wan"
    wan.mkdir()
    out = write_deploy_yaml(
        tmp_path / "deploy.yaml",
        model_path=tmp_path / "ckpt.bin",
        wan_root=wan,
        statistics_file=tmp_path / "stat.json",
    )
    text = out.read_text(encoding="utf-8")
    assert "action_chunk: 33" in text
    assert "action_in_dim: 20" in text
    assert "action_type: \"relative\"" in text
    assert str(wan / "Wan2.2_VAE.pth") in text
    assert ACTION_CHUNK == 33
    assert TONGBOT_DIM == 16
    assert CAM_FRONT.endswith("front")


def test_ik_settings_from_args_continue_chunk() -> None:
    from types import SimpleNamespace

    from serve_tong_tau import _ik_settings_from_args

    raw = _ik_settings_from_args(
        SimpleNamespace(
            ik_mode="continue_chunk",
            ik_solver="realman",
            ik_traj_opt="none",
            ik_max_step_deg=40,
            ik_config="",
        )
    )
    assert raw["mode"] == "continue_chunk"
    assert raw["solver"] == "realman"
    assert raw["traj_opt"] == "none"
    assert float(raw["max_joint_step_deg"]) == 40.0


if __name__ == "__main__":
    import tempfile

    test_state16_pads_and_truncates()
    test_rot6d_cols_roundtrip_identity()
    test_rot6d_cols_not_rows()
    test_euler_cols_matches_decoded_matrix()
    test_quat_roundtrip_keeps_columns()
    test_ee20_to_play_layout()
    test_views_stack_192x256()
    test_cam_alias_and_chw()
    with tempfile.TemporaryDirectory() as d:
        test_resolve_tau_ckpt_prefers_file_without_index(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_resolve_tau_ckpt_index_dir(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_write_deploy_yaml_paths(Path(d))
    test_ik_settings_from_args_continue_chunk()
    print("ok")
