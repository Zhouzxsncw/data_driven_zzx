"""把 29D 身体系轨迹编成 GR00T SONIC 的 64D ``motion_token``。

官方 token 来自 SONIC encoder（``token_state`` / model_encoder.onnx，G1 mode）。
本机 ``gear_sonic_deploy/policy`` 只有 yaml、没有 onnx 时，用 FSQ 1/16 网格
把 29 位置 + 29 速度 pack 进 64 维，好让 ``train.py`` / ``UNITREE_G1_SONIC``
能读、能反传。这 **不是** 官方 FSQ latent：上机前要用 encoder 重写 parquet。

若 ``SONIC_ENCODER_ONNX`` 指向 encoder，这里预留接口（输入维必须匹配 obs yaml）。
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from sonic_schema import BODY29_INDEX, LEFT_HAND_INDEX, RIGHT_HAND_INDEX

# Isaac-GR00T gear_sonic/utils/inference/initial_poses.py 站立 token（仅作参考/备用）
STANDING_TOKEN = np.array(
    [
        -0.0625, 0.0000, -0.0625, -0.1250, -0.1875, -0.0625, 0.1875,
        0.2500, 0.1875, -0.1250, 0.0625, -0.0625, -0.2500, -0.2500,
        -0.3125, -0.0625, 0.0000, -0.0625, -0.1250, -0.1875, 0.0000,
        -0.2500, 0.0000, -0.2500, -0.0625, 0.0625, 0.1250, -0.1250,
        0.2500, 0.1875, 0.2500, -0.1250, 0.1250, 0.1875, -0.0625,
        0.0000, -0.1875, -0.1875, 0.2500, 0.0000, 0.0000, -0.1250,
        0.0625, 0.0000, -0.0625, -0.0625, 0.1875, -0.0625, 0.0000,
        0.0625, 0.1250, 0.0625, 0.1250, 0.0625, 0.1250, 0.0000,
        0.1250, 0.1875, 0.0000, 0.0000, 0.0625, 0.0625, 0.1875,
        0.0625,
    ],
    dtype=np.float32,
)


def _fsq16(x: np.ndarray) -> np.ndarray:
    return (np.round(np.clip(x, -1.0, 1.0) * 16.0) / 16.0).astype(np.float32)


def unpack_body29_fsq16(token: np.ndarray) -> np.ndarray:
    """(T, 64) FSQ pack → (T, 29) 身体系关节。pack 是 ``body/2``，这里乘回去。"""
    token = np.asarray(token, dtype=np.float32)
    if token.shape[-1] < 29:
        raise ValueError(f"motion_token last dim {token.shape[-1]} < 29")
    return token[..., :29] * 2.0


def token_and_hands_to_q43(token: np.ndarray, left_hand: np.ndarray, right_hand: np.ndarray) -> np.ndarray:
    """64D token + 7+7 手 → SONIC 43 关节（绝对）。支持 (T, D) 或 (B, T, D)。"""
    body = unpack_body29_fsq16(token)
    q43 = np.zeros(body.shape[:-1] + (43,), dtype=np.float32)
    q43[..., BODY29_INDEX] = body
    q43[..., LEFT_HAND_INDEX] = np.asarray(left_hand, dtype=np.float32)
    q43[..., RIGHT_HAND_INDEX] = np.asarray(right_hand, dtype=np.float32)
    return q43


def pack_body29_fsq16(q43: np.ndarray, qd43: np.ndarray | None = None) -> np.ndarray:
    """q43/qd43: (T, 43) SONIC 序 → (T, 64) FSQ 1/16。"""
    q43 = np.asarray(q43, dtype=np.float32)
    t = q43.shape[0]
    body = q43[:, BODY29_INDEX]
    if qd43 is None:
        vel = np.zeros_like(body)
        vel[1:] = np.diff(body, axis=0) * 50.0
    else:
        vel = np.asarray(qd43, dtype=np.float32)[:, BODY29_INDEX]
    packed = np.zeros((t, 64), dtype=np.float32)
    packed[:, :29] = _fsq16(body / 2.0)
    packed[:, 29:58] = _fsq16(vel / 8.0)
    return packed


def encode_motion_token(q43: np.ndarray, qd43: np.ndarray | None = None) -> np.ndarray:
    """优先官方 encoder；否则 body29 FSQ pack。"""
    onnx = os.environ.get("SONIC_ENCODER_ONNX", "").strip()
    if onnx and Path(onnx).is_file():
        try:
            return _encode_onnx(q43, qd43, Path(onnx))
        except Exception as exc:  # noqa: BLE001
            print(f"[motion_token] encoder 失败，回退 FSQ pack: {exc}", flush=True)
    return pack_body29_fsq16(q43, qd43)


def _encode_onnx(q43: np.ndarray, qd43: np.ndarray | None, path: Path) -> np.ndarray:
    """G1-mode encoder 需要 1751 维 obs；缺 observation 装配时直接失败给上层回退。"""
    import onnxruntime as ort  # type: ignore

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    dim = int(inp.shape[-1]) if inp.shape[-1] not in (None, "None") else -1
    if dim != 1751:
        raise RuntimeError(f"encoder 输入维 {inp.shape}，不是 G1 1751，拒绝瞎填")
    raise RuntimeError("1751 维 obs 装配未接（需要 C++ GatherMotion*）；用 FSQ pack")
