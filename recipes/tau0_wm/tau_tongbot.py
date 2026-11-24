"""τ0-WM × TongBot 00036 EE 20D 推理契约（无 GPU 依赖）。

对齐 H100 后训练 ``posttrain_tongbot_00036_ee.yaml``：

  - 模型内部 20D 相对 EE，``action_space=eef6d``，chunk 33
  - rot6d = 旋转矩阵**前两列** ``(r00,r10,r20, r01,r11,r21)``
    （WALL-OSS 是前两行；混用等于旋转符号反了）
  - 三路独立 192×256（H×W），再叠成 ``(v,h,w,3)`` uint8
  - 相机键 ``observation.images.{front,left,right}``
  - 真机客户端仍发 16D 关节；GPU FK → play 出绝对 EE20，客户端 IK 后再回 16D
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

CAM_FRONT = "observation.images.front"
CAM_LEFT = "observation.images.left"
CAM_RIGHT = "observation.images.right"
CAM_KEYS = (CAM_FRONT, CAM_LEFT, CAM_RIGHT)
CAM_ALIASES = {
    CAM_FRONT: (CAM_FRONT, "observation.images.cam_high", "cam_high", "top_head"),
    CAM_LEFT: (CAM_LEFT, "observation.images.cam_left_wrist", "cam_left_wrist", "hand_left"),
    CAM_RIGHT: (CAM_RIGHT, "observation.images.cam_right_wrist", "cam_right_wrist", "hand_right"),
}

TONGBOT_DIM = 16
EE_DIM = 20
ACTION_CHUNK = 33
VIDEO_CHUNK = 9
# 训练 sample_size / 官方 img_size：[H, W]
IMG_H = 192
IMG_W = 256
IMG_SIZE_HW = (IMG_H, IMG_W)
DEFAULT_DENOISE = 10
DEFAULT_N_EXEC = 8
DEFAULT_PORT = 1244
HANDSHAKE_MODEL = "tau0wm"

# 20D：[L xyz | L rot6d(列) | L grip | R xyz | R rot6d | R grip]
LEFT_XYZ = slice(0, 3)
LEFT_ROT6 = slice(3, 9)
LEFT_GRIP = 9
RIGHT_XYZ = slice(10, 13)
RIGHT_ROT6 = slice(13, 19)
RIGHT_GRIP = 19

DEFAULT_CKPT = "/home/xk/VLA/h100_ckpts/tau0wm_tongbot_00036_ee"
DEFAULT_PRETRAINED = "/data/pretrained/tau-0-wm"
DEFAULT_WAN = "/data/pretrained/Wan2.2-TI2V-5B"
DEFAULT_STAT = str(
    Path(__file__).resolve().parent / "configs" / "stat_file" / "tongbot_00036_ee_rela.json"
)
H100_OUTPUT = (
    "/data1/tau0wm/outputs/tongbot_00036_ee/tau0wm_vam5b_ee_rela_192x256_bs1"
)
DEFAULT_PROMPT = (
    "With both hands, pick up the yellow banana and the yellow lemon and put "
    "them into the black box; then take the green vegetables and place them "
    "into the blue box."
)
def _wall_oss_scripts() -> Path:
    env = os.environ.get("TAU_CONVERTER_DIR")
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates.append(Path(__file__).resolve().parent.parent / "walloss")
    candidates.append(Path("/home/xk/VLA/wall_oss_ppu_setup/scripts"))
    for p in candidates:
        if (p / "ee_infer_pipeline.py").is_file() or (
            p / "convert_tongbot_joint_to_ee.py"
        ).is_file():
            return p
    return candidates[0]


WALL_OSS_SCRIPTS = _wall_oss_scripts()

_WEIGHT_INDEXES = (
    "diffusion_pytorch_model.safetensors.index.json",
    "diffusion_pytorch_model.bin.index.json",
)
_WEIGHT_FILES = (
    "diffusion_pytorch_model.bin",
    "diffusion_pytorch_model.safetensors",
)


def state16(state: Any) -> np.ndarray:
    arr = np.asarray(state, dtype=np.float32).reshape(-1)
    if arr.size < TONGBOT_DIM:
        out = np.zeros(TONGBOT_DIM, dtype=np.float32)
        out[: arr.size] = arr
        return out
    return np.ascontiguousarray(arr[:TONGBOT_DIM], dtype=np.float32)


def euler_zyx_to_rot6d_cols(euler_xyz: np.ndarray) -> np.ndarray:
    """ZYX 欧拉角 → 列约定 rot6d = (r00,r10,r20, r01,r11,r21)。

    与 τ0-WM ``matrix_to_rotation_6d`` 一致；不是 WALL-OSS 的前两行。
    """
    e = np.atleast_2d(np.asarray(euler_xyz, dtype=np.float64))
    roll, pitch, yaw = e[:, 0], e[:, 1], e[:, 2]
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    r00, r01 = cy * cp, cy * sp * sr - sy * cr
    r10, r11 = sy * cp, sy * sp * sr + cy * cr
    r20, r21 = -sp, cp * sr
    out = np.stack([r00, r10, r20, r01, r11, r21], axis=1)
    return out[0] if np.asarray(euler_xyz).ndim == 1 else out


def rot6d_cols_to_matrix(r6: np.ndarray) -> np.ndarray:
    """列约定 rot6d → 3×3。前 3 = 第 0 列，后 3 = 第 1 列，第 2 列 = 叉积。"""
    r6 = np.asarray(r6, dtype=np.float64)
    single = r6.ndim == 1
    if single:
        r6 = r6[np.newaxis, :]
    c0 = r6[:, 0:3]
    c1 = r6[:, 3:6]
    c0 = c0 / (np.linalg.norm(c0, axis=1, keepdims=True) + 1e-12)
    c1 = c1 - np.sum(c1 * c0, axis=1, keepdims=True) * c0
    c1 = c1 / (np.linalg.norm(c1, axis=1, keepdims=True) + 1e-12)
    c2 = np.cross(c0, c1)
    R = np.stack([c0, c1, c2], axis=-1)
    return R[0] if single else R


def matrix_to_rot6d_cols(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    single = R.ndim == 2
    if single:
        R = R[np.newaxis, ...]
    out = np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)
    return out[0] if single else out


def matrix_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """3×3 → xyzw，与 τ0 ``TauPolicy.play`` / ``quaternion_to_rotation_6d`` 对齐。"""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = float(np.trace(R))
    if t > 0.0:
        s = 0.5 / np.sqrt(t + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        if i == 0:
            s = 2.0 * np.sqrt(max(1e-12, 1.0 + R[0, 0] - R[1, 1] - R[2, 2]))
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = 2.0 * np.sqrt(max(1e-12, 1.0 + R[1, 1] - R[0, 0] - R[2, 2]))
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(max(1e-12, 1.0 + R[2, 2] - R[0, 0] - R[1, 1]))
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    q = q / n
    if q[3] < 0:
        q = -q
    return q


def rot6d_cols_to_quat_xyzw(r6: np.ndarray) -> np.ndarray:
    return matrix_to_quat_xyzw(rot6d_cols_to_matrix(r6))


def ee20_to_play_state_and_grippers(ee20: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """绝对 EE20（列约定 rot6d）→ TauPolicy.play 的 14D xyz+quat 与 2D 夹爪。"""
    ee = np.asarray(ee20, dtype=np.float64).reshape(-1)
    if ee.size < EE_DIM:
        raise ValueError(f"EE 须 20D，得到 {ee.size}")
    ql = rot6d_cols_to_quat_xyzw(ee[LEFT_ROT6])
    qr = rot6d_cols_to_quat_xyzw(ee[RIGHT_ROT6])
    state = np.concatenate([ee[LEFT_XYZ], ql, ee[RIGHT_XYZ], qr]).astype(np.float32)
    grips = np.array([ee[LEFT_GRIP], ee[RIGHT_GRIP]], dtype=np.float32)
    return state, grips


def obs_image_to_hwc_uint8(img: Any) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"图像须为 HWC 或 CHW，得到 shape={arr.shape}")
    if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        amax = float(np.nanmax(arr)) if arr.size else 0.0
        if amax <= 1.5:
            arr = np.clip(arr, 0.0, 1.0) * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return np.ascontiguousarray(arr[..., :3])


def lookup_cam(obs: Mapping[str, Any], logical_key: str) -> Any:
    for alias in CAM_ALIASES[logical_key]:
        if alias in obs:
            return obs[alias]
    raise KeyError(f"obs 缺相机 {logical_key}；keys={list(obs)}")


def resize_view(img: np.ndarray, size_hw: tuple[int, int] = IMG_SIZE_HW) -> np.ndarray:
    h, w = int(size_hw[0]), int(size_hw[1])
    arr = obs_image_to_hwc_uint8(img)
    if arr.shape[0] == h and arr.shape[1] == w:
        return arr
    try:
        import cv2

        return cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)
    except ImportError:
        from PIL import Image

        return np.asarray(Image.fromarray(arr).resize((w, h), Image.BILINEAR))


def views_from_obs(obs: Mapping[str, Any]) -> np.ndarray:
    """三路 front/left/right → ``(3, 192, 256, 3)`` uint8。"""
    views = [resize_view(lookup_cam(obs, k)) for k in CAM_KEYS]
    return np.stack(views, axis=0)


def _step_num(path: Path) -> int | None:
    m = re.fullmatch(r"step_(\d+)", path.name)
    return int(m.group(1)) if m else None


def _dir_has_index(path: Path) -> bool:
    return any((path / name).is_file() for name in _WEIGHT_INDEXES)


def _dir_weight_file(path: Path) -> Path | None:
    for name in _WEIGHT_FILES:
        cand = path / name
        if cand.is_file():
            return cand
    shards = sorted(path.glob("diffusion_pytorch_model-*.bin")) + sorted(
        path.glob("diffusion_pytorch_model-*.safetensors")
    )
    return shards[0] if shards else None


def resolve_tau_ckpt(ckpt: str | Path) -> Path:
    """指向 ``load_checkpoints`` 能吃的路径：分片目录或单个 .bin/.safetensors。

    后训练 ``save_pretrained(step_N)`` 通常是目录里一份 ``diffusion_pytorch_model.bin``
    （没有 index.json）——这种必须把**文件**传进去，不能传目录。
    发布预训练 ``tau-0-wm`` 是带 ``*.bin.index.json`` 的分片目录，传目录。
    """
    p = Path(ckpt).expanduser().resolve()
    if p.is_file():
        return p
    if not p.is_dir():
        raise FileNotFoundError(f"找不到 τ0-WM 权重: {p}")
    if _dir_has_index(p):
        return p
    weight = _dir_weight_file(p)
    if weight is not None:
        return weight
    steps: list[tuple[int, Path]] = []
    for child in p.iterdir():
        n = _step_num(child) if child.is_dir() else None
        if n is not None:
            steps.append((n, child))
    if steps:
        return resolve_tau_ckpt(sorted(steps)[-1][1])
    raise FileNotFoundError(
        f"找不到 τ0-WM 权重（需要 diffusion_pytorch_model.bin / index / step_N）: {p}"
    )


def tau_weights_ready(ckpt: str | Path) -> bool:
    try:
        resolve_tau_ckpt(ckpt)
        return True
    except FileNotFoundError:
        return False


def apply_column_rot6d_patches(scripts_dir: str | Path | None = None) -> None:
    """把 WALL-OSS FK/IK 的行约定 rot6d 换成 τ0-WM 列约定。

    ``observe_ee20`` 走 ``euler_zyx_to_rot6d``；``ik_physical`` 走 ``rot6d_to_matrix``。
    两边必须同一约定，否则 IK 等于在解转置姿态。
    """
    scripts = Path(scripts_dir or WALL_OSS_SCRIPTS).expanduser().resolve()
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import convert_tongbot_joint_to_ee as conv  # noqa: WPS433
    import ee_infer_pipeline as pipe  # noqa: WPS433

    conv.euler_zyx_to_rot6d = euler_zyx_to_rot6d_cols
    pipe.rot6d_to_matrix = rot6d_cols_to_matrix
    pipe.matrix_to_rot6d = matrix_to_rot6d_cols


def write_deploy_yaml(
    path: str | Path,
    *,
    model_path: str | Path,
    wan_root: str | Path = DEFAULT_WAN,
    statistics_file: str | Path = DEFAULT_STAT,
) -> Path:
    """写一份 TongBot 部署 yaml，给 ``TauPolicy(config_file=...)`` 用。"""
    wan = Path(wan_root).expanduser().resolve()
    model = Path(model_path).expanduser().resolve()
    stat = Path(statistics_file).expanduser().resolve()
    text = f"""# τ0-WM TongBot 00036 EE 20D 真机部署（由 recipes/tau0_wm/tau_tongbot.py 生成）
# 对齐官方 configs/deployment/tau_pretrain_rela_eef6d.yaml，路径换成本地 Wan / SFT ckpt / 00036 统计。
action_dim: 20
gripper_dim: 1
chunk: {VIDEO_CHUNK}
action_chunk: {ACTION_CHUNK}
img_size: [{IMG_H}, {IMG_W}]
action_type: "relative"
action_space: "eef6d"
statistics_file: {stat}
norm_type: "meanstd"

vae_class: Wan2_2_VAE
vae_class_path: models/wan_2_2_models/vae/vae2_2.py
vae_path: {wan / "Wan2.2_VAE.pth"}

text_encoder:
  checkpoint_path: {wan / "models_t5_umt5-xxl-enc-bf16.pth"}
  text_len: 512
  tokenizer_path: {wan / "google" / "umt5-xxl"}
textenc_class: T5EncoderModel
textenc_class_path: models/wan_2_2_models/text_encoder/t5.py

diffusion_model:
  config:
    action_dim: 1024
    action_ffn_dim: 2048
    action_in_dim: 20
    action_max_seq_len: 60
    action_num_heads: 16
    cross_attn_norm: true
    dim: 3072
    eps: 1.0e-06
    ffn_dim: 14336
    freq_dim: 256
    in_dim: 48
    model_type: ti2v
    num_heads: 24
    num_layers: 30
    out_dim: 48
    patch_size:
    - 1
    - 2
    - 2
    qk_norm: true
    text_dim: 4096
    text_len: 512
    use_ae: true
    window_size:
    - -1
    - 1
  model_path: {model}

diffusion_model_class: WanModel
diffusion_model_class_path: models/wan_2_2_models/transformers/model.py

pipeline_class: WanTI2V
pipeline_class_path: models/wan_2_2_models/pipeline/textimage2video.py
"""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return out.resolve()
