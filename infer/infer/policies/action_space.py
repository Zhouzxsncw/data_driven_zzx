"""Detect TongBot ACTION SPACE (joint vs EE) and map EE → IK joints.

Detection order:
  1. YAML ``policy.action_space`` (joint | ee)
  2. policy.type=walloss / tau0wm → EE; GPU infer 回 abs EE20，客户端立刻 ws solve_ik → 16D
  3. **train_record.json** next to the checkpoint (written at train start)
  4. dataset ``meta/action_space.json``
  5. lingbot-vla-v2 robot_config ``actions`` keys
     (``action.end.position`` → EE, ``action.arm.position`` → joint)
  6. ``lingbot_robo_name`` ending with ``_ee``
  7. numeric heuristic on a predicted 16D chunk (unit quat + workspace xyz)

Control:
  joint → send 16D logical joints (``send_action_15d``)
  ee    → LingBot 16D xyz+quat_xyzw → WALL-OSS EE20 → RealMan IK → 16D joints
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_ROBOT_CONFIG_DIR = _REPO_ROOT / "lingbot-vla-v2" / "configs" / "robot_configs"


def _ee_pipeline_scripts() -> Path:
    for p in (
        _REPO_ROOT / "recipes" / "walloss",
        _REPO_ROOT / "wall_oss_ppu_setup" / "scripts",
        Path("/home/xk/VLA/wall_oss_ppu_setup/scripts"),
    ):
        if (p / "ee_infer_pipeline.py").is_file():
            return p
    return _REPO_ROOT / "recipes" / "walloss"


_EE_PIPELINE_SCRIPTS = _ee_pipeline_scripts()

# Infer cam_* → dataset origin keys used by 00073 / 00086 / 00065
IMAGE_MAP_LEFT_FRONT_RIGHT = {
    "observation.cam_left": "observation.images.left",
    "observation.cam_mid": "observation.images.front",
    "observation.cam_right": "observation.images.right",
    "observation.images.left": "observation.images.left",
    "observation.images.front": "observation.images.front",
    "observation.images.right": "observation.images.right",
}

# 00036 tongbot_16d
IMAGE_MAP_CAM_LEFT_MID_RIGHT = {
    "observation.cam_left": "observation.images.cam_left",
    "observation.cam_mid": "observation.images.cam_mid",
    "observation.cam_right": "observation.images.cam_right",
    "observation.images.cam_left": "observation.images.cam_left",
    "observation.images.cam_mid": "observation.images.cam_mid",
    "observation.images.cam_right": "observation.images.cam_right",
}


@dataclass(frozen=True)
class ActionSpaceDecision:
    action_space: str  # "joint" | "ee"
    control: str  # "joint" | "ik"
    source: str
    robot_config_path: Optional[str] = None
    layout: str = ""

    def log_line(self) -> str:
        return (
            f"ACTION SPACE={self.action_space} → 控制="
            f"{'关节角度' if self.control == 'joint' else '末端 IK'} "
            f"({self.source}"
            f"{'; ' + self.layout if self.layout else ''})"
        )


def _iter_robot_config_items(section: Any) -> List[Tuple[str, Dict[str, Any]]]:
    out: List[Tuple[str, Dict[str, Any]]] = []
    for item in section or []:
        if isinstance(item, dict) and item:
            key = next(iter(item))
            val = item[key]
            out.append((str(key), val if isinstance(val, dict) else {}))
        elif isinstance(item, str):
            out.append((item, {}))
    return out


def action_feature_keys(robot_config: Dict[str, Any]) -> List[str]:
    return [k for k, _ in _iter_robot_config_items(robot_config.get("actions"))]


def detect_action_space_from_robot_config(robot_config: Dict[str, Any]) -> str:
    keys = action_feature_keys(robot_config)
    has_end = any("end.position" in k for k in keys)
    has_arm = any("arm.position" in k for k in keys)
    if has_end:
        return "ee"
    if has_arm:
        return "joint"
    return "joint"


def image_map_from_robot_config(robot_config: Dict[str, Any]) -> Dict[str, str]:
    """Map infer camera_keys onto robot_config image origin_keys."""
    origins: Dict[str, str] = {}
    for key, spec in _iter_robot_config_items(robot_config.get("images")):
        origin = spec.get("origin_keys")
        if isinstance(origin, str) and origin:
            origins[key] = origin
        elif isinstance(origin, list) and origin:
            first = origin[0]
            if isinstance(first, str):
                origins[key] = first

    def _pick(*needles: str) -> Optional[str]:
        for feat, origin in origins.items():
            low = feat.lower()
            if any(n in low for n in needles):
                return origin
        return None

    top = _pick("camera_top", "cam_mid", "front")
    left = _pick("wrist_left", "cam_left")
    right = _pick("wrist_right", "cam_right")
    mapping: Dict[str, str] = {}
    pairs = (
        ("observation.cam_left", left),
        ("observation.cam_mid", top),
        ("observation.cam_right", right),
        ("observation.images.left", left),
        ("observation.images.front", top),
        ("observation.images.right", right),
        ("observation.images.cam_left", left),
        ("observation.images.cam_mid", top),
        ("observation.images.cam_right", right),
    )
    for src, dst in pairs:
        if dst:
            mapping[src] = dst
    for origin in origins.values():
        mapping.setdefault(origin, origin)
    return mapping


def find_robot_config(
    robo_name: str,
    explicit: Optional[str | Path] = None,
) -> Optional[Path]:
    if explicit:
        path = Path(explicit).expanduser()
        if path.is_file():
            return path
    name = str(robo_name or "").strip()
    if not name:
        return None
    candidates: List[Path] = []
    env_root = os.environ.get("LINGBOT_VLA_ROOT")
    if env_root:
        candidates.append(Path(env_root) / "configs" / "robot_configs" / f"{name}.yaml")
    candidates.append(_DEFAULT_ROBOT_CONFIG_DIR / f"{name}.yaml")
    for path in candidates:
        if path.is_file():
            return path
    return None


def load_robot_config(path: Path) -> Dict[str, Any]:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"robot_config 不是 mapping: {path}")
    return payload


def _normalize_space_token(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if not text or text in ("auto", "detect", "none"):
        return None
    if text in ("ee", "end", "eef", "cartesian", "pose"):
        return "ee"
    if text in ("joint", "joints", "qpos", "arm"):
        return "joint"
    raise ValueError(f"未知 action_space={raw!r}；请用 joint / ee / auto")


def looks_like_lingbot_ee16(action: np.ndarray) -> bool:
    """Unit-quat + workspace xyz, as opposed to 16D joint rad."""
    arr = np.asarray(action, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[-1] < 16:
        return False
    xyz_max = float(max(np.max(np.abs(arr[:, 0:3])), np.max(np.abs(arr[:, 7:10]))))
    qn_l = np.linalg.norm(arr[:, 3:7], axis=-1)
    qn_r = np.linalg.norm(arr[:, 10:14], axis=-1)
    return (
        float(np.mean(np.abs(qn_l - 1.0))) < 0.15
        and float(np.mean(np.abs(qn_r - 1.0))) < 0.15
        and 0.05 < xyz_max < 1.5
    )


def extract_dataset_space(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Pull action_space / control / layout from train_record or dataset sidecar."""
    if not isinstance(record, dict):
        return None
    nested = record.get("dataset") if isinstance(record.get("dataset"), dict) else {}
    space = (
        nested.get("action_space")
        or record.get("action_space")
        or (record.get("hparams") or {}).get("action_space")
    )
    if space in ("ee", "joint"):
        pass
    else:
        text = str(record.get("action") or record.get("action_space") or "")
        low = text.lower()
        if "end.position" in low or " quat" in low or low.endswith("ee") or "lingbot ee" in low:
            space = "ee"
        elif "joint" in low or "arm.position" in low:
            space = "joint"
        else:
            space = None
    if space not in ("joint", "ee"):
        return None
    control = (
        nested.get("control")
        or record.get("control")
        or (record.get("hparams") or {}).get("control")
    )
    if control not in ("joint", "ik"):
        control = "ik" if space == "ee" else "joint"
    return {
        "action_space": space,
        "control": control,
        "obs_space": nested.get("obs_space") or record.get("obs_space") or "joint",
        "action_layout": nested.get("action_layout")
        or record.get("action_layout")
        or str(record.get("action") or ""),
        "robot_config": nested.get("robot_config") or record.get("robot_config"),
        "robot_config_path": nested.get("robot_config_path")
        or record.get("robot_config_path"),
    }


def find_train_record(
    ckpt_dir: Optional[str | Path] = None,
    extra: Optional[Sequence[str | Path]] = None,
) -> Optional[Path]:
    candidates: List[Path] = []
    for raw in extra or []:
        if raw:
            candidates.append(Path(raw).expanduser())
    if ckpt_dir:
        cur = Path(ckpt_dir).expanduser()
        for _ in range(5):
            candidates.append(cur / "train_record.json")
            if cur.parent == cur:
                break
            cur = cur.parent
    for path in candidates:
        if path.is_file() and path.suffix.lower() == ".json":
            return path
    return None


def find_dataset_action_space(dataset_root: Optional[str | Path] = None) -> Optional[Path]:
    if not dataset_root:
        return None
    root = Path(dataset_root).expanduser()
    for rel in (
        "meta/action_space.json",
        "meta/lingbot_ee_action.json",
        "pipeline_record.json",
    ):
        path = root / rel
        if path.is_file():
            return path
    return None


def load_json_record(path: Path) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def resolve_action_space(
    *,
    override: Any = None,
    robo_name: str = "",
    robot_config_path: Optional[str | Path] = None,
    policy_type: Any = None,
    train_record: Optional[Dict[str, Any]] = None,
    train_record_path: Optional[str | Path] = None,
    dataset_root: Optional[str | Path] = None,
    ckpt_dir: Optional[str | Path] = None,
) -> ActionSpaceDecision:
    ptype = str(getattr(policy_type, "value", policy_type) or "").strip().lower()
    if ptype in ("walloss", "wall-oss", "wall_oss", "wall-x", "wallx"):
        return ActionSpaceDecision(
            action_space="ee",
            control="ik",
            source="policy.type=walloss",
            layout="WALL-OSS 相对 EE20；infer 回绝对 EE20；ws solve_ik → 16D",
        )
    if ptype in ("tau0wm", "tau0", "tau-0-wm", "tau_0_wm", "tau-0", "tauwm"):
        return ActionSpaceDecision(
            action_space="ee",
            control="ik",
            source="policy.type=tau0wm",
            layout="τ0-WM 相对 EE20（列约定 rot6d）；infer 回绝对 EE20；ws solve_ik → 16D",
        )

    forced = _normalize_space_token(override)
    cfg_path = find_robot_config(robo_name, robot_config_path)
    layout = ""
    detected: Optional[str] = None
    if cfg_path is not None:
        robot_cfg = load_robot_config(cfg_path)
        detected = detect_action_space_from_robot_config(robot_cfg)
        keys = action_feature_keys(robot_cfg)
        layout = "actions=" + ",".join(keys)

    if forced is not None:
        control = "ik" if forced == "ee" else "joint"
        return ActionSpaceDecision(
            action_space=forced,
            control=control,
            source="yaml:action_space",
            robot_config_path=str(cfg_path) if cfg_path else None,
            layout=layout,
        )

    record = train_record
    rec_path = Path(train_record_path).expanduser() if train_record_path else None
    if record is None:
        rec_path = rec_path or find_train_record(ckpt_dir)
        if rec_path is not None:
            record = load_json_record(rec_path)
    extracted = extract_dataset_space(record) if record else None
    if extracted:
        return ActionSpaceDecision(
            action_space=extracted["action_space"],
            control=extracted["control"],
            source=f"train_record:{(rec_path or Path('train_record.json')).name}",
            robot_config_path=extracted.get("robot_config_path")
            or (str(cfg_path) if cfg_path else None),
            layout=extracted.get("action_layout") or layout,
        )

    ds_path = find_dataset_action_space(dataset_root)
    if ds_path is not None:
        extracted = extract_dataset_space(load_json_record(ds_path) or {})
        if extracted:
            return ActionSpaceDecision(
                action_space=extracted["action_space"],
                control=extracted["control"],
                source=f"dataset:{ds_path.name}",
                robot_config_path=extracted.get("robot_config_path")
                or (str(cfg_path) if cfg_path else None),
                layout=extracted.get("action_layout") or layout,
            )

    if detected is not None:
        return ActionSpaceDecision(
            action_space=detected,
            control="ik" if detected == "ee" else "joint",
            source=f"robot_config:{cfg_path.name}",
            robot_config_path=str(cfg_path),
            layout=layout,
        )

    name = str(robo_name or "").strip().lower()
    if name.endswith("_ee") or "_ee_" in name:
        return ActionSpaceDecision(
            action_space="ee",
            control="ik",
            source="robo_name",
            layout="LingBot EE16 xyz+quat_xyzw",
        )
    return ActionSpaceDecision(
        action_space="joint",
        control="joint",
        source="default",
        layout="16D 逻辑关节 [L7|R7|Lgrip|Rgrip]",
    )


def quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(-1)[:4]
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = q / n
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_rot6d(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    single = R.ndim == 2
    if single:
        R = R[None, ...]
    out = np.concatenate([R[:, 0, :], R[:, 1, :]], axis=1)
    return out[0] if single else out


def lingbot_ee16_to_ee20(ee16: np.ndarray) -> np.ndarray:
    """LingBot [L xyz+quat_xyzw7 | R xyz+quat_xyzw7 | grips2] → WALL-OSS EE20."""
    arr = np.asarray(ee16, dtype=np.float64)
    single = arr.ndim == 1
    if single:
        arr = arr[None, :]
    if arr.shape[-1] < 16:
        raise ValueError(f"期望 LingBot EE 16D，得到 {arr.shape}")
    n = int(arr.shape[0])
    out = np.zeros((n, 20), dtype=np.float64)
    for i in range(n):
        out[i, 0:3] = arr[i, 0:3]
        out[i, 3:9] = matrix_to_rot6d(quat_xyzw_to_matrix(arr[i, 3:7]))
        out[i, 9] = arr[i, 14]
        out[i, 10:13] = arr[i, 7:10]
        out[i, 13:19] = matrix_to_rot6d(quat_xyzw_to_matrix(arr[i, 10:14]))
        out[i, 19] = arr[i, 15]
    return out[0] if single else out


def load_ee_ik_pipeline(
    sdk_root: Optional[str | Path] = None,
    *,
    n_exec: int = 8,
    ik: Optional[Any] = None,
):
    """Lazy-import RealMan IK used by WALL-OSS / τ0-WM / LingBot-VLA. Needs RM_API."""
    root = Path(sdk_root or os.path.expanduser("~/RM_API")).expanduser()
    scripts = str(_EE_PIPELINE_SCRIPTS)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from ee_infer_pipeline import EEInferPipeline  # type: ignore
    from ik_config import parse_ik_config  # type: ignore

    settings = parse_ik_config(ik) if ik is not None else None
    return EEInferPipeline(root, n_exec=int(n_exec), ik=settings)


def ik_abs_ee20_chunk(
    pipeline: Any,
    ee20_chunk: np.ndarray,
    seed_joints16: np.ndarray,
    n_exec: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Absolute EE20 → 16D logical joints via WALL-OSS ``ik_abs_chunk``.

    τ0-WM and WALL-OSS GPU serve now return abs EE20; this is the client IK.
    Empty plans hold the seed as a 1-row chunk (same as the old server abort).
    """
    ee = np.asarray(ee20_chunk, dtype=np.float64)
    if ee.ndim == 1:
        ee = ee[None, :]
    if ee.shape[-1] != 20:
        raise ValueError(f"need EE20, got {ee.shape}")
    n_model = int(ee.shape[0])
    n_tgt = n_model if n_exec is None else max(1, min(n_model, int(n_exec)))
    seed = np.asarray(seed_joints16, dtype=np.float64).reshape(-1)[:16].copy()
    t0 = time.perf_counter()
    plans = pipeline.ik_abs_chunk(seed, ee[:n_tgt], n_exec=n_tgt)
    ik_ms = (time.perf_counter() - t0) * 1e3
    if not plans:
        hold = seed.astype(np.float32).reshape(1, 16)
        return hold, {
            "n_planned": 0,
            "n_requested": n_tgt,
            "aborted_ik": True,
            "cartesian_limited": False,
            "held_ik": True,
            "rejected_step_deg": 0.0,
            "cartesian_t": 0.0,
            "max_step_deg": 0.0,
            "ik_ms": ik_ms,
        }
    scripts = str(_EE_PIPELINE_SCRIPTS)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from ee_infer_pipeline import plan_to_logical16  # type: ignore

    out = np.stack(
        [np.asarray(plan_to_logical16(p), dtype=np.float32) for p in plans],
        axis=0,
    )
    return out, {
        "n_planned": len(plans),
        "n_requested": n_tgt,
        "aborted_ik": len(plans) < n_tgt,
        "cartesian_limited": any(bool(p.cartesian_limited) for p in plans),
        "held_ik": any(bool(p.held) for p in plans),
        "rejected_step_deg": max(
            (float(p.rejected_step_deg) for p in plans), default=0.0
        ),
        "cartesian_t": min((float(p.cartesian_t) for p in plans), default=1.0),
        "max_step_deg": max(
            (max(p.left_step_max_deg, p.right_step_max_deg) for p in plans),
            default=0.0,
        ),
        "ik_ms": ik_ms,
    }


def ik_lingbot_ee16_chunk(
    pipeline: Any,
    ee16_chunk: np.ndarray,
    seed_joints16: np.ndarray,
    n_exec: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Absolute LingBot EE16 → 16D logical joints, WALL-OSS ``ik_abs_chunk``.

    Only the first ``n_exec`` waypoints are solved (same as WALL-OSS open-loop
    horizon). The rest of a long model chunk is ignored. Failed steps hold last q.
    夹爪用模型 EE16 的后两维，不走 IK。返回的 info 与 ``ik_abs_ee20_chunk`` 同 schema，
    给 eval_log / [IK] / infer_probe 用。
    """
    ee16 = np.asarray(ee16_chunk, dtype=np.float64)
    if ee16.ndim == 1:
        ee16 = ee16[None, :]
    n_model = int(ee16.shape[0])
    n_tgt = n_model if n_exec is None else max(1, min(n_model, int(n_exec)))
    ee16 = ee16[:n_tgt]
    ee20 = lingbot_ee16_to_ee20(ee16)
    seed = np.asarray(seed_joints16, dtype=np.float64).reshape(-1)[:16].copy()
    joints, info = ik_abs_ee20_chunk(pipeline, ee20, seed, n_exec=n_tgt)
    settings = getattr(pipeline, "ik_settings", None)
    mode = getattr(settings, "mode", None) if settings is not None else None
    if mode:
        info.setdefault("ik_mode", str(mode))
    out = np.zeros((n_tgt, 16), dtype=np.float32)
    last = seed.copy()
    n_plan = int(joints.shape[0])
    for i in range(n_tgt):
        if i < n_plan:
            last = np.asarray(joints[i], dtype=np.float64).reshape(-1)[:16].copy()
        last[14:16] = ee16[i, 14:16]
        out[i] = last.astype(np.float32)
    info["n_ok"] = int(info.get("n_planned") or 0)
    return out, info


def format_control_zh(control: str) -> str:
    return "关节角度" if control == "joint" else "末端 IK"
