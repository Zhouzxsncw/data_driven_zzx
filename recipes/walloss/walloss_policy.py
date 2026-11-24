#!/usr/bin/env python3
"""TongBot WALL-OSS-0.5 inference.

  images + 16D joints
    → FK absolute EE20
    → flow matching relative EE
    → relative EE → absolute EE20
    → (open-loop still does IK here; websocket serve returns abs EE, client IK)

Used by the websocket server and the 00086 open-loop eval. Official wall-x
serving (desktop/ex001) is not used; this builds dual-arm 6D-EE observations
matching the jiangsuan-16 SFT YAML.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

LOGGER = logging.getLogger("walloss_policy")

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CKPT = Path(
    "/data/wall-oss/outputs/walloss05_tongbot_ee_00086/latest_jiangsuan16"
)
DEFAULT_WALLX = Path("/data/wall-oss/src/wall-x")
DEFAULT_PROCESSOR = Path("/data/wall-oss/pretrained/Qwen2.5-VL-3B-Instruct")
DEFAULT_SDK = Path("/home/xk/RM_API")
DEFAULT_PROMPT = (
    "Using both hands, pull open the drawer with the blue handle using your left hand, "
    "take out the white medicine box with your right hand, place it into the green drawer, "
    "and finally close the drawer."
)
EE_DIM = 20
PAD_DIM = 6
CAM_TO_VIEW = {
    "observation.images.front": "face_view",
    "observation.images.left": "left_wrist_view",
    "observation.images.right": "right_wrist_view",
    "face_view": "face_view",
    "left_wrist_view": "left_wrist_view",
    "right_wrist_view": "right_wrist_view",
}
NORM_KEY = "tongbot_00086"


def ensure_wallx_on_path(wallx: Path | str = DEFAULT_WALLX) -> None:
    wallx = Path(wallx)
    if str(wallx) not in sys.path:
        sys.path.insert(0, str(wallx))
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    sdk = os.environ.get("RM_SDK_ROOT", str(DEFAULT_SDK))
    if sdk and sdk not in sys.path:
        sys.path.insert(0, sdk)


def prepare_checkpoint(
    ckpt_dir: Path | str,
    *,
    processor_src: Path | str = DEFAULT_PROCESSOR,
) -> Path:
    """Make a jiangsuan-16 ckpt loadable on this machine.

    Serving reads ``observation.state`` / ``action`` from ``norm_stats.json``,
    but SFT wrote ``observation.state.ee`` / ``action.ee``. Also copy the Qwen
    chat template so AutoProcessor can assemble from the ckpt tokenizer.
    """
    ckpt_dir = Path(ckpt_dir).expanduser().resolve()
    if not (ckpt_dir / "model.safetensors").is_file():
        raise FileNotFoundError(f"missing model.safetensors: {ckpt_dir}")

    stats_path = ckpt_dir / "norm_stats.json"
    if stats_path.is_file():
        raw = json.loads(stats_path.read_text(encoding="utf-8"))
        ns = raw.get("norm_stats", raw)
        changed = False
        if "observation.state.ee" in ns and "observation.state" not in ns:
            ns["observation.state"] = ns["observation.state.ee"]
            changed = True
        if "action.ee" in ns and "action" not in ns:
            ns["action"] = ns["action.ee"]
            changed = True
        if changed:
            stats_path.write_text(
                json.dumps({"norm_stats": ns}, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            LOGGER.info("aliased norm_stats.json keys to observation.state / action")

    src = Path(processor_src)
    for name in ("chat_template.json", "merges.txt"):
        src_f = src / name
        dst_f = ckpt_dir / name
        if src_f.is_file() and not dst_f.exists():
            shutil.copy2(src_f, dst_f)
            LOGGER.info("copied %s into ckpt", name)
    return ckpt_dir


def to_hwc_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim != 3 or arr.shape[-1] not in (1, 3):
        raise ValueError(f"expect HWC/CHW image, got {arr.shape}")
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if np.issubdtype(arr.dtype, np.floating):
        max_v = float(np.nanmax(arr)) if arr.size else 1.0
        if max_v <= 1.0 + 1e-3:
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)
    return np.ascontiguousarray(arr)


def _flatten_train_config(train_config: dict) -> None:
    task = train_config.get("task") or {}
    for key in ("dof_config", "ar_dof_config", "agent_pos_config"):
        if not train_config.get(key) and isinstance(task, dict) and task.get(key):
            train_config[key] = task[key]


def _ee20_to_state_fields(ee20: np.ndarray) -> dict[str, np.ndarray]:
    ee = np.asarray(ee20, dtype=np.float64).reshape(-1)
    if ee.size < EE_DIM:
        raise ValueError(f"EE20 need 20 dims, got {ee.size}")
    return {
        "left_ee_cartesian_pos": ee[0:3],
        "left_ee_rotation_6D": ee[3:9],
        "left_gripper": ee[9:10],
        "right_ee_cartesian_pos": ee[10:13],
        "right_ee_rotation_6D": ee[13:19],
        "right_gripper": ee[19:20],
    }


def _ckpt_embed_rows(ckpt_dir: Path) -> int:
    from safetensors import safe_open

    path = str(Path(ckpt_dir) / "model.safetensors")
    with safe_open(path, framework="pt") as f:
        if "model.embed_tokens.weight" not in f.keys():
            return 0
        return int(f.get_slice("model.embed_tokens.weight").get_shape()[0])


def _expand_tokenizer_to_ckpt(processor, ckpt_dir: Path) -> None:
    embed_rows = _ckpt_embed_rows(ckpt_dir)
    tok = processor.tokenizer
    n = len(tok)
    if embed_rows <= n:
        return
    extra = [f"<|extra_token_{i}|>" for i in range(embed_rows - n)]
    tok.add_tokens(extra)
    LOGGER.info("padded tokenizer %s -> %s to match ckpt embed_tokens", n, embed_rows)


class _TongWallxWrapper:
    """WallxModelWrapper that keeps SFT embed rows (155765), not the raw Qwen vocab."""

    def __init__(self, infer_cfg, ckpt_dir: Path):
        from wall_x._vendor.harrix.serving._wallx_infer.model_wrapper import (  # noqa: E402
            WallxModelWrapper,
        )

        class _Wrapper(WallxModelWrapper):
            def _load_processor(self):
                super()._load_processor()
                _expand_tokenizer_to_ckpt(self.processor, ckpt_dir)

        self._impl = _Wrapper(infer_cfg)

    def __getattr__(self, name: str):
        return getattr(self._impl, name)


class WallOssTongPolicy:
    """Load jiangsuan-16 SFT weights and run TongBot EE inference."""

    def __init__(
        self,
        ckpt_dir: Path | str = DEFAULT_CKPT,
        *,
        wallx: Path | str = DEFAULT_WALLX,
        processor_src: Path | str = DEFAULT_PROCESSOR,
        sdk_root: Path | str = DEFAULT_SDK,
        instruction: str = DEFAULT_PROMPT,
        n_exec: int = 8,
        device: str = "cuda:0",
        num_inference_timesteps: int = 10,
        action_horizon: int = 32,
        ik: Optional[dict | object] = None,
    ):
        ensure_wallx_on_path(wallx)
        self.ckpt_dir = prepare_checkpoint(ckpt_dir, processor_src=processor_src)
        self.instruction = instruction
        self.n_exec = int(n_exec)
        self.device = device
        self.sdk_root = Path(sdk_root).expanduser()

        from wall_x._vendor.harrix.serving._wallx_infer.base_dataclass import (  # noqa: E402
            RobotStateActionData,
        )
        from wall_x._vendor.harrix.serving._wallx_infer.infer_config import (  # noqa: E402
            InferConfig,
        )

        from ee_infer_pipeline import EEInferPipeline  # noqa: E402
        from ik_config import parse_ik_config  # noqa: E402

        os.environ.setdefault("WALLX_VISION_ATTN_IMPLEMENTATION", "flash_attention_2")
        self.infer_cfg = InferConfig(
            checkpoint_path=str(self.ckpt_dir),
            train_config_path=str(self.ckpt_dir / "config.yml"),
            robot_type="desktop",
            action_horizon=int(action_horizon),
            model_device=device,
            num_inference_timesteps=int(num_inference_timesteps),
            norm_key=NORM_KEY,
            cam_names=["face_view", "left_wrist_view", "right_wrist_view"],
        )
        _flatten_train_config(self.infer_cfg.train_config)
        ckpt_s = str(self.ckpt_dir)
        self.infer_cfg.train_config["processor_path"] = ckpt_s
        model_cfg = self.infer_cfg.train_config.get("model")
        if isinstance(model_cfg, dict):
            model_cfg["processor_path"] = ckpt_s
        self.wrapper = _TongWallxWrapper(self.infer_cfg, self.ckpt_dir)
        self.RobotStateActionData = RobotStateActionData
        self.ik = EEInferPipeline(self.sdk_root, n_exec=self.n_exec, ik=parse_ik_config(ik))
        LOGGER.info(
            "loaded %s norm_key=%s action_dim=%s horizon=%s n_exec=%s ik_mode=%s ik_solver=%s traj_opt=%s max_step=%.1f",
            self.ckpt_dir,
            self.wrapper.norm_key,
            self.infer_cfg.action_dim,
            self.infer_cfg.action_horizon,
            self.n_exec,
            self.ik.ik_settings.mode,
            self.ik.ik_settings.solver,
            self.ik.ik_settings.traj_opt,
            self.ik.ik_settings.max_joint_step_deg,
        )

    def apply_ik_config(self, raw) -> None:
        from ik_config import parse_ik_config  # noqa: E402

        prev = self.ik.ik_settings
        settings = self.ik.apply_ik_settings(parse_ik_config(raw))
        if settings == prev:
            return
        LOGGER.info(
            "IK config mode=%s solver=%s traj_opt=%s max_step=%.1f psi_step=%.1f dls_lambda=%.3f",
            settings.mode,
            settings.solver,
            settings.traj_opt,
            settings.max_joint_step_deg,
            settings.psi_step_deg,
            settings.dls_lambda,
        )

    def _build_observation(
        self, images: dict[str, np.ndarray], ee20: np.ndarray
    ) -> dict[str, Any]:
        rsa = self.RobotStateActionData(config=self.infer_cfg)
        for key, value in _ee20_to_state_fields(ee20).items():
            rsa.save_state_data_with_key(np.asarray(value, dtype=np.float64), key)
        horizon = int(self.infer_cfg.action_horizon)
        dim = int(self.infer_cfg.action_dim or (EE_DIM + PAD_DIM))
        mask = np.ones((1, horizon, dim), dtype=np.float32)
        if dim > EE_DIM:
            mask[..., EE_DIM:] = 0.0
        rsa.dof_mask = mask
        obs: dict[str, Any] = {"robot_state_action_data": rsa}
        for src_key, view in CAM_TO_VIEW.items():
            if src_key in images:
                obs[view] = to_hwc_uint8(images[src_key])
        missing = [k for k in self.wrapper.cam_names if k not in obs]
        if missing:
            raise KeyError(f"missing cameras {missing}; got {list(images)}")
        return obs

    def predict_rel_ee(
        self,
        images: dict[str, np.ndarray],
        ee20: np.ndarray,
        instruction: Optional[str] = None,
    ) -> np.ndarray:
        """Return unnormalized relative EE chunk (H, 20)."""
        prompt = (instruction or self.instruction).strip() or self.instruction
        obs = self._build_observation(images, ee20)
        out = self.wrapper.infer_flow_action(obs, prompt)
        pred = out["predict_action"]
        if hasattr(pred, "detach"):
            pred = pred.detach().cpu().numpy()
        pred = np.asarray(pred, dtype=np.float32)
        if pred.ndim == 3:
            pred = pred[0]
        if pred.shape[-1] < EE_DIM:
            raise ValueError(f"model action dim {pred.shape} < {EE_DIM}")
        return np.ascontiguousarray(pred[:, :EE_DIM])

    def predict_abs_ee20(
        self,
        images: dict[str, np.ndarray],
        joints16: np.ndarray,
        instruction: Optional[str] = None,
        *,
        n_exec: Optional[int] = None,
        rel_ee: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """FK → flow relative EE → absolute EE. No IK. Returns (T, 20)."""
        from ee_infer_pipeline import ee20_to_absolute

        joints = np.asarray(joints16, dtype=np.float64).reshape(-1)
        if joints.size < 16:
            raise ValueError(f"need 16D joints, got {joints.size}")
        joints = joints[:16]
        t0 = time.perf_counter()
        ee20 = self.ik.observe_ee20(joints)
        t1 = time.perf_counter()
        rel = (
            np.asarray(rel_ee, dtype=np.float32)
            if rel_ee is not None
            else self.predict_rel_ee(images, ee20, instruction)
        )
        t2 = time.perf_counter()
        abs_ee = ee20_to_absolute(rel, ee20)
        t3 = time.perf_counter()
        n = int(self.n_exec if n_exec is None else n_exec)
        n = max(1, min(n, int(abs_ee.shape[0])))
        out = np.ascontiguousarray(abs_ee[:n], dtype=np.float32)
        info = {
            "n_requested": n,
            "n_ee": int(out.shape[0]),
            "fk_ms": (t1 - t0) * 1e3,
            "flow_ms": (t2 - t1) * 1e3,
            "convert_ms": (t3 - t2) * 1e3,
            "ik_ms": 0.0,
            "rel_ee": rel,
            "abs_ee": abs_ee,
            "ee20": ee20,
        }
        return out, info

    def predict_logical16(
        self,
        images: dict[str, np.ndarray],
        joints16: np.ndarray,
        instruction: Optional[str] = None,
        *,
        n_exec: Optional[int] = None,
        rel_ee: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """FK → flow relative EE → absolute EE → continuous IK. Returns (T, 16).

        Open-loop eval still solves IK here. The websocket serve uses
        ``predict_abs_ee20`` and leaves IK to the client.

        Pass ``rel_ee`` to skip a second flow call (open-loop already has it).

        ``info`` also carries the per-stage wall clock in ms (``fk_ms`` /
        ``flow_ms`` / ``convert_ms`` / ``ik_ms``) so a slow infer can be
        attributed to the model or to the IK instead of being one black box.
        ``flow_ms`` is 0.0 when ``rel_ee`` is supplied (no model call happened).
        """
        from ee_infer_pipeline import plan_to_logical16

        joints = np.asarray(joints16, dtype=np.float64).reshape(-1)
        if joints.size < 16:
            raise ValueError(f"need 16D joints, got {joints.size}")
        joints = joints[:16]
        _abs_out, info = self.predict_abs_ee20(
            images, joints, instruction, n_exec=n_exec, rel_ee=rel_ee
        )
        n = int(info["n_requested"])
        t3 = time.perf_counter()
        plans = self.ik.ik_abs_chunk(joints, info["abs_ee"], n_exec=n)
        t4 = time.perf_counter()
        stage_ms = {
            "fk_ms": float(info["fk_ms"]),
            "flow_ms": float(info["flow_ms"]),
            "convert_ms": float(info["convert_ms"]),
            "ik_ms": (t4 - t3) * 1e3,
        }
        rows = [plan_to_logical16(p).astype(np.float32) for p in plans]
        if not rows:
            hold = np.asarray(joints, dtype=np.float32).reshape(1, 16)
            info = {
                "aborted_ik": True,
                "cartesian_limited": False,
                "held_ik": True,
                "n_planned": 0,
                "n_requested": n,
                "rejected_step_deg": 0.0,
                "cartesian_t": 0.0,
                **stage_ms,
                "rel_ee": info["rel_ee"],
                "abs_ee": info["abs_ee"],
                "ee20": info["ee20"],
            }
            return hold, info
        actions = np.stack(rows, axis=0)
        info = {
            "aborted_ik": len(plans) < n,
            "cartesian_limited": any(bool(p.cartesian_limited) for p in plans),
            "held_ik": any(bool(p.held) for p in plans),
            "n_planned": len(plans),
            "n_requested": n,
            "rejected_step_deg": max((float(p.rejected_step_deg) for p in plans), default=0.0),
            "cartesian_t": min((float(p.cartesian_t) for p in plans), default=1.0),
            **stage_ms,
            "rel_ee": info["rel_ee"],
            "abs_ee": info["abs_ee"],
            "ee20": info["ee20"],
            "max_step_deg": max(
                (max(p.left_step_max_deg, p.right_step_max_deg) for p in plans),
                default=0.0,
            ),
        }
        return actions, info
