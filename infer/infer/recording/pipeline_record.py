"""标准推理 PIPELINE 数据记录。

每次 test 写入:
  - pipeline_record.json     完整结构化记录（训练 + 推理 + 对照差）
  - train_record.json        训练配置快照（从权重旁固化文件拷入）
  - pipeline_record_history.jsonl  父目录追加一行，供后续表格分析

schema: lingbot.infer.pipeline_record / v1
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import socket
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from infer.config_loader import InferConfig, vlm_planner_enabled
from infer.recording.git_info import get_infer_git_info

logger = logging.getLogger(__name__)

PIPELINE_RECORD_FILENAME = "pipeline_record.json"
PIPELINE_HISTORY_FILENAME = "pipeline_record_history.jsonl"
TRAIN_RECORD_FILENAME = "train_record.json"
SCHEMA_NAME = "lingbot.infer.pipeline_record"
SCHEMA_VERSION = 1

_CKPT_STEP_RE = re.compile(r"checkpoint_step_(\d+)", re.I)

_TRAIN_INFER_COMPARE_KEYS = (
    "attn_window",
    "guidance_scale",
    "action_guidance_scale",
    "num_inference_steps",
    "action_num_inference_steps",
    "snr_shift",
    "action_snr_shift",
    "video_exec_step",
    "frame_chunk_size",
    "action_per_frame",
    "env_type",
    "height",
    "width",
    "action_dim",
    "action_norm_method",
    "used_action_channel_ids",
    "action_space",
    "control",
)


def write_pipeline_record(
    test_dir: Path,
    cfg: InferConfig,
    *,
    policy: Any,
    git_info: Optional[Dict[str, Any]] = None,
    va_infer: Optional[Dict[str, Any]] = None,
    client_hparams: Optional[Dict[str, Any]] = None,
    started_at: Optional[str] = None,
) -> Dict[str, Any]:
    """写标准 PIPELINE 记录，返回完整 record。失败只记日志，不打断推理。"""
    test_dir = Path(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)
    record = build_pipeline_record(
        test_dir,
        cfg,
        policy=policy,
        git_info=git_info,
        va_infer=va_infer,
        client_hparams=client_hparams,
        started_at=started_at,
    )
    _copy_train_sidecar(test_dir, record.get("training") or {})
    path = test_dir / PIPELINE_RECORD_FILENAME
    path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    compact = compact_pipeline_row(record)
    hist = test_dir.parent / PIPELINE_HISTORY_FILENAME
    try:
        with hist.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(compact, ensure_ascii=False, default=str) + "\n")
    except OSError:
        logger.exception("无法追加 %s", hist)
    return record


def build_pipeline_record(
    test_dir: Path,
    cfg: InferConfig,
    *,
    policy: Any,
    git_info: Optional[Dict[str, Any]] = None,
    va_infer: Optional[Dict[str, Any]] = None,
    client_hparams: Optional[Dict[str, Any]] = None,
    started_at: Optional[str] = None,
) -> Dict[str, Any]:
    git_info = dict(git_info or get_infer_git_info())
    va_infer = dict(va_infer or {})
    va_infer.pop("config_yaml", None)
    client_hparams = dict(client_hparams or {})
    started_at = started_at or datetime.now().astimezone().isoformat()
    training = _load_training_record(cfg)
    checkpoint = _checkpoint_section(cfg, va_infer)
    inference = _inference_section(cfg, policy, va_infer, client_hparams)
    train_infer_diff = _train_infer_diff(
        training.get("hparams") or {},
        {**va_infer, **(inference.get("server") or {})},
    )
    train_infer_diff.update(_rtc_train_infer_diff(inference.get("client") or {}))
    inference["train_infer_diff"] = train_infer_diff
    return {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "recorded_at": datetime.now().astimezone().isoformat(),
        "identity": {
            "experiment_name": cfg.experiment_name,
            "policy_type": cfg.policy.type.value,
            "engine": cfg.engine.value,
            "test_dir": str(test_dir),
            "started_at": started_at,
            "host": socket.gethostname(),
            "user": os.environ.get("USER") or os.environ.get("LOGNAME"),
            "gpu": _gpu_snapshot(),
        },
        "code": git_info,
        "checkpoint": checkpoint,
        "training": training,
        "inference": inference,
        "robot": _robot_section(cfg),
        "dataset": _dataset_section(cfg, training),
        "artifacts": {
            "pipeline_record": PIPELINE_RECORD_FILENAME,
            "train_record": TRAIN_RECORD_FILENAME,
            "client_config": "client_config.yaml",
            "server_config": "server_config.yaml",
            "meta": "meta.json",
            "inference_params": "inference_params.json",
        },
    }


def compact_pipeline_row(record: Dict[str, Any]) -> Dict[str, Any]:
    """history jsonl 一行：后续 pandas / 对照实验直接读这个。"""
    ident = record.get("identity") or {}
    ckpt = record.get("checkpoint") or {}
    train_h = (record.get("training") or {}).get("hparams") or {}
    infer = record.get("inference") or {}
    server = infer.get("server") or {}
    client = infer.get("client") or {}
    diff = infer.get("train_infer_diff") or {}
    code = record.get("code") or {}
    return {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "recorded_at": record.get("recorded_at"),
        "started_at": ident.get("started_at"),
        "test_dir": ident.get("test_dir"),
        "experiment_name": ident.get("experiment_name"),
        "policy_type": ident.get("policy_type"),
        "host": ident.get("host"),
        "ckpt_dir": ckpt.get("ckpt_dir"),
        "checkpoint_step": ckpt.get("checkpoint_step"),
        "weight_path": ckpt.get("weight_path"),
        "task_prompt": client.get("task_prompt"),
        "n_action_steps": client.get("n_action_steps"),
        "fps": client.get("fps"),
        "va_obs_fps": client.get("va_obs_fps"),
        "action_space": client.get("action_space"),
        "control": client.get("control"),
        "lingbot_robo_name": client.get("lingbot_robo_name"),
        "num_inference_steps": server.get("num_inference_steps"),
        "action_num_inference_steps": server.get("action_num_inference_steps"),
        "guidance_scale": server.get("guidance_scale"),
        "action_guidance_scale": server.get("action_guidance_scale"),
        "attn_window": server.get("attn_window"),
        "video_exec_step": server.get("video_exec_step"),
        "snr_shift": server.get("snr_shift"),
        "action_snr_shift": server.get("action_snr_shift"),
        "infer_source": server.get("source"),
        "train_num_steps": train_h.get("num_steps"),
        "train_learning_rate": train_h.get("learning_rate"),
        "train_batch_size": train_h.get("batch_size"),
        "train_cfg_prob": train_h.get("cfg_prob"),
        "train_guidance_scale": train_h.get("guidance_scale"),
        "train_num_inference_steps": train_h.get("num_inference_steps"),
        "train_action_num_inference_steps": train_h.get("action_num_inference_steps"),
        "train_action_space": (record.get("training") or {}).get("action_space")
        or train_h.get("action_space"),
        "train_control": (record.get("training") or {}).get("control")
        or train_h.get("control"),
        "train_infer_changed_keys": sorted(diff.keys()),
        "infer_git_commit_short": code.get("infer_git_commit_short"),
        "infer_git_branch": code.get("infer_git_branch"),
        "infer_git_dirty": code.get("infer_git_dirty"),
    }


def resolve_train_record_path(cfg: InferConfig) -> Optional[Path]:
    ckpt = Path(cfg.policy.ckpt_dir).expanduser()
    candidates: List[Path] = []
    explicit = getattr(cfg.policy, "va_train_config", None)
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.extend(
        [
            ckpt / TRAIN_RECORD_FILENAME,
            ckpt.parent / TRAIN_RECORD_FILENAME,
            ckpt.parent.parent / TRAIN_RECORD_FILENAME,
            ckpt.parent.parent.parent / TRAIN_RECORD_FILENAME,
        ]
    )
    ds_root = getattr(cfg.policy, "dataset_root", None)
    if ds_root:
        candidates.append(Path(ds_root).expanduser() / "meta" / "action_space.json")
        candidates.append(Path(ds_root).expanduser() / TRAIN_RECORD_FILENAME)
    for path in candidates:
        if path.is_file() and path.suffix.lower() == ".json":
            return path
    return None


def _load_training_record(cfg: InferConfig) -> Dict[str, Any]:
    path = resolve_train_record_path(cfg)
    if path is None:
        return {
            "found": False,
            "path": None,
            "hparams": {},
            "note": "未找到 train_record.json；请把训练配置固化到 ckpt 旁或 policy.va_train_config",
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("读取 train_record 失败 (%s): %s", path, exc)
        return {"found": False, "path": str(path), "hparams": {}, "error": str(exc)}
    if not isinstance(payload, dict):
        return {"found": False, "path": str(path), "hparams": {}, "error": "not a json object"}
    hparams = payload.get("hparams")
    if not isinstance(hparams, dict):
        hparams = {
            k: v
            for k, v in payload.items()
            if k not in ("schema", "schema_version", "source_files", "notes")
        }
    return {
        "found": True,
        "path": str(path),
        "schema": payload.get("schema"),
        "schema_version": payload.get("schema_version"),
        "run_name": payload.get("run_name"),
        "save_root": payload.get("save_root"),
        "dataset_path": payload.get("dataset_path"),
        "source_files": payload.get("source_files") or [],
        "hparams": hparams,
        "notes": payload.get("notes"),
        "dataset": payload.get("dataset") if isinstance(payload.get("dataset"), dict) else {},
        "action_space": payload.get("action_space")
        or (payload.get("dataset") or {}).get("action_space")
        or hparams.get("action_space"),
        "obs_space": payload.get("obs_space")
        or (payload.get("dataset") or {}).get("obs_space")
        or hparams.get("obs_space"),
        "control": payload.get("control")
        or (payload.get("dataset") or {}).get("control")
        or hparams.get("control"),
    }


def _copy_train_sidecar(test_dir: Path, training: Dict[str, Any]) -> None:
    src = training.get("path")
    if src:
        src_path = Path(src)
        if src_path.is_file():
            shutil.copy2(src_path, test_dir / TRAIN_RECORD_FILENAME)
    for item in training.get("source_files") or []:
        raw = item.get("path") if isinstance(item, dict) else item
        if not raw:
            continue
        p = Path(str(raw)).expanduser()
        if not p.is_file():
            continue
        dest_dir = test_dir / "train_configs"
        dest_dir.mkdir(exist_ok=True)
        shutil.copy2(p, dest_dir / p.name)


def _checkpoint_section(cfg: InferConfig, va_infer: Dict[str, Any]) -> Dict[str, Any]:
    ckpt_dir = Path(cfg.policy.ckpt_dir).expanduser()
    weight = _find_weight_file(ckpt_dir, va_infer.get("pretrained_model_path"))
    transformer_cfg = None
    cfg_path = ckpt_dir / "transformer" / "config.json"
    if cfg_path.is_file():
        try:
            transformer_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            transformer_cfg = None
    stat = _file_stat(weight)
    return {
        "ckpt_dir": str(ckpt_dir),
        "checkpoint_step": _checkpoint_step(ckpt_dir, weight),
        "weight_path": str(weight) if weight else None,
        "weight_size_bytes": stat.get("size_bytes"),
        "weight_mtime": stat.get("mtime"),
        "pretrained_model_path": va_infer.get("pretrained_model_path"),
        "base_model_path": va_infer.get("base_model_path"),
        "transformer_config": transformer_cfg,
        "transformer_config_path": str(cfg_path) if cfg_path.is_file() else None,
    }


def _inference_section(
    cfg: InferConfig,
    policy: Any,
    va_infer: Dict[str, Any],
    client_hparams: Dict[str, Any],
) -> Dict[str, Any]:
    pol = cfg.policy
    server_keys = (
        "source",
        "config_path",
        "pretrained_model_path",
        "base_model_path",
        "param_dtype",
        "enable_offload",
        "env_type",
        "attn_window",
        "frame_chunk_size",
        "height",
        "width",
        "patch_size",
        "action_dim",
        "action_per_frame",
        "used_action_channel_ids",
        "action_norm_method",
        "num_inference_steps",
        "action_num_inference_steps",
        "video_exec_step",
        "snr_shift",
        "action_snr_shift",
        "guidance_scale",
        "action_guidance_scale",
        "num_chunks_to_infer",
        "decode_pred_video",
        "output_type",
        "scheduler",
        "flow_shift",
        "obs_cam_keys",
        "server_prompt",
        "started_at",
    )
    server = {k: va_infer[k] for k in server_keys if k in va_infer}
    client = {
        "config_path": str(cfg.config_path) if cfg.config_path else None,
        "lingbot_host": client_hparams.get("lingbot_host")
        or getattr(pol, "lingbot_host", None),
        "lingbot_port": client_hparams.get("lingbot_port")
        or getattr(pol, "lingbot_port", None),
        "n_action_steps": int(getattr(policy, "n_action_steps", 0) or 0),
        "n_action_steps_override": pol.n_action_steps_override,
        "chunk_size": getattr(policy, "chunk_size", None),
        "use_action_chunk": bool(pol.use_action_chunk),
        "action_scale": float(pol.action_scale),
        "action_clip_abs": pol.action_clip_abs,
        "va_obs_fps": pol.va_obs_fps,
        "fps": cfg.session.fps,
        "task_prompt": pol.task_prompt,
        "prompt_from_subtask": bool(getattr(pol, "prompt_from_subtask", False)),
        "vlm_planner": (
            {
                "enabled": vlm_planner_enabled(pol),
                "weight_dir": str(pol.vlm_planner.weight_dir)
                if pol.vlm_planner
                else None,
                "device": pol.vlm_planner.device if pol.vlm_planner else None,
                "camera": pol.vlm_planner.camera if pol.vlm_planner else None,
                "interval_sec": pol.vlm_planner.interval_sec if pol.vlm_planner else None,
            }
            if getattr(pol, "vlm_planner", None) or getattr(pol, "prompt_from_subtask", False)
            else None
        ),
        "camera_keys": list(pol.camera_keys) if pol.camera_keys else None,
        "keep_dims": list(pol.keep_dims) if pol.keep_dims else None,
        "rtc_enabled": bool(pol.rtc_enabled),
        "rtc_config": pol.rtc_config,
        "async_blending": bool(getattr(pol, "async_blending", False)),
        "blend_horizon": getattr(pol, "blend_horizon", None),
        "blend_schedule": getattr(pol, "blend_schedule", None),
        "lingbot_robo_name": getattr(pol, "lingbot_robo_name", None),
        "action_space": getattr(policy, "action_space", None)
        or getattr(pol, "action_space", None),
        "control": getattr(policy, "control", None),
        "action_space_source": getattr(policy, "action_space_source", None),
        "robot_config_path": getattr(policy, "robot_config_path", None)
        or (
            str(pol.lingbot_robot_config)
            if getattr(pol, "lingbot_robot_config", None)
            else None
        ),
        "va_server_config": str(pol.va_server_config) if pol.va_server_config else None,
        "va_train_config": str(pol.va_train_config) if getattr(pol, "va_train_config", None) else None,
        "session_task_mode": cfg.session.task_mode,
        "pedal_task_control": bool(cfg.session.pedal_task_control),
        "tasks_per_group": cfg.session.tasks_per_group
        if cfg.session.tasks_per_group is not None
        else pol.tasks_per_group,
        "max_duration_sec": cfg.session.max_duration_sec,
    }
    return {"server": server, "client": client}


def _robot_section(cfg: InferConfig) -> Dict[str, Any]:
    net = cfg.network
    dr = cfg.direct_robot
    cam = cfg.camera_sender
    devices = []
    for dev in cam.devices:
        devices.append(
            {
                "key": dev.key,
                "serial": dev.serial,
                "width": dev.width,
                "height": dev.height,
                "fps": dev.fps,
            }
        )
    return {
        "high_host": net.high_host,
        "low_host": net.low_host,
        "state_port": net.state_port,
        "control_port": net.control_port,
        "gripper_port": net.gripper_port,
        "gripper_state_port": net.gripper_state_port,
        "camera_port": net.camera_port,
        "direct_robot": bool(dr.enabled),
        "gripper_mode": dr.gripper_mode if dr.enabled else None,
        "gripper_state_source": dr.gripper_state_source if dr.enabled else None,
        "robot_l_host": dr.robot_l_host if dr.enabled else None,
        "robot_r_host": dr.robot_r_host if dr.enabled else None,
        "init_pose_source": dr.init_pose_source if dr.enabled else None,
        "reset_to_init_on_task_start": dr.reset_to_init_on_task_start if dr.enabled else None,
        "cameras": devices,
        "enable_depth": cam.enable_depth,
        "camera_transport": cam.transport,
    }


def _dataset_section(cfg: InferConfig, training: Dict[str, Any]) -> Dict[str, Any]:
    root = cfg.policy.dataset_root
    info = None
    n_episodes = None
    if root is not None:
        info_path = Path(root) / "meta" / "info.json"
        if info_path.is_file():
            try:
                info = json.loads(info_path.read_text(encoding="utf-8"))
                n_episodes = info.get("total_episodes")
            except (OSError, json.JSONDecodeError):
                info = None
    return {
        "dataset_root": str(root) if root else None,
        "train_dataset_path": training.get("dataset_path"),
        "total_episodes": n_episodes,
        "robot_type": (info or {}).get("robot_type") if isinstance(info, dict) else None,
        "dataset_fps": (info or {}).get("fps") if isinstance(info, dict) else None,
        "codebase_version": (info or {}).get("codebase_version") if isinstance(info, dict) else None,
    }


def _rtc_train_infer_diff(client: Dict[str, Any]) -> Dict[str, Any]:
    """RTC / 异步推理是推理 CONFIG，训练侧没有对应项。"""
    rtc = client.get("rtc_config") or {}
    enabled = bool(client.get("rtc_enabled"))
    blending = bool(client.get("async_blending") or rtc.get("async_blending"))
    out = {
        "rtc": {"train": False, "infer": enabled},
        "async_inference": {
            "train": False,
            "infer": bool(enabled and rtc.get("async_inference")),
        },
        "async_blending": {
            "train": False,
            "infer": blending,
        },
    }
    return {k: v for k, v in out.items() if v["train"] != v["infer"]}


def _train_infer_diff(train_h: Dict[str, Any], infer_h: Dict[str, Any]) -> Dict[str, Any]:
    diff: Dict[str, Any] = {}
    for key in _TRAIN_INFER_COMPARE_KEYS:
        if key not in train_h or key not in infer_h:
            continue
        train_v = train_h.get(key)
        infer_v = infer_h.get(key)
        if train_v != infer_v:
            diff[key] = {"train": train_v, "infer": infer_v}
    return diff


def _checkpoint_step(ckpt_dir: Path, weight: Optional[Path]) -> Optional[int]:
    for text in (str(ckpt_dir), str(weight) if weight else ""):
        m = _CKPT_STEP_RE.search(text)
        if m:
            return int(m.group(1))
    return None


def _find_weight_file(ckpt_dir: Path, pretrained_model_path: Any) -> Optional[Path]:
    candidates: List[Path] = []
    if pretrained_model_path:
        p = Path(str(pretrained_model_path))
        candidates.append(p)
        candidates.append(p / "diffusion_pytorch_model.safetensors")
    tf = ckpt_dir / "transformer"
    candidates.append(tf / "diffusion_pytorch_model.safetensors")
    if tf.is_dir():
        candidates.extend(sorted(tf.glob("*.safetensors")))
    for path in candidates:
        if path.is_file():
            return path
        if path.is_symlink():
            try:
                resolved = path.resolve()
            except OSError:
                continue
            if resolved.is_file():
                return resolved
    return None


def _file_stat(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        st = path.stat()
    except OSError:
        return {}
    return {
        "size_bytes": int(st.st_size),
        "mtime": datetime.fromtimestamp(st.st_mtime).astimezone().isoformat(),
    }


def _gpu_snapshot() -> Optional[List[Dict[str, str]]]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    gpus: List[Dict[str, str]] = []
    for line in (completed.stdout or "").splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 3:
            gpus.append({"index": parts[0], "name": parts[1], "memory_total": parts[2]})
    return gpus or None


def format_pipeline_record_line(record: Dict[str, Any]) -> str:
    row = compact_pipeline_row(record)
    diff_keys = row.get("train_infer_changed_keys") or []
    return (
        "PIPELINE 记录: ckpt_step=%s video_steps=%s action_steps=%s "
        "guidance=%s/%s attn_window=%s n_action_steps=%s "
        "action_space=%s control=%s "
        "train_vs_infer_changed=%s -> %s"
        % (
            row.get("checkpoint_step"),
            row.get("num_inference_steps"),
            row.get("action_num_inference_steps"),
            row.get("guidance_scale"),
            row.get("action_guidance_scale"),
            row.get("attn_window"),
            row.get("n_action_steps"),
            row.get("action_space"),
            row.get("control"),
            ",".join(diff_keys) if diff_keys else "none",
            PIPELINE_RECORD_FILENAME,
        )
    )
