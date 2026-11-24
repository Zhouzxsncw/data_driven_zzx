"""推理功能 / I/O 清单：突出「会改变控制或观测」的核心项。

产物:
  - feature_manifest.json（机器可读）
  - run.log 中文块：核心项写清「开了会怎样 / 关了会怎样」
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from infer.config_loader import InferConfig, vlm_planner_enabled
from infer.recording.git_info import get_infer_git_info

FEATURE_MANIFEST_FILENAME = "feature_manifest.json"
FEATURE_MANIFEST_SCHEMA_VERSION = 2


def _path_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def _camera_keys_label(camera_keys: Optional[Sequence[str]]) -> str:
    if camera_keys is None:
        return "auto（从 checkpoint 推断）"
    if len(camera_keys) == 0:
        return "纯 state（无相机图）"
    return ", ".join(str(k) for k in camera_keys)


def _item(
    *,
    item_id: str,
    name: str,
    group: str,
    priority: str,
    status: str,
    value: Any = None,
    effect: str,
    config_key: str = "",
) -> Dict[str, Any]:
    """一条清单项。

    priority: core | secondary
    status: 开 / 关 / 模式值（如 udp、buffer）
    effect: 对推理行为意味着什么（人读重点）
    """
    return {
        "id": item_id,
        "name": name,
        "group": group,
        "priority": priority,
        "status": status,
        "value": value,
        "effect": effect,
        "config_key": config_key,
    }


def _device_summaries(cfg: InferConfig) -> List[Dict[str, Any]]:
    devices: List[Dict[str, Any]] = []
    for i, dev in enumerate(cfg.camera_sender.devices):
        devices.append(
            {
                "index": i,
                "key": dev.key,
                "serial": dev.serial,
                "capture_hw": (
                    [dev.width, dev.height]
                    if dev.width is not None and dev.height is not None
                    else None
                ),
                "model_hw": (
                    [dev.model_width, dev.model_height]
                    if dev.model_width is not None and dev.model_height is not None
                    else None
                ),
                "fps": int(dev.fps),
            }
        )
    return devices


def build_feature_manifest(
    cfg: InferConfig,
    *,
    runtime: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    rt = dict(runtime or {})
    pol = cfg.policy
    cam = cfg.camera_sender
    dr = cfg.direct_robot
    exo = cfg.exo_safety
    rec = cfg.recording
    ses = cfg.session
    log = cfg.logging
    ev = cfg.evaluation
    ab = cfg.ablation
    bi = cfg.batch_infer

    camera_keys = rt.get("camera_keys", pol.camera_keys)
    pedal_on = bool(rt.get("pedal_task_control", ses.pedal_task_control))
    exo_on = bool(rt.get("exo_safety_enabled", exo.enabled))
    golden_on = bool(rt.get("golden_capture", rec.golden_capture))
    n_action_steps = rt.get("n_action_steps", pol.n_action_steps_override)
    chunk_size = rt.get("chunk_size")
    test_dir = rt.get("test_dir")
    tasks_per_group = rt.get(
        "tasks_per_group",
        ses.tasks_per_group if ses.tasks_per_group is not None else pol.tasks_per_group,
    )

    needs_camera = not (isinstance(camera_keys, list) and len(camera_keys) == 0)
    camera_on = bool(cam.enabled) and needs_camera
    grip_mode = str(dr.gripper_mode).lower() if dr.enabled else "none"
    grip_src = str(dr.gripper_state_source).lower() if dr.enabled else "n/a"

    # ---- 核心：改变控制/观测 ----
    core: List[Dict[str, Any]] = [
        _item(
            item_id="engine",
            name="推理引擎",
            group="control",
            priority="core",
            status=cfg.engine.value,
            value=cfg.engine.value,
            effect=(
                "native=新引擎(SyncEngine+脚踏组)；legacy=旧缓冲对齐逻辑"
            ),
            config_key="engine",
        ),
        _item(
            item_id="direct_robot",
            name="机械臂连接",
            group="control",
            priority="core",
            status="直连 RealMan SDK" if dr.enabled else "UDP 下位机桥",
            value="direct" if dr.enabled else "udp_bridge",
            effect=(
                "开=上位机直连臂；关=走旧 UDP 桥收状态/发控制"
                if dr.enabled
                else "当前走 UDP 桥，不读 RM SDK"
            ),
            config_key="direct_robot.enabled",
        ),
        _item(
            item_id="gripper_control",
            name="夹爪控制通道",
            group="control",
            priority="core",
            status={
                "udp": "UDP 下发 (30015)",
                "local": "本机 Dynamixel",
                "none": "不控夹爪",
            }.get(grip_mode, grip_mode),
            value=grip_mode,
            effect=(
                "udp=指令发给下位机 only_gripper；"
                "local=本机串口直接拧电机；"
                "none=关节动、夹爪不下发"
            ),
            config_key="direct_robot.gripper_mode",
        ),
        _item(
            item_id="gripper_state_source",
            name="夹爪观测来源（进模型的 state[14/15]）",
            group="observation",
            priority="core",
            status={
                "udp": "UDP 实测 (30016 Dynamixel)",
                "buffer": "同步缓冲池对齐值",
                "n/a": "不适用",
            }.get(grip_src, grip_src),
            value=grip_src,
            effect=(
                "udp=用下位机刚推上来的夹爪开口比覆盖观测（更贴近真机）；"
                "buffer=用与相机对齐的 buffer 夹爪维；若缺失或占位 0，"
                "自动用上一拍 action 夹爪（再退到初始开口），不再写死 0"
            ),
            config_key="direct_robot.gripper_state_source",
        ),
        _item(
            item_id="camera_input",
            name="视觉输入",
            group="observation",
            priority="core",
            status="开" if camera_on else "关",
            value={
                "enabled": camera_on,
                "keys_label": _camera_keys_label(
                    None if camera_keys is None else list(camera_keys)
                ),
                "transport": cam.transport if camera_on else None,
            },
            effect=(
                f"开=模型吃图像，keys={_camera_keys_label(None if camera_keys is None else list(camera_keys))}；"
                "关=纯 state 推理（camera_keys=[] 或缺相机）"
            ),
            config_key="camera_sender.enabled / policy.camera_keys",
        ),
        _item(
            item_id="depth",
            name="深度图",
            group="observation",
            priority="core",
            status="开" if (camera_on and cam.enable_depth) else "关",
            value=bool(camera_on and cam.enable_depth),
            effect=(
                "开=采 depth；recording.save_depth_raw 落 uint16 mm PNG（可分析），"
                "save_depth_videos 另存 JET 预览 mp4；关=只有 RGB"
            ),
            config_key="camera_sender.enable_depth",
        ),
        _item(
            item_id="keep_dims",
            name="模型状态维 keep_dims",
            group="observation",
            priority="core",
            status=str(list(pol.keep_dims)),
            value=list(pol.keep_dims),
            effect="决定送进模型的关节/夹爪子集；改它等于换观测与动作布局",
            config_key="policy.keep_dims",
        ),
        _item(
            item_id="rtc",
            name="Pi0.5 RTC",
            group="control",
            priority="core",
            status="开" if pol.rtc_enabled else "关",
            value=(
                pol.rtc_config
                if pol.rtc_enabled
                else bool(pol.rtc_enabled)
            ),
            effect="开=RTC prefix inpainting；关=独立采样",
            config_key="policy.rtc_enabled / policy.rtc_config",
        ),
        _item(
            item_id="async_blending",
            name="Async blending",
            group="control",
            priority="core",
            status="开" if bool(getattr(pol, "async_blending", False)) else "关",
            value={
                "async_blending": bool(getattr(pol, "async_blending", False)),
                "blend_horizon": getattr(pol, "blend_horizon", None),
                "blend_schedule": getattr(pol, "blend_schedule", None),
            },
            effect="开=新包到了把 leftover 与新 chunk 重叠融合；关=chunk 硬切",
            config_key="policy.async_blending / policy.blend_horizon / policy.blend_schedule",
        ),
        _item(
            item_id="vlm_planner",
            name="两层推理 VLM→subtask",
            group="language",
            priority="core",
            status="开" if vlm_planner_enabled(pol) else "关",
            value={
                "prompt_from_subtask": bool(getattr(pol, "prompt_from_subtask", False)),
                "enabled": vlm_planner_enabled(pol),
                "weight_dir": str(pol.vlm_planner.weight_dir)
                if getattr(pol, "vlm_planner", None)
                else None,
            },
            effect=(
                "开=上层 VLM 每 interval 生成当前 subtask，动作模型吃这段字；"
                "关=整场只用 task_prompt（与 prompt_from_subtask 训练不对齐）"
            ),
            config_key="policy.prompt_from_subtask / policy.vlm_planner",
        ),
        _item(
            item_id="use_action_chunk",
            name="整段 Chunk 一次下发",
            group="control",
            priority="core",
            status="开" if pol.use_action_chunk else "关",
            value=bool(pol.use_action_chunk),
            effect="开=按 chunk 批执行；关=按 n_action_steps 逐步开环（当前常用）",
            config_key="policy.use_action_chunk",
        ),
        _item(
            item_id="pedal_task_control",
            name="脚踏任务段",
            group="session",
            priority="core",
            status="开" if pedal_on else "关",
            value=bool(pedal_on),
            effect=(
                f"开=用 {cfg.pedal.task_toggle} 切任务段，连续采多任务；"
                "关=单次 test 计时跑完即停"
            ),
            config_key="session.pedal_task_control",
        ),
        _item(
            item_id="exo_safety",
            name="外骨骼接管",
            group="session",
            priority="core",
            status="开" if exo_on else "关",
            value=bool(exo_on),
            effect=(
                f"开={cfg.pedal.exo_takeover} 对齐后 "
                f"{cfg.pedal.confirm} 人控 / {cfg.pedal.exo_safety_infer} 回推理；"
                "关=无外骨骼链路"
            ),
            config_key="exo_safety.enabled",
        ),
        _item(
            item_id="init_pose",
            name="回初始位 / 随机初始化",
            group="session",
            priority="core",
            status=(
                f"组开始复位({dr.init_pose_source})"
                if dr.enabled and dr.reset_to_init_on_group_start
                else "不自动复位"
            )
            + (
                " + 任务段随机位姿"
                if dr.enabled and dr.random_task_init_pose
                else ""
            ),
            value={
                "reset_to_init_on_group_start": bool(
                    dr.enabled and dr.reset_to_init_on_group_start
                ),
                "init_pose_source": dr.init_pose_source if dr.enabled else None,
                "random_task_init_pose": bool(
                    dr.enabled and dr.random_task_init_pose
                ),
            },
            effect=(
                "复位开=每组开始把臂回到标准/数据集位；"
                "随机开=每个任务段在圆盘内随机左臂初始位（测泛化）"
            ),
            config_key="direct_robot.reset_to_init_on_group_start / random_task_init_pose",
        ),
    ]

    # ---- 次要：录制 / 评估 / 批量（不影响当场控制）----
    secondary: List[Dict[str, Any]] = [
        _item(
            item_id="save_videos",
            name="录 RGB 视频",
            group="recording",
            priority="secondary",
            status="开" if (rec.enabled and rec.save_videos) else "关",
            value=bool(rec.enabled and rec.save_videos),
            effect="开→videos/；关=不落盘视频（省磁盘，不影响控制）",
            config_key="recording.save_videos",
        ),
        _item(
            item_id="save_depth_raw",
            name="录 depth raw (mm)",
            group="recording",
            priority="secondary",
            status="开" if (rec.enabled and rec.save_depth_raw) else "关",
            value=bool(rec.enabled and rec.save_depth_raw),
            effect="开→videos/*_depth*_u16/ 下 uint16 mm PNG；可分析/进训，不是伪彩",
            config_key="recording.save_depth_raw",
        ),
        _item(
            item_id="save_depth_videos",
            name="录 depth 伪彩预览",
            group="recording",
            priority="secondary",
            status="开" if (rec.enabled and rec.save_depth_videos) else "关",
            value=bool(rec.enabled and rec.save_depth_videos),
            effect="开→JET 伪彩 mp4（仅预览，不可反算 mm）",
            config_key="recording.save_depth_videos",
        ),
        _item(
            item_id="save_trajectory",
            name="录轨迹",
            group="recording",
            priority="secondary",
            status="开" if (rec.enabled and rec.save_trajectory) else "关",
            value=bool(rec.enabled and rec.save_trajectory),
            effect="开→trajectory_data/；关=没有离线轨迹文件",
            config_key="recording.save_trajectory",
        ),
        _item(
            item_id="eval_log",
            name="eval_log + 对比图",
            group="eval",
            priority="secondary",
            status=(
                "开(含对比图)"
                if ev.eval_log_enabled and ev.eval_log_generate_plots
                else ("开(无图)" if ev.eval_log_enabled else "关")
            ),
            value={
                "eval_log": bool(ev.eval_log_enabled),
                "plots": bool(ev.eval_log_enabled and ev.eval_log_generate_plots),
            },
            effect="开→eval_log.jsonl / eval_plots/；关=事后难画指令vs反馈曲线",
            config_key="evaluation.eval_log_*",
        ),
        _item(
            item_id="infer_probe",
            name="infer_probe.jsonl（AI-readable）",
            group="eval",
            priority="secondary",
            status="开" if ev.eval_log_enabled else "关",
            value=bool(ev.eval_log_enabled),
            effect="开→infer_probe.jsonl（IK 计数 typed，不含图）；关=AI 只能从 eval_log 兼容解码",
            config_key="evaluation.eval_log_enabled",
        ),
        _item(
            item_id="evaluation_report",
            name="自动评估报告",
            group="eval",
            priority="secondary",
            status="开" if ev.enabled else "关",
            value=bool(ev.enabled),
            effect="开→evaluation_report.*；关=只有原始日志",
            config_key="evaluation.enabled",
        ),
        _item(
            item_id="device_health",
            name="设备健康/状态轮询计时",
            group="logging",
            priority="secondary",
            status="开" if log.device_health_enabled else "关",
            value=bool(log.device_health_enabled),
            effect="开→device_health*.json(l)，含反馈超时与 poll 耗时探针",
            config_key="logging.device_health_enabled",
        ),
        _item(
            item_id="batch_score_sheet",
            name="批量人工评分表",
            group="batch",
            priority="secondary",
            status="开" if bi.feedback_score_sheet else "关",
            value=bool(bi.feedback_score_sheet),
            effect="开=每组生成评分 .xlsx；关=不弹表",
            config_key="batch_infer.feedback_score_sheet",
        ),
        _item(
            item_id="batch_vla_auto_eval",
            name="批量 VLA 自动评估",
            group="batch",
            priority="secondary",
            status="开" if bi.vla_auto_eval.enabled else "关",
            value=bool(bi.vla_auto_eval.enabled),
            effect=f"开→每组 {bi.vla_auto_eval.output_dirname}/；关=不做规则打分",
            config_key="batch_infer.vla_auto_eval.enabled",
        ),
        _item(
            item_id="ablation_yaml",
            name="YAML 消融会话",
            group="batch",
            priority="secondary",
            status="开" if ab.enabled else "关",
            value=bool(ab.enabled),
            effect="开=InferSession 按 ablation.runs 扫配置；批量真机一般保持关",
            config_key="ablation.enabled",
        ),
        _item(
            item_id="interactive_task_prompt",
            name="任务间改 prompt (OOD)",
            group="session",
            priority="secondary",
            status="开" if ses.interactive_task_prompt else "关",
            value=bool(ses.interactive_task_prompt),
            effect="开=每任务段可改语言指令；ACT 通常关",
            config_key="session.interactive_task_prompt",
        ),
        _item(
            item_id="golden_capture",
            name="Golden Capture",
            group="recording",
            priority="secondary",
            status="开" if golden_on else "关",
            value=bool(golden_on),
            effect="开=额外抓取对照数据；日常推理关",
            config_key="recording.golden_capture",
        ),
    ]

    items = core + secondary
    inputs: Dict[str, Any] = {
        "experiment_name": cfg.experiment_name,
        "config_path": _path_str(cfg.config_path),
        "infer_git": get_infer_git_info(),
        "policy": {
            "type": pol.type.value,
            "act_backend": pol.act_backend,
            "pi05_backend": getattr(pol, "pi05_backend", "lerobot"),
            "openpi_config_name": getattr(pol, "openpi_config_name", None),
            "openpi_action_dim": getattr(pol, "openpi_action_dim", None),
            "openpi_adapt_to_pi": getattr(pol, "openpi_adapt_to_pi", None),
            "openpi_delta_action_dims": getattr(pol, "openpi_delta_action_dims", None),
            "openpi_interleave_grippers": bool(getattr(pol, "openpi_interleave_grippers", False)),
            "openpi_vector_norm": getattr(pol, "openpi_vector_norm", None),
            "ckpt_dir": _path_str(pol.ckpt_dir),
            "task_prompt": pol.task_prompt,
            "prompt_from_subtask": bool(getattr(pol, "prompt_from_subtask", False)),
            "keep_dims": list(pol.keep_dims),
            "camera_keys": None if camera_keys is None else list(camera_keys),
            "camera_keys_label": _camera_keys_label(
                None if camera_keys is None else list(camera_keys)
            ),
            "n_action_steps": n_action_steps,
            "chunk_size": chunk_size,
            "action_scale": pol.action_scale,
            "action_clip_abs": pol.action_clip_abs,
            "aloha_checkpoint_name": pol.aloha_checkpoint_name,
            "aloha_stats_path": _path_str(pol.aloha_stats_path),
            "aloha_model_args": dict(pol.aloha_model_args),
        },
        "session": {
            "fps": ses.fps,
            "task_mode": ses.task_mode,
            "tasks_per_group": tasks_per_group,
            "max_duration_sec": ses.max_duration_sec,
        },
        "cameras": {
            "enabled": camera_on,
            "transport": cam.transport,
            "enable_depth": bool(cam.enable_depth),
            "devices": _device_summaries(cfg),
        },
        "robot": {
            "direct_robot": bool(dr.enabled),
            "gripper_mode": grip_mode,
            "gripper_state_source": grip_src,
            "state_poll_hz": dr.state_poll_hz if dr.enabled else None,
            "high_host": cfg.network.high_host,
            "low_host": cfg.network.low_host,
            "gripper_control_port": cfg.network.gripper_port,
            "gripper_state_port": cfg.network.gripper_state_port,
        },
        "sync": {
            "max_time_diff": cfg.sync.max_time_diff,
            "sync_delay": cfg.sync.sync_delay,
            "prefill_sec": cfg.sync.prefill_sec,
        },
    }

    outputs: Dict[str, Any] = {
        "test_dir": _path_str(test_dir),
        "inference_dir": _path_str(cfg.resolve_inference_dir()),
        "will_write": [
            name
            for name, on in [
                ("run.log / meta / feature_manifest", True),
                ("videos/", bool(rec.enabled and rec.save_videos)),
                ("record/", bool(rec.enabled)),
                ("trajectory_data/", bool(rec.enabled and rec.save_trajectory)),
                ("eval_log.jsonl", bool(ev.eval_log_enabled)),
                ("infer_probe.jsonl", bool(ev.eval_log_enabled)),
                ("eval_plots/", bool(ev.eval_log_enabled and ev.eval_log_generate_plots)),
                ("evaluation_report.*", bool(ev.enabled and ev.save_report)),
                ("device_health*", bool(log.device_health_enabled)),
                ("machine_flow.jsonl", bool(log.machine_flow_enabled)),
                (f"{bi.vla_auto_eval.output_dirname}/", bool(bi.vla_auto_eval.enabled)),
                ("评分表.xlsx", bool(bi.feedback_score_sheet)),
            ]
            if on
        ],
    }

    return {
        "schema_version": FEATURE_MANIFEST_SCHEMA_VERSION,
        "generated_at": datetime.now().astimezone().isoformat(),
        "summary": {
            "core_count": len(core),
            "secondary_count": len(secondary),
            "gripper_mode": grip_mode,
            "gripper_state_source": grip_src,
            "camera_on": camera_on,
        },
        # 兼容旧字段名：features = 全部项
        "features": items,
        "core": core,
        "secondary": secondary,
        "inputs": inputs,
        "outputs": outputs,
        "runtime": rt,
    }


def _ckpt_loss_line(ckpt_dir: object, policy_type: object) -> str:
    """纯文本。颜色由终端 handler 加，run.log 里保持无 ANSI。"""
    try:
        from infer.recording.ckpt_loss_notice import notice_for_ckpt

        return notice_for_ckpt(ckpt_dir, str(policy_type or "")).plain
    except Exception:
        return "LOSS 未找到（读取失败，不影响推理）"


def _format_item_line(item: Dict[str, Any]) -> str:
    status = str(item.get("status", "?"))
    name = str(item.get("name", item.get("id")))
    effect = str(item.get("effect") or "").strip()
    # 状态可能较长，单独一行 effect
    line = f"  [{status}]  {name}"
    if effect:
        line += f"\n           → {effect}"
    return line


def format_feature_manifest_text(manifest: Dict[str, Any]) -> str:
    """人读块：核心在前，次要折叠成短列表。"""
    lines: List[str] = []
    sep = "=" * 72
    lines.append(sep)
    lines.append("【本次推理怎么配的 — 核心看这里】")
    lines.append(sep)

    inputs = manifest.get("inputs") or {}
    pol = inputs.get("policy") or {}
    ses = inputs.get("session") or {}
    robot = inputs.get("robot") or {}
    summary = manifest.get("summary") or {}

    lines.append(
        f"{inputs.get('experiment_name', '?')}  |  {pol.get('type', '?')}  |  "
        f"fps={ses.get('fps')}  n_action_steps={pol.get('n_action_steps')}  "
        f"tasks/组={ses.get('tasks_per_group')}"
    )
    lines.append(f"ckpt: {pol.get('ckpt_dir')}")
    lines.append(_ckpt_loss_line(pol.get("ckpt_dir"), pol.get("type")))
    lines.append(f"prompt: {pol.get('task_prompt')}")
    lines.append("")

    # 一眼看清的三行摘要
    lines.append("【一眼摘要】")
    lines.append(
        f"  夹爪控制 = {summary.get('gripper_mode')}    "
        f"夹爪观测 = {summary.get('gripper_state_source')}    "
        f"视觉 = {'开' if summary.get('camera_on') else '关'}"
    )
    lines.append(
        f"  说明: gripper_mode=怎么发指令；"
        f"gripper_state_source=模型看到的夹爪是 UDP实测 还是 buffer对齐值"
    )
    lines.append(
        f"  端口: 控制→{robot.get('gripper_control_port')}  "
        f"状态回传→{robot.get('gripper_state_port')}  "
        f"low_host={robot.get('low_host')}"
    )
    lines.append("")

    lines.append("【核心 — 会改变控制/观测】")
    for item in manifest.get("core") or []:
        lines.append(_format_item_line(item))
    lines.append("")

    lines.append("【次要 — 只影响落盘/评估，不改当场控制】")
    for item in manifest.get("secondary") or []:
        status = item.get("status", "?")
        name = item.get("name", item.get("id"))
        lines.append(f"  [{status}]  {name}")
    lines.append("")

    lines.append("【会写入的文件】")
    will = (manifest.get("outputs") or {}).get("will_write") or []
    lines.append("  " + (" | ".join(will) if will else "(无)"))
    if (manifest.get("outputs") or {}).get("test_dir"):
        lines.append(f"  test_dir: {manifest['outputs']['test_dir']}")
    lines.append(sep)
    return "\n".join(lines)


def write_feature_manifest(
    out_dir: Path,
    manifest: Dict[str, Any],
    *,
    filename: str = FEATURE_MANIFEST_FILENAME,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return path


def build_and_write_feature_manifest(
    cfg: InferConfig,
    out_dir: Path,
    *,
    runtime: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    runtime = dict(runtime or {})
    runtime.setdefault("test_dir", str(out_dir))
    manifest = build_feature_manifest(cfg, runtime=runtime)
    write_feature_manifest(out_dir, manifest)
    return manifest
