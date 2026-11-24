"""自动生成 experiments/index/ 实验条目 YAML。

每次推理 run 结束后，从 test_dir 内的产物自动提取关键信息，
在 experiments/index/ 目录下写入一份精简 YAML，供后续检索和对照。

YAML 格式:
  run_id: 20260909_143022_fastwam_00073
  model: fastwam
  task: "00073"
  dataset: "00073"
  experiment_name: fastwam_tongbot_16d_00073
  git:
    infer_commit: "abc1234"
    infer_branch: "main"
    infer_dirty: false
  local:
    test_dir: /path/to/test
    ckpt_dir: /path/to/ckpt
    pipeline_history: /path/to/pipeline_record_history.jsonl
  config:
    fps: 15
    n_action_steps: 10
    action_space: null
    control: null
    guidance_scale: null
    action_guidance_scale: null
    attn_window: null
    num_inference_steps: null
  checkpoint:
    step: 25000
    weight_path: /path/to/weights.pt
  training:
    dataset_path: /path/to/dataset
    learning_rate: null
    batch_size: null
    num_steps: null
  metrics:
    duration_sec: 120.5
    stop_reason: user_stop
  server:
    requests_log: /path/to/fastwam_requests.jsonl
  notes: ""
  created_at: "2026-09-09T14:30:22+08:00"
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

EXPERIMENT_INDEX_DIR_NAME = "experiments/index"
EXPERIMENT_INDEX_FILENAME_PREFIX = "exp_"


def _safe_read_json(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _extract_run_id(test_dir: Path, pipeline_record: Optional[Dict[str, Any]]) -> str:
    """生成 run_id: YYYYMMDD_HHMMSS_<model>_<task>"""
    ident = (pipeline_record or {}).get("identity") or {}
    started_at = ident.get("started_at") or datetime.now().astimezone().isoformat()
    try:
        dt = datetime.fromisoformat(started_at)
        ts = dt.strftime("%Y%m%d_%H%M%S")
    except Exception:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model = ident.get("policy_type") or test_dir.parent.name or "unknown"
    # 从 experiment_name 中提取数据集标识
    exp_name = ident.get("experiment_name") or test_dir.name
    # 尝试提取末尾的数字部分作为 task
    parts = exp_name.rsplit("_", 1)
    task = parts[-1] if parts and parts[-1].isdigit() else exp_name
    return f"{ts}_{model}_{task}"


def _extract_model_name(pipeline_record: Optional[Dict[str, Any]], test_dir: Path) -> str:
    ident = (pipeline_record or {}).get("identity") or {}
    return ident.get("policy_type") or "unknown"


def _extract_task_id(pipeline_record: Optional[Dict[str, Any]]) -> str:
    ds = (pipeline_record or {}).get("dataset") or {}
    root = ds.get("dataset_root") or ""
    if root:
        # 从路径提取: /home/xk/dataset/00073_xxx → 00073
        name = Path(root).name
        parts = name.split("_", 1)
        if parts and parts[0].isdigit():
            return parts[0]
    # 退而求其次：从 experiment_name 提取
    ident = (pipeline_record or {}).get("identity") or {}
    exp_name = ident.get("experiment_name") or ""
    parts = exp_name.rsplit("_", 1)
    if parts and parts[-1].isdigit():
        return parts[-1]
    return ""


def _find_server_requests_log(test_dir: Path) -> Optional[str]:
    """尝试找到服务端请求级日志（在 recipes/<model>/ 下）。"""
    pipeline_path = test_dir / "pipeline_record.json"
    record = _safe_read_json(pipeline_path)
    if not record:
        return None
    ident = record.get("identity") or {}
    model = ident.get("policy_type") or ""
    # 尝试在常见位置查找
    candidates = [
        test_dir / f"{model}_requests.jsonl",
        test_dir.parent / f"{model}_requests.jsonl",
    ]
    # 也尝试 recipes/ 目录
    try:
        recipes_dir = test_dir.resolve().parents[3] / "recipes" / model
        if recipes_dir.is_dir():
            candidates.append(recipes_dir / f"{model}_requests.jsonl")
    except (IndexError, OSError):
        pass
    for c in candidates:
        if c.is_file():
            return str(c)
    return None


def _timing_entry(timing_stats: Dict[str, Any]) -> Dict[str, Any]:
    """把 inference_summary 的 timing_stats 压成索引里的一段精简 timing 摘要。"""
    out: Dict[str, Any] = {
        "frame_count": timing_stats.get("frame_count"),
        "effective_fps": timing_stats.get("effective_fps"),
        "infer_p50_ms": (timing_stats.get("infer_ms") or {}).get("p50_ms"),
        "infer_p95_ms": (timing_stats.get("infer_ms") or {}).get("p95_ms"),
        "model_p50_ms": (timing_stats.get("model_ms") or {}).get("p50_ms"),
        "model_p95_ms": (timing_stats.get("model_ms") or {}).get("p95_ms"),
        "preprocess_pct": timing_stats.get("preprocess_pct"),
        "model_pct": timing_stats.get("model_pct"),
        "postprocess_pct": timing_stats.get("postprocess_pct"),
    }
    # 服务端分段耗时 p50：只有回传 server_timing 的模型（WALL-OSS 等）才有，
    # 缺键时整块不写，避免给其它模型的索引条目加一堆 null。
    stage = {
        label: (timing_stats.get(key) or {}).get("p50_ms")
        for label, key in (
            ("fk", "fk_ms"),
            ("flow", "flow_ms"),
            ("convert", "convert_ms"),
            ("ik", "ik_ms"),
            ("server_total", "server_infer_ms"),
            ("network", "network_ms"),
        )
        if timing_stats.get(key)
    }
    if stage:
        out["stage_p50_ms"] = stage
    return out


def _human_eval_entry(human_eval: Dict[str, Any]) -> Dict[str, Any]:
    """把 inference_summary 的 human_eval 压成索引摘要（逐条明细留在 marks 文件）。"""
    counts = human_eval.get("verdict_counts") or {}
    out: Dict[str, Any] = {
        # asked>0 但 answered_valid=0 时 rate 为 null：是"没评"，不是"0% 成功"
        "asked": human_eval.get("asked"),
        "answered_valid": human_eval.get("answered_valid"),
        "success_rate": human_eval.get("success_rate"),
        "partial_inclusive_success_rate": human_eval.get(
            "partial_inclusive_success_rate"
        ),
        "success": counts.get("success", 0),
        "partial": counts.get("partial", 0),
        "fail": counts.get("fail", 0),
        "no_answer": human_eval.get("no_answer"),
        "described": human_eval.get("described"),
        "unparsed": human_eval.get("unparsed"),
    }
    # 判为失败/部分完成的推理帧号，便于直接跳回 eval_log 对应行复核
    bad_frames = [
        mark.get("inference_frame_index")
        for mark in human_eval.get("marks") or []
        if mark.get("verdict") in ("fail", "partial")
        and isinstance(mark.get("inference_frame_index"), int)
    ]
    if bad_frames:
        out["bad_frames"] = bad_frames[:100]
    return out


def build_experiment_index_entry(
    test_dir: Path,
    *,
    server_requests_log: Optional[str] = None,
    notes: str = "",
) -> Dict[str, Any]:
    """从 test_dir 产物构建 experiment index 条目。"""
    pipeline_path = test_dir / "pipeline_record.json"
    pipeline = _safe_read_json(pipeline_path)
    summary_path = test_dir / "inference_summary.json"
    summary = _safe_read_json(summary_path)
    eval_report_path = test_dir / "evaluation_report.json"
    eval_report = _safe_read_json(eval_report_path)

    ident = (pipeline or {}).get("identity") or {}
    ckpt = (pipeline or {}).get("checkpoint") or {}
    training = (pipeline or {}).get("training") or {}
    infer = (pipeline or {}).get("inference") or {}
    server = infer.get("server") or {}
    client = infer.get("client") or {}
    code = (pipeline or {}).get("code") or {}

    run_id = _extract_run_id(test_dir, pipeline)
    model = _extract_model_name(pipeline, test_dir)
    task = _extract_task_id(pipeline)

    entry: Dict[str, Any] = {
        "run_id": run_id,
        "model": model,
        "task": task,
        "dataset": task,
        "experiment_name": ident.get("experiment_name") or "",
    }

    # Git 信息
    entry["git"] = {
        "infer_commit": code.get("infer_git_commit_short") or code.get("infer_git_commit"),
        "infer_branch": code.get("infer_git_branch"),
        "infer_dirty": code.get("infer_git_dirty"),
    }

    # 本地路径
    history_candidates = [
        test_dir.parent / "pipeline_record_history.jsonl",
        test_dir / "pipeline_record_history.jsonl",
    ]
    history_path = None
    for h in history_candidates:
        if h.is_file():
            history_path = str(h)
            break
    entry["local"] = {
        "test_dir": str(test_dir),
        "ckpt_dir": ckpt.get("ckpt_dir"),
        "pipeline_history": history_path,
    }

    # 推理配置（关键可调参数）
    entry["config"] = {
        "fps": client.get("fps"),
        "n_action_steps": client.get("n_action_steps"),
        "action_space": client.get("action_space"),
        "control": client.get("control"),
        "guidance_scale": server.get("guidance_scale"),
        "action_guidance_scale": server.get("action_guidance_scale"),
        "attn_window": server.get("attn_window"),
        "num_inference_steps": server.get("num_inference_steps"),
        "action_num_inference_steps": server.get("action_num_inference_steps"),
        "snr_shift": server.get("snr_shift"),
        "action_snr_shift": server.get("action_snr_shift"),
        "video_exec_step": server.get("video_exec_step"),
        "task_prompt": client.get("task_prompt"),
        "rtc_enabled": client.get("rtc_enabled"),
        "action_scale": client.get("action_scale"),
        "action_clip_abs": client.get("action_clip_abs"),
        "use_action_chunk": client.get("use_action_chunk"),
    }

    # Checkpoint
    entry["checkpoint"] = {
        "step": ckpt.get("checkpoint_step"),
        "weight_path": ckpt.get("weight_path"),
    }

    # 训练配置
    train_h = training.get("hparams") or {}
    entry["training"] = {
        "dataset_path": training.get("dataset_path"),
        "learning_rate": train_h.get("learning_rate"),
        "batch_size": train_h.get("batch_size"),
        "num_steps": train_h.get("num_steps"),
        "cfg_prob": train_h.get("cfg_prob"),
        "train_guidance_scale": train_h.get("guidance_scale"),
    }

    # 训练-推理 diff
    diff = infer.get("train_infer_diff") or {}
    if diff:
        entry["train_infer_diff"] = {
            "changed_keys": sorted(diff.keys()),
            "details": diff,
        }

    # 运行摘要
    entry["metrics"] = {
        "duration_sec": (summary or {}).get("duration_sec"),
        "stop_reason": (summary or {}).get("stop_reason"),
    }

    # 推理耗时分解统计（从 inference_summary.json 中的 timing_stats）
    timing_stats = (summary or {}).get("timing_stats")
    if timing_stats:
        entry["metrics"]["timing"] = _timing_entry(timing_stats)

    # 系统资源（从 inference_summary.json）
    sys_info = (summary or {}).get("system")
    if sys_info:
        entry["metrics"]["system"] = sys_info

    # 人工评测（Ctrl+. 追问）：asked / 有效判定数 / 成功率（未评测为 null）
    human_eval = (summary or {}).get("human_eval")
    if human_eval:
        entry["metrics"]["human_eval"] = _human_eval_entry(human_eval)

    # 自动指标摘要（如果 evaluation_report.json 存在）
    if eval_report:
        metrics = eval_report.get("metrics") or {}
        auto_metrics: Dict[str, Any] = {}
        for key in [
            "runtime.sync_warning_rate",
            "runtime.infer_latency_p95_sec",
            "runtime.tracking_error_mean",
            "quality.action_smoothness",
            "safety.collision_spike_score",
            "efficiency.episode_duration_sec",
        ]:
            if key in metrics:
                md = metrics[key]
                auto_metrics[key] = md.get("value") if isinstance(md, dict) else md
        if auto_metrics:
            entry["metrics"]["auto"] = auto_metrics
        for score_key in ["runtime_composite_score", "vla_composite_score"]:
            v = eval_report.get(score_key)
            if v is not None:
                entry["metrics"][score_key] = v

    # 设备健康摘要
    health_summary = _safe_read_json(test_dir / "device_health_summary.json")
    if health_summary:
        entry["metrics"]["device_health"] = {
            "overall_status": health_summary.get("overall_status"),
            "issue_count": health_summary.get("issue_count", 0),
        }

    # 服务端日志 + 服务端统计摘要
    srv_log = server_requests_log or _find_server_requests_log(test_dir)
    if srv_log:
        srv_entry: Dict[str, Any] = {"requests_log": srv_log}
        # 尝试从服务端日志提取统计摘要（session_end 事件）
        srv_stats = _extract_server_stats(srv_log)
        if srv_stats:
            srv_entry["stats"] = srv_stats
        entry["server"] = srv_entry

    # 可复现性：配置 hash（用于快速判断两次实验配置是否相同）
    config_hash = _config_hash(entry.get("config") or {}, entry.get("checkpoint") or {})
    if config_hash:
        entry["config_hash"] = config_hash

    entry["notes"] = notes
    entry["created_at"] = datetime.now().astimezone().isoformat()

    return entry


def write_experiment_index_yaml(
    test_dir: Path,
    *,
    index_dir: Optional[Path] = None,
    server_requests_log: Optional[str] = None,
    notes: str = "",
    overwrite: bool = False,
) -> Optional[Path]:
    """生成并写入 experiment index YAML。

    Args:
        test_dir: 推理产物目录
        index_dir: experiments/index/ 目录（默认自动查找）
        server_requests_log: 服务端请求日志路径
        notes: 人工备注
        overwrite: 同名文件是否覆盖

    Returns:
        写入的 YAML 路径，失败返回 None
    """
    try:
        entry = build_experiment_index_entry(
            test_dir,
            server_requests_log=server_requests_log,
            notes=notes,
        )
    except Exception:
        logger.exception("构建 experiment index 条目失败")
        return None

    # 确定 index 目录
    if index_dir is None:
        # 从 test_dir 向上查找 experiments/index/
        try:
            repo_root = test_dir.resolve().parents[3]
            candidate = repo_root / EXPERIMENT_INDEX_DIR_NAME
            if candidate.parent.is_dir():
                index_dir = candidate
        except (IndexError, OSError):
            pass
    if index_dir is None:
        # 退到当前目录的 experiments/index/
        index_dir = Path.cwd() / EXPERIMENT_INDEX_DIR_NAME

    index_dir.mkdir(parents=True, exist_ok=True)

    run_id = entry.get("run_id", "unknown")
    filename = f"{EXPERIMENT_INDEX_FILENAME_PREFIX}{run_id}.yaml"
    yaml_path = index_dir / filename

    if yaml_path.is_file() and not overwrite:
        # 追加时间戳避免覆盖
        ts = datetime.now().strftime("%H%M%S")
        filename = f"{EXPERIMENT_INDEX_FILENAME_PREFIX}{run_id}_{ts}.yaml"
        yaml_path = index_dir / filename

    try:
        yaml_text = _dict_to_yaml(entry)
        yaml_path.write_text(yaml_text, encoding="utf-8")
        logger.info("实验条目已写入: %s", yaml_path)
        return yaml_path
    except Exception:
        logger.exception("写入 experiment index YAML 失败")
        return None


def _dict_to_yaml(data: Any, indent: int = 0) -> str:
    """简易 dict→YAML 序列化（不依赖 pyyaml，避免跨环境依赖）。"""
    lines: list[str] = []
    prefix = "  " * indent

    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{prefix}{key}:")
                lines.append(_dict_to_yaml(value, indent + 1))
            elif isinstance(value, list):
                if all(isinstance(v, (str, int, float, bool, type(None))) for v in value):
                    formatted = [_yaml_scalar(v) for v in value]
                    lines.append(f"{prefix}{key}: [{', '.join(formatted)}]")
                else:
                    lines.append(f"{prefix}{key}:")
                    for item in value:
                        lines.append(f"{prefix}- {_yaml_scalar(item)}")
            elif isinstance(value, str) and "\n" in value:
                lines.append(f"{prefix}{key}: |")
                for line in value.splitlines():
                    lines.append(f"{prefix}  {line}")
            else:
                lines.append(f"{prefix}{key}: {_yaml_scalar(value)}")
    return "\n".join(lines)


def _yaml_scalar(value: Any) -> str:
    """将 Python 值转为 YAML 标量字符串。"""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    s = str(value)
    # 需要引号的情况
    if not s or s.startswith(("{", "[", "'", '"')) or ":" in s or "#" in s:
        escaped = s.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return s


def _extract_server_stats(srv_log_path: str) -> Optional[Dict[str, Any]]:
    """从服务端 JSONL 日志的 session_end 事件提取统计摘要。"""
    try:
        path = Path(srv_log_path)
        if not path.is_file():
            return None
        # 从末尾向前找 session_end（通常最后几行）
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        for line in reversed(lines[-20:]):
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if row.get("type") == "server_session_end":
                stats: Dict[str, Any] = {}
                for key in ("total_requests", "error_count", "duration_sec",
                            "effective_fps", "infer_ms_stats"):
                    if key in row:
                        stats[key] = row[key]
                if row.get("last_gpu"):
                    stats["last_gpu"] = row["last_gpu"]
                return stats if stats else None
    except Exception:
        pass
    return None


def _config_hash(config: Dict[str, Any], checkpoint: Dict[str, Any]) -> Optional[str]:
    """计算配置 hash，用于快速对比两次实验的参数是否相同。"""
    try:
        # 只取关键配置字段
        payload = {
            "config": {k: v for k, v in sorted(config.items()) if v is not None},
            "ckpt_step": checkpoint.get("step"),
        }
        text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(text.encode()).hexdigest()[:12]
    except Exception:
        return None
