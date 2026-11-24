"""服务端推理请求级结构化日志。

所有 serve_*.py 共享此模块，为每个 WebSocket 推理请求写 JSONL 行。

产物: server_requests.jsonl（与 serve_*.log 同目录，或 --log-dir 指定）

每行 schema:
  {
    "type": "infer_request",
    "request_id": 42,
    "client": "127.0.0.1:52384",
    "wall_time": 1725900000.123,
    "elapsed_sec": 12.345,
    "infer_ms": 87.2,
    "timing": {
      "preprocess_ms": 5.1,
      "model_ms": 75.3,
      "postprocess_ms": 6.8,
      "network_send_ms": 1.2,
    },
    "input": {
      "obs_keys": ["observation.images.front", ...],
      "state_shape": [16],
      "state_range": [-1.23, 2.45],
      "has_reset": false,
      "has_compute_kv_cache": false,
    },
    "output": {
      "action_shape": [32, 16],
      "action_range": [-2.1, 1.8],
      "action_mean": 0.012,
      "action_std": 0.34,
      "action_magnitude": 1.23,
    },
    "gpu": {
      "mem_used_mb": 12345,
      "mem_total_mb": 24576,
      "utilization_pct": 95,
      "temperature_c": 62,
    },
    "system": {
      "cpu_pct": 45.2,
      "ram_used_gb": 24.5,
      "ram_total_gb": 64.0,
    }
  }
"""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _action_stats(action: Any) -> Dict[str, Any]:
    """从 action ndarray 提取摘要（不存原始数据）。"""
    try:
        arr = np.asarray(action)
    except Exception:
        return {"shape": None, "error": "not_array"}
    if arr.size == 0:
        return {"shape": list(arr.shape), "empty": True}
    flat = arr.reshape(-1).astype(np.float64)
    stats: Dict[str, Any] = {
        "shape": list(arr.shape),
        "range": [round(float(flat.min()), 4), round(float(flat.max()), 4)],
        "mean": round(float(flat.mean()), 6),
        "std": round(float(flat.std()), 6),
        "magnitude": round(float(np.linalg.norm(flat)), 4),
    }
    # 关节限位距离（取归一化后绝对值最大的维度）
    abs_max = float(np.abs(flat).max())
    if abs_max > 0:
        stats["max_abs_dim"] = int(np.argmax(np.abs(flat.reshape(-1))))
    return stats


def _obs_summary(obs: Any) -> Dict[str, Any]:
    """从 obs 提取 keys 和 shape（不存图像数据）。"""
    if not isinstance(obs, (list, tuple)):
        return {"type": type(obs).__name__, "count": 0}
    result: Dict[str, Any] = {"count": len(obs)}
    if obs and isinstance(obs[0], dict):
        last = obs[-1]
        result["obs_keys"] = sorted(str(k) for k in last.keys())
        result["per_camera_shape"] = {}
        for k, v in last.items():
            if isinstance(v, np.ndarray):
                result["per_camera_shape"][str(k)] = list(v.shape)
    return result


def _state_summary(state: Any) -> Dict[str, Any]:
    """从 state 向量提取摘要。"""
    try:
        arr = np.asarray(state, dtype=np.float64).reshape(-1)
    except Exception:
        return {"shape": None}
    if arr.size == 0:
        return {"shape": [0]}
    return {
        "shape": [int(arr.size)],
        "range": [round(float(arr.min()), 4), round(float(arr.max()), 4)],
        "mean": round(float(arr.mean()), 6),
    }


def _gpu_snapshot(device_index: int = 0) -> Optional[Dict[str, Any]]:
    """轻量级 GPU 状态快照。"""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
                f"--id={device_index}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=1.0,
            check=False,
        )
        if result.returncode != 0:
            return None
        parts = [p.strip() for p in result.stdout.strip().split(",")]
        if len(parts) >= 4:
            snap: Dict[str, Any] = {
                "mem_used_mb": int(float(parts[0])),
                "mem_total_mb": int(float(parts[1])),
                "utilization_pct": int(float(parts[2])),
                "temperature_c": int(float(parts[3])),
            }
            if len(parts) >= 5:
                snap["power_w"] = round(float(parts[4]), 1)
            return snap
    except Exception:
        pass
    return None


def _system_snapshot() -> Optional[Dict[str, Any]]:
    """轻量级系统资源快照（CPU + 内存，不依赖 psutil）。"""
    try:
        snap: Dict[str, Any] = {}
        # CPU 使用率（/proc/stat 1 秒间隔太慢，用 loadavg 替代）
        loadavg_path = Path("/proc/loadavg")
        if loadavg_path.is_file():
            parts = loadavg_path.read_text().split()
            if len(parts) >= 3:
                snap["load_avg_1m"] = float(parts[0])
                snap["load_avg_5m"] = float(parts[1])
        # 内存使用
        meminfo_path = Path("/proc/meminfo")
        if meminfo_path.is_file():
            meminfo: Dict[str, int] = {}
            for line in meminfo_path.read_text().splitlines()[:10]:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        meminfo[parts[0].rstrip(":")] = int(parts[1])
                    except ValueError:
                        pass
            total_kb = meminfo.get("MemTotal", 0)
            avail_kb = meminfo.get("MemAvailable", 0)
            if total_kb > 0:
                snap["ram_used_gb"] = round((total_kb - avail_kb) / (1024 * 1024), 1)
                snap["ram_total_gb"] = round(total_kb / (1024 * 1024), 1)
                snap["ram_used_pct"] = round((total_kb - avail_kb) / total_kb * 100, 1)
        return snap if snap else None
    except Exception:
        return None


class ServerRequestLogger:
    """异步安全的服务端请求级日志器。

    用法::

        srv_logger = ServerRequestLogger(log_path, gpu_device=0)
        srv_logger.start()  # 启动后台 GPU 采样线程

        # 每个请求:
        srv_logger.log_request(
            request_id=42,
            client_addr="127.0.0.1:52384",
            infer_ms=87.2,
            request_data={"obs": [...], "state": [...]},
            response_data={"action": np.array(...)},
        )

        srv_logger.close()
    """

    def __init__(
        self,
        log_path: Path,
        *,
        gpu_device: int = 0,
        gpu_sample_interval_sec: float = 5.0,
        enabled: bool = True,
        model_name: str = "unknown",
    ) -> None:
        self.enabled = enabled
        self.log_path = Path(log_path)
        self._gpu_device = gpu_device
        self._gpu_interval = gpu_sample_interval_sec
        self._model_name = model_name
        self._fp = None
        self._lock = threading.Lock()
        self._t0 = time.time()
        self._gpu_thread: Optional[threading.Thread] = None
        self._gpu_stop = threading.Event()
        self._latest_gpu: Optional[Dict[str, Any]] = None
        self._latest_sys: Optional[Dict[str, Any]] = None
        self._request_count = 0
        self._error_count = 0
        # 运行统计（用于 session 结束时的摘要）
        self._infer_ms_list: List[float] = []  # 保留用于计算分位数

        if self.enabled:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._fp = self.log_path.open("a", encoding="utf-8")
            # 写入 session header
            sys_snap = _system_snapshot()
            self._write_row({
                "type": "server_session_start",
                "wall_time": time.time(),
                "log_path": str(self.log_path),
                "gpu_device": gpu_device,
                "model_name": model_name,
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "system": sys_snap,
            })

    def start(self) -> None:
        """启动后台 GPU 采样线程。"""
        if not self.enabled or self._gpu_thread is not None:
            return
        self._gpu_thread = threading.Thread(
            target=self._gpu_loop,
            name="srv-logger-gpu",
            daemon=True,
        )
        self._gpu_thread.start()

    def _gpu_loop(self) -> None:
        while not self._gpu_stop.is_set():
            snap = _gpu_snapshot(self._gpu_device)
            if snap is not None:
                with self._lock:
                    self._latest_gpu = snap
            sys_snap = _system_snapshot()
            if sys_snap is not None:
                with self._lock:
                    self._latest_sys = sys_snap
            self._gpu_stop.wait(self._gpu_interval)

    def log_request(
        self,
        *,
        request_id: Optional[int] = None,
        client_addr: Optional[str] = None,
        infer_ms: float,
        request_data: Any = None,
        response_data: Any = None,
        error: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
        timing: Optional[Dict[str, float]] = None,
    ) -> None:
        """记录一个推理请求。

        Args:
            timing: 耗时分解（毫秒），例如
                {"preprocess_ms": 5.1, "model_ms": 75.3, "postprocess_ms": 6.8}
        """
        if not self.enabled or self._fp is None:
            return

        with self._lock:
            self._request_count += 1
            rid = request_id if request_id is not None else self._request_count
            if error:
                self._error_count += 1
            # 记录 infer_ms 用于统计
            self._infer_ms_list.append(infer_ms)

        # 提取输入摘要
        input_info: Dict[str, Any] = {}
        if isinstance(request_data, dict):
            is_reset = bool(request_data.get("reset"))
            is_kv = bool(request_data.get("compute_kv_cache"))
            obs = request_data.get("obs")
            state = request_data.get("state")
            input_info = {
                "has_reset": is_reset,
                "has_compute_kv_cache": is_kv,
            }
            if obs is not None:
                input_info["obs"] = _obs_summary(obs)
            if state is not None:
                input_info["state"] = _state_summary(state)

        # 提取输出摘要
        output_info: Dict[str, Any] = {}
        if isinstance(response_data, dict):
            action = response_data.get("action")
            if action is not None:
                output_info["action"] = _action_stats(action)
            server_timing = response_data.get("server_timing")
            if isinstance(server_timing, dict):
                output_info["server_timing"] = server_timing

        # GPU + 系统快照
        with self._lock:
            gpu = dict(self._latest_gpu) if self._latest_gpu else None
            sys_snap = dict(self._latest_sys) if self._latest_sys else None

        row: Dict[str, Any] = {
            "type": "infer_request",
            "request_id": rid,
            "client": client_addr,
            "wall_time": time.time(),
            "elapsed_sec": round(time.time() - self._t0, 3),
            "infer_ms": round(infer_ms, 2),
        }
        if timing:
            row["timing"] = {k: round(v, 2) for k, v in timing.items()}
        row["input"] = input_info
        row["output"] = output_info
        if gpu:
            row["gpu"] = gpu
        if sys_snap:
            row["system"] = sys_snap
        if error:
            row["error"] = error
        if extra:
            row.update(extra)

        self._write_row(row)

    def log_connection(
        self,
        *,
        client_addr: Optional[str],
        event: str = "connect",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """记录连接事件。"""
        if not self.enabled or self._fp is None:
            return
        row: Dict[str, Any] = {
            "type": f"connection_{event}",
            "client": client_addr,
            "wall_time": time.time(),
            "elapsed_sec": round(time.time() - self._t0, 3),
        }
        if metadata:
            row["metadata"] = metadata
        self._write_row(row)

    def log_model_loaded(
        self,
        *,
        model_name: str,
        ckpt_path: str,
        load_time_sec: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """记录模型加载事件。"""
        if not self.enabled or self._fp is None:
            return
        row: Dict[str, Any] = {
            "type": "model_loaded",
            "wall_time": time.time(),
            "model": model_name,
            "ckpt_path": ckpt_path,
            "load_time_sec": round(load_time_sec, 2),
        }
        if extra:
            row.update(extra)
        gpu = _gpu_snapshot(self._gpu_device)
        if gpu:
            row["gpu_after_load"] = gpu
        self._write_row(row)

    def get_stats_summary(self) -> Dict[str, Any]:
        """返回累积统计（用于 server session 结束时打印）。

        包含 infer_ms 的 p50/p95/p99/max 分位数。
        """
        summary: Dict[str, Any] = {
            "total_requests": self._request_count,
            "error_count": self._error_count,
            "duration_sec": round(time.time() - self._t0, 1),
            "log_path": str(self.log_path),
            "model_name": self._model_name,
        }
        if self._infer_ms_list:
            arr = np.array(self._infer_ms_list)
            summary["infer_ms_stats"] = {
                "mean": round(float(arr.mean()), 2),
                "p50": round(float(np.percentile(arr, 50)), 2),
                "p95": round(float(np.percentile(arr, 95)), 2),
                "p99": round(float(np.percentile(arr, 99)), 2),
                "max": round(float(arr.max()), 2),
                "min": round(float(arr.min()), 2),
            }
            # 有效 FPS（1000ms / mean_infer_ms）
            mean_ms = float(arr.mean())
            if mean_ms > 0:
                summary["effective_fps"] = round(1000.0 / mean_ms, 1)
        # 最后一个 GPU / 系统快照
        with self._lock:
            if self._latest_gpu:
                summary["last_gpu"] = self._latest_gpu
            if self._latest_sys:
                summary["last_system"] = self._latest_sys
        return summary

    def _write_row(self, row: Dict[str, Any]) -> None:
        if self._fp is None:
            return
        with self._lock:
            try:
                self._fp.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                self._fp.flush()
            except Exception:
                logger.exception("写入 server request log 失败")

    def close(self) -> Dict[str, Any]:
        """关闭日志器，返回统计摘要。"""
        self._gpu_stop.set()
        if self._gpu_thread is not None:
            self._gpu_thread.join(timeout=2.0)
            self._gpu_thread = None
        summary = self.get_stats_summary()
        if self.enabled and self._fp is not None:
            self._write_row({
                "type": "server_session_end",
                "wall_time": time.time(),
                **summary,
            })
            self._fp.close()
            self._fp = None
        return summary


def resolve_server_log_path(
    log_file_env: Optional[str] = None,
    *,
    default_dir: Optional[Path] = None,
    model_name: str = "server",
) -> Path:
    """解析 server_requests.jsonl 的路径。

    优先级:
    1. 环境变量 log_file_env
    2. default_dir / f"{model_name}_requests.jsonl"
    3. 当前目录 / f"{model_name}_requests.jsonl"
    """
    import os

    if log_file_env:
        env_val = os.environ.get(log_file_env)
        if env_val:
            return Path(env_val)
    if default_dir:
        return default_dir / f"{model_name}_requests.jsonl"
    return Path(f"{model_name}_requests.jsonl")
