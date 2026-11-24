"""一次推理的三份记录：相机帧、推理、下发。

相机帧的 frame_index 就是 mp4 里的帧号，t_capture 是这一帧的主机时间。
录像按控制频率留下，不是相机的原始帧率。
推理行用 frame_index 指向模型输入的那一帧，同时保留 t_image / t_state / t_wall。
下发行用 tick_id 指回那一次推理，并带上这一步关节、画面和命令的时间。
"""

from __future__ import annotations

import threading
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_CAMERAS = ("cam_left", "cam_mid", "cam_third", "cam_fourth")


def _floats(value: Any) -> Optional[List[float]]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return None
    return arr.reshape(-1).tolist()


def _chunk(value: Any) -> Optional[List[List[float]]]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 1:
        return [arr.tolist()]
    if arr.ndim != 2:
        return None
    return arr.tolist()


def _task_id(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        task = int(value)
    except (TypeError, ValueError):
        return None
    if task <= 0:
        return None
    return task


class SessionRecord:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._camera_frames: List[Dict[str, Any]] = []
        self._ticks: List[Dict[str, Any]] = []
        self._steps: List[Dict[str, Any]] = []
        self._journal_path: Optional[Path] = None

    def set_journal(self, path: Path) -> None:
        """Append recovery events independently of optional machine_flow logging."""
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._journal_path = path

    def _journal(self, event: str, row: Dict[str, Any]) -> None:
        if self._journal_path is not None:
            with self._journal_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"event": event, "row": row,
                                         "record_monotonic_ns": time.monotonic_ns()},
                                        ensure_ascii=False) + "\n")

    @classmethod
    def recover_journal(cls, path: Path) -> "SessionRecord":
        """Replay appends, task deletion and frame shifts; tolerate a torn final line."""
        rec = cls()
        lines = path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                if index == len(lines) - 1:
                    break
                raise
            event, row = entry["event"], entry["row"]
            if event == "reset":
                rec.reset()
            elif event == "drop_task":
                rec.drop_task(row["task_id"])
            elif event == "shift_frame_indices":
                rec.shift_frame_indices(**row)
            else:
                target = {"camera": rec._camera_frames, "tick": rec._ticks,
                          "step": rec._steps}[event]
                target.append(row)
        return rec

    def reset(self) -> None:
        with self._lock:
            self._journal("reset", {})
            self._camera_frames.clear()
            self._ticks.clear()
            self._steps.clear()

    def drop_task(self, task_id: Any) -> None:
        """丢掉这一段的记录。重做时帧号从 0 再计，不能和丢掉的帧留在同一张表里。"""
        tid = _task_id(task_id)
        if tid is None:
            return
        with self._lock:
            self._journal("drop_task", {"task_id": tid})
            self._camera_frames = [r for r in self._camera_frames if r.get("task_id") != tid]
            self._ticks = [r for r in self._ticks if r.get("task_id") != tid]
            self._steps = [r for r in self._steps if r.get("task_id") != tid]

    def add_camera_frame(
        self,
        *,
        camera: str,
        frame_index: int,
        t_capture: float,
        task_id: Any,
    ) -> None:
        row = {
            "camera": str(camera),
            "frame_index": int(frame_index),
            "t_capture": float(t_capture),
            "task_id": _task_id(task_id),
        }
        with self._lock:
            self._journal("camera", row)
            self._camera_frames.append(row)

    def add_tick(
        self,
        *,
        tick_id: int,
        task_id: Any,
        t_wall: float,
        t_state: float,
        t_image: float,
        infer_sec: float,
        prompt: str,
        state: Any,
        raw_action: Any,
        post_action: Any,
        frame_refs: Optional[Dict[str, Dict[str, Any]]] = None,
        execute_start: Optional[int] = None,
        exec_steps: Optional[int] = None,
        t_infer_start: Optional[float] = None,
        t_infer_end: Optional[float] = None,
        infer_monotonic_start_ns: Optional[int] = None,
        infer_monotonic_end_ns: Optional[int] = None,
    ) -> None:
        row: Dict[str, Any] = {
            "tick_id": int(tick_id),
            "task_id": _task_id(task_id),
            "t_wall": float(t_wall),
            "t_state": float(t_state),
            "t_image": float(t_image),
            "infer_sec": float(infer_sec),
            "prompt": str(prompt or ""),
            "state": _floats(state),
            "raw_action": _chunk(raw_action),
            "post_action": _chunk(post_action),
            "execute_start": execute_start,
            "exec_steps": exec_steps,
            "t_infer_start": t_infer_start,
            "t_infer_end": t_infer_end,
            "infer_monotonic_start_ns": infer_monotonic_start_ns,
            "infer_monotonic_end_ns": infer_monotonic_end_ns,
        }
        refs = frame_refs or {}
        for cam in _CAMERAS:
            ref = refs.get(cam) or {}
            idx = ref.get("frame_index")
            cap = ref.get("t_capture")
            row[f"frame_index_{cam}"] = None if idx is None else int(idx)
            row[f"t_capture_{cam}"] = None if cap is None else float(cap)
        with self._lock:
            self._journal("tick", row)
            self._ticks.append(row)

    def shift_frame_indices(self, camera: str, task_id: Any, min_local: int) -> None:
        """往某段视频中间插入一帧后，把已经写下的帧号往后挪一格。"""
        tid = _task_id(task_id)
        col = f"frame_index_{camera}"
        floor = int(min_local)
        with self._lock:
            self._journal("shift_frame_indices", {"camera": camera, "task_id": tid,
                                                  "min_local": floor})
            for row in self._camera_frames:
                if row.get("camera") != camera or row.get("task_id") != tid:
                    continue
                if int(row["frame_index"]) >= floor:
                    row["frame_index"] = int(row["frame_index"]) + 1
            for row in (*self._ticks, *self._steps):
                if row.get("task_id") != tid:
                    continue
                idx = row.get(col)
                if idx is not None and int(idx) >= floor:
                    row[col] = int(idx) + 1

    def add_step(
        self,
        *,
        tick_id: int,
        task_id: Any,
        step_index: int,
        t_send: float,
        t_state_pre: Optional[float],
        t_state_post: Optional[float],
        state_pre: Any,
        action_cmd: Any,
        state_post: Any,
        t_model_in: Optional[float] = None,
        t_model_state: Optional[float] = None,
        t_model_out: Optional[float] = None,
        frame_refs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        row = {
            "tick_id": int(tick_id),
            "task_id": _task_id(task_id),
            "step_index": int(step_index),
            "t_send": float(t_send),
            "t_state_pre": None if t_state_pre is None else float(t_state_pre),
            "t_state_post": None if t_state_post is None else float(t_state_post),
            "state_pre": _floats(state_pre),
            "action_cmd": _floats(action_cmd),
            "state_post": _floats(state_post),
            "t_model_in": None if t_model_in is None else float(t_model_in),
            "t_model_state": None if t_model_state is None else float(t_model_state),
            "t_model_out": None if t_model_out is None else float(t_model_out),
            "feedback_fresh": None if t_state_post is None else bool(
                t_state_post > t_send and (t_state_pre is None or t_state_post > t_state_pre)),
            "feedback_minus_send_ms": None if t_state_post is None else
                1000.0 * (float(t_state_post) - float(t_send)),
        }
        refs = frame_refs or {}
        for cam in _CAMERAS:
            ref = refs.get(cam) or {}
            idx = ref.get("frame_index")
            cap = ref.get("t_capture")
            row[f"frame_index_{cam}"] = None if idx is None else int(idx)
            row[f"t_capture_{cam}"] = None if cap is None else float(cap)
        with self._lock:
            self._journal("step", row)
            self._steps.append(row)

    def flush(self, directory: Path) -> Dict[str, Any]:
        import pandas as pd

        directory.mkdir(parents=True, exist_ok=True)
        with self._lock:
            camera_rows = list(self._camera_frames)
            tick_rows = list(self._ticks)
            step_rows = list(self._steps)
        written = {
            "camera_frames": _write_table(pd, directory / "camera_frames.parquet", camera_rows, _camera_frame()),
            "ticks": _write_table(pd, directory / "ticks.parquet", tick_rows, _tick_row()),
            "steps": _write_table(pd, directory / "steps.parquet", step_rows, _step_row()),
        }
        return {
            "dir": str(directory),
            "camera_frames": len(camera_rows),
            "ticks": len(tick_rows),
            "steps": len(step_rows),
            "paths": written,
        }


def _camera_frame() -> Dict[str, Any]:
    return {"camera": "", "frame_index": 0, "t_capture": 0.0, "task_id": None}


def _tick_row() -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "tick_id": 0,
        "task_id": None,
        "t_wall": 0.0,
        "t_state": 0.0,
        "t_image": 0.0,
        "infer_sec": 0.0,
        "prompt": "",
        "state": None,
        "raw_action": None,
        "post_action": None,
        "execute_start": None,
        "exec_steps": None,
        "t_infer_start": None,
        "t_infer_end": None,
        "infer_monotonic_start_ns": None,
        "infer_monotonic_end_ns": None,
    }
    for cam in _CAMERAS:
        row[f"frame_index_{cam}"] = None
        row[f"t_capture_{cam}"] = None
    return row


def _step_row() -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "tick_id": 0,
        "task_id": None,
        "step_index": 0,
        "t_send": 0.0,
        "t_state_pre": None,
        "t_state_post": None,
        "state_pre": None,
        "action_cmd": None,
        "state_post": None,
        "t_model_in": None,
        "t_model_state": None,
        "t_model_out": None,
        "feedback_fresh": None,
        "feedback_minus_send_ms": None,
    }
    for cam in _CAMERAS:
        row[f"frame_index_{cam}"] = None
        row[f"t_capture_{cam}"] = None
    return row


def _write_table(pd: Any, path: Path, rows: List[Dict[str, Any]], template: Dict[str, Any]) -> str:
    columns = list(template.keys())
    frame = pd.DataFrame(rows, columns=columns)
    for col in ("task_id", "frame_index"):
        if col in frame.columns:
            frame[col] = frame[col].astype("Int64")
    for cam in _CAMERAS:
        col = f"frame_index_{cam}"
        if col in frame.columns:
            frame[col] = frame[col].astype("Int64")
    temporary = path.with_suffix(".partial.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)
    return str(path)
