"""推理产物保存：视频 / 轨迹。"""

from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from infer.recording.depth_video import depth_u16_to_video_rgb, save_depth_u16_sequence

logger = logging.getLogger(__name__)

CAMERA_NAME_MAP = {
    "cam_left": "cam_left_wrist",
    "cam_mid": "cam_mid_chest",
    "cam_third": "cam_third_view",
    "cam_fourth": "cam_fourth_view",
}

DEPTH_CAMERA_NAME_MAP = {
    "cam_left_depth": "cam_left_wrist_depth",
    "cam_mid_depth": "cam_mid_chest_depth",
}


def empty_recorded_data() -> Dict[str, list]:
    return {
        "cam_left": [],
        "cam_left_timestamp": [],
        "cam_left_state_timestamp": [],
        "cam_left_frame_index": [],
        "cam_left_action_step_index": [],
        "cam_left_record_timestamp": [],
        "cam_mid": [],
        "cam_mid_timestamp": [],
        "cam_mid_state_timestamp": [],
        "cam_mid_frame_index": [],
        "cam_mid_action_step_index": [],
        "cam_mid_record_timestamp": [],
        "cam_left_depth": [],
        "cam_left_depth_timestamp": [],
        "cam_left_depth_state_timestamp": [],
        "cam_left_depth_frame_index": [],
        "cam_left_depth_action_step_index": [],
        "cam_left_depth_record_timestamp": [],
        "cam_mid_depth": [],
        "cam_mid_depth_timestamp": [],
        "cam_mid_depth_state_timestamp": [],
        "cam_mid_depth_frame_index": [],
        "cam_mid_depth_action_step_index": [],
        "cam_mid_depth_record_timestamp": [],
        "cam_third": [],
        "cam_third_timestamp": [],
        "cam_fourth": [],
        "cam_fourth_timestamp": [],
        "state": [],
        "action": [],
        "timestamp": [],
        "state_timestamp": [],
        "image_timestamp": [],
        "frame_index": [],
        "action_step_index": [],
        "task_segment_index": [],
    }


def estimate_video_fps(timestamps: List[float], default_fps: float) -> float:
    if len(timestamps) < 2:
        return float(default_fps)
    diffs = [t2 - t1 for t1, t2 in zip(timestamps[:-1], timestamps[1:]) if t2 > t1]
    if not diffs:
        return float(default_fps)
    median_dt = statistics.median(diffs)
    if median_dt <= 0:
        return float(default_fps)
    return float(max(1.0, min(60.0, 1.0 / median_dt)))


def _segment_action_windows(recorded: Dict[str, list]) -> Dict[int, tuple]:
    """每个任务段“机械臂真正在动”的时间窗。

    起点 = 该段首帧动作对应的模型输入图像时间（image_timestamp）；
    终点 = 该段最后一步动作下发时间（timestamp）+ 3s（观察收尾/稳定过程）。
    段号来自 recorded["task_segment_index"]（仅动作步写入）。
    """
    seg_ids = recorded.get("task_segment_index") or []
    rec_ts = recorded.get("timestamp") or []
    img_ts = recorded.get("image_timestamp") or []
    if not seg_ids or len(seg_ids) != len(rec_ts):
        return {}
    windows: Dict[int, list] = {}
    for i, seg in enumerate(seg_ids):
        if seg is None:
            continue
        try:
            seg_i = int(seg)
        except (TypeError, ValueError):
            continue
        if seg_i <= 0:
            continue
        t_rec = float(rec_ts[i])
        t_img = float(img_ts[i]) if i < len(img_ts) else t_rec
        if seg_i in windows:
            lo, hi = windows[seg_i]
            windows[seg_i] = [min(lo, t_img), max(hi, t_rec)]
        else:
            windows[seg_i] = [t_img, t_rec]
    return {seg: (lo, hi + 3.0) for seg, (lo, hi) in windows.items()}


def create_video_writer(
    output_path: Path,
    frame_size: Tuple[int, int],
    fps: float,
) -> Tuple[Optional[cv2.VideoWriter], Path]:
    """创建 VideoWriter。优先 mp4v（软件编码，机器无关）。

    勿把 avc1 放首位：不少 OpenCV/FFmpeg 会映射到 h264_v4l2m2m，
    无 V4L2 硬件编码器时会打 ERROR 再失败，最终仍落到 mp4v。
    """
    width, height = frame_size
    candidates = [
        (output_path, "mp4v"),
        (output_path, "avc1"),
        (output_path.with_suffix(".avi"), "XVID"),
        (output_path.with_suffix(".avi"), "MJPG"),
    ]
    for path_candidate, fourcc_name in candidates:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
        writer = cv2.VideoWriter(str(path_candidate), fourcc, float(fps), (width, height))
        if writer.isOpened():
            logger.info("视频编码器: %s -> %s @ %.2ffps", fourcc_name, path_candidate, fps)
            return writer, path_candidate
        writer.release()
        logger.debug("视频编码器不可用: %s (%s)", fourcc_name, path_candidate)
    return None, output_path


def _list_value(recorded: Dict[str, list], key: str, index: int) -> Any:
    values = recorded.get(key, [])
    if index < len(values):
        return values[index]
    return None


def _series_values(recorded: Dict[str, list], key: str, length: int) -> List[Any]:
    values = list(recorded.get(key, []))
    if len(values) >= length:
        return values[:length]
    return values + [None] * (length - len(values))


def _write_video_timestamp_index(
    path: Path,
    *,
    cam_name: str,
    video_path: Path,
    recorded: Dict[str, list],
    source_indices: List[int],
    camera_suffix: str = "",
    frame_paths: Optional[List[Path]] = None,
    depth_format: Optional[str] = None,
) -> int:
    """一行对应 video_path 里的一帧。video_frame_index 是该文件内的帧号。

    depth raw PNG 序列时：``frame_paths[i]`` 指向第 i 帧 uint16 PNG，
    ``depth_format="uint16_png_mm"``。
    """
    camera_name = DEPTH_CAMERA_NAME_MAP.get(cam_name) or CAMERA_NAME_MAP.get(cam_name) or cam_name
    if camera_suffix:
        camera_name = f"{camera_name}{camera_suffix}"
    source = (
        "aligned_model_input"
        if cam_name in {"cam_left", "cam_mid", "cam_left_depth", "cam_mid_depth"}
        else "camera_packet"
    )
    rows_written = 0
    with path.open("a", encoding="utf-8") as fp:
        for out_i, src_i in enumerate(source_indices):
            row = {
                "type": "video_frame_timestamp",
                "camera": camera_name,
                "camera_key": cam_name,
                "video_path": str(video_path),
                "video_frame_index": out_i,
                "source_frame_index": src_i,
                "source": source,
                "image_timestamp": _list_value(recorded, f"{cam_name}_timestamp", src_i),
                "state_timestamp": _list_value(recorded, f"{cam_name}_state_timestamp", src_i),
                "inference_frame_index": _list_value(recorded, f"{cam_name}_frame_index", src_i),
                "action_step_index": _list_value(recorded, f"{cam_name}_action_step_index", src_i),
                "record_timestamp": _list_value(recorded, f"{cam_name}_record_timestamp", src_i),
                "task_segment_index": _list_value(recorded, f"{cam_name}_task_id", src_i),
                "state_reference_semantics": "logged_reference_not_simultaneous_capture",
            }
            if depth_format is not None:
                row["depth_format"] = depth_format
                row["depth_unit"] = "mm"
                row["depth_dtype"] = "uint16"
            if frame_paths is not None and out_i < len(frame_paths):
                row["raw_path"] = str(frame_paths[out_i])
            fp.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            rows_written += 1
    return rows_written


def _save_depth_raw_for_cam(
    *,
    videos_dir: Path,
    cam_name: str,
    frames: List[np.ndarray],
    source_indices: List[int],
    name_stem: str,
    recorded: Dict[str, list],
    timestamp_index_path: Path,
) -> Dict[str, Any]:
    """把选定帧写成 uint16 mm PNG 序列，并挂到 frame_timestamps。"""
    selected = [frames[i] for i in source_indices]
    out_dir = videos_dir / f"{name_stem}_u16"
    info = save_depth_u16_sequence(out_dir, selected)
    frame_paths = [Path(p) for p in info["paths"]]
    rows = _write_video_timestamp_index(
        timestamp_index_path,
        cam_name=cam_name,
        video_path=out_dir,
        recorded=recorded,
        source_indices=source_indices,
        camera_suffix="_u16",
        frame_paths=frame_paths,
        depth_format="uint16_png_mm",
    )
    logger.info(
        "depth raw 已保存: %s (frames=%d, uint16 mm PNG)",
        out_dir,
        info["n_frames"],
    )
    return {**info, "timestamp_rows": rows}


def _write_rgb_video(
    target: Path,
    frames: List[np.ndarray],
    frame_size: Tuple[int, int],
    fps: float,
) -> Optional[Path]:
    writer, real_path = create_video_writer(target, frame_size, fps)
    if writer is None:
        return None
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    return real_path


def task_local_frame_index(task_ids: list, global_index: int) -> int:
    """该帧在同一 task_id 里的序号，与这一段 mp4 的帧号一致。"""
    task_id = task_ids[global_index]
    return sum(1 for t in task_ids[: global_index + 1] if t == task_id) - 1


MIN_TASK_VIDEO_FRAMES = 10


def _frames_by_task(recorded: Dict[str, list], cam_name: str, n: int) -> Optional[Dict[int, List[int]]]:
    task_ids = recorded.get(f"{cam_name}_task_id") or []
    if len(task_ids) != n:
        return None
    groups: Dict[int, List[int]] = {}
    for i, tid in enumerate(task_ids):
        if tid is None:
            continue
        try:
            tid_i = int(tid)
        except (TypeError, ValueError):
            continue
        if tid_i <= 0:
            continue
        groups.setdefault(tid_i, []).append(i)
    if not groups:
        return None
    return groups


def record_extra_camera_frame(
    recorded: Dict[str, list],
    *,
    cam_key: str,
    ts_key: str,
    image_rgb: np.ndarray,
    timestamp: float,
    last_ts_attr: str,
    state: Dict[str, Optional[float]],
    depth_u16: Optional[np.ndarray] = None,
    depth_key: Optional[str] = None,
    task_id: Optional[int] = None,
    state_ts: Optional[float] = None,
    action_step_index: Optional[int] = None,
    inference_frame_index: Optional[int] = None,
) -> Optional[int]:
    """按相机包时间戳去重追加。返回缓冲里的全局序号；重复时间戳返回 None。

    同时把该帧对应的 state 时间戳 / 动作步号写进帧缓冲，
    ``frame_timestamps.jsonl`` 因此能直接索引到关节角。
    """
    last_ts = state.get(last_ts_attr)
    if last_ts is not None and timestamp <= last_ts:
        return None
    recorded[cam_key].append(np.copy(image_rgb))
    recorded[ts_key].append(float(timestamp))
    recorded.setdefault(f"{cam_key}_task_id", []).append(task_id)
    _append_frame_index_value(recorded, f"{cam_key}_state_timestamp", state_ts)
    _append_frame_index_value(recorded, f"{cam_key}_action_step_index", action_step_index)
    _append_frame_index_value(recorded, f"{cam_key}_frame_index", inference_frame_index)
    state[last_ts_attr] = float(timestamp)
    if depth_u16 is not None and depth_key is not None:
        recorded[depth_key].append(np.copy(depth_u16))
        recorded[f"{depth_key}_timestamp"].append(float(timestamp))
        recorded.setdefault(f"{depth_key}_task_id", []).append(task_id)
        _append_frame_index_value(recorded, f"{depth_key}_state_timestamp", state_ts)
        _append_frame_index_value(recorded, f"{depth_key}_action_step_index", action_step_index)
        _append_frame_index_value(recorded, f"{depth_key}_frame_index", inference_frame_index)
    return len(recorded[cam_key]) - 1


def _append_frame_index_value(recorded: Dict[str, list], key: str, value: Any) -> None:
    """保证索引键与相机帧列表等长；缺口补 None。"""
    seq = recorded.setdefault(key, [])
    cam_key = key.rsplit("_", 1)[0]
    for suffix in ("_state_timestamp", "_action_step_index", "_frame_index"):
        if key.endswith(suffix):
            cam_key = key[: -len(suffix)]
            break
    target_len = len(recorded.get(cam_key, []))
    while len(seq) < target_len - 1:
        seq.append(None)
    if len(seq) == target_len - 1:
        seq.append(value)


def _insert_if_aligned(
    recorded: Dict[str, list],
    key: str,
    index: int,
    value: Any,
    expected_len: int,
) -> None:
    seq = recorded.get(key)
    if not isinstance(seq, list) or len(seq) != expected_len:
        return
    seq.insert(index, value)


def place_recorded_camera_frame(
    recorded: Dict[str, list],
    *,
    cam_key: str,
    image_rgb: np.ndarray,
    timestamp: float,
    last_ts_state: Dict[str, Optional[float]],
    task_id: Optional[int] = None,
    depth_u16: Optional[np.ndarray] = None,
    force: bool = False,
    min_interval_sec: float = 0.0,
    state_ts: Optional[float] = None,
    action_step_index: Optional[int] = None,
    inference_frame_index: Optional[int] = None,
) -> Tuple[Optional[int], str]:
    """按保存间隔留下一帧。返回 (全局序号, appended|inserted|existing|skipped)。

    普通采样比上次留下的帧更近则跳过，录像因此落在控制频率上。
    force 用来留下模型真正吃进去的那一帧：它比后写的帧更早时，按采集时间插回去。
    """
    ts = float(timestamp)
    ts_key = f"{cam_key}_timestamp"
    ts_list = recorded.setdefault(ts_key, [])
    for j in range(len(ts_list) - 1, -1, -1):
        if float(ts_list[j]) == ts:
            # A camera callback may have saved this frame before model inference.
            # Enrich missing references without duplicating/reassigning the frame.
            if force and _list_value(recorded, f"{cam_key}_task_id", j) == task_id:
                for camera_key in (cam_key, f"{cam_key}_depth"):
                    camera_times = recorded.get(f"{camera_key}_timestamp", [])
                    if j >= len(camera_times) or float(camera_times[j]) != ts:
                        continue
                    if _list_value(recorded, f"{camera_key}_task_id", j) != task_id:
                        continue
                    references = (("state_timestamp", state_ts),
                                  ("action_step_index", action_step_index),
                                  ("frame_index", inference_frame_index))
                    if any(value is not None and _list_value(recorded, f"{camera_key}_{suffix}", j) not in (None, value)
                           for suffix, value in references):
                        continue  # Never combine parts of different inference references.
                    for suffix, value in references:
                        values = recorded.setdefault(f"{camera_key}_{suffix}", [])
                        while len(values) < len(camera_times):
                            values.append(None)
                        if values[j] is None and value is not None:
                            values[j] = value
            return j, "existing"
    last_ts = last_ts_state.get(cam_key)
    due = last_ts is None or ts - float(last_ts) >= float(min_interval_sec)
    if not force and not due:
        return None, "skipped"
    if last_ts is None or ts > float(last_ts):
        depth_key = f"{cam_key}_depth" if depth_u16 is not None else None
        idx = record_extra_camera_frame(
            recorded,
            cam_key=cam_key,
            ts_key=ts_key,
            image_rgb=image_rgb,
            timestamp=ts,
            last_ts_attr=cam_key,
            state=last_ts_state,
            depth_u16=depth_u16,
            depth_key=depth_key,
            task_id=task_id,
            state_ts=state_ts,
            action_step_index=action_step_index,
            inference_frame_index=inference_frame_index,
        )
        if idx is None:
            return None, "skipped"
        return idx, "appended"
    insert_at = next((j for j, old in enumerate(ts_list) if ts < float(old)), len(ts_list))
    n = len(ts_list)
    frames = recorded.setdefault(cam_key, [])
    if len(frames) != n:
        return None, "skipped"
    frames.insert(insert_at, np.copy(image_rgb))
    ts_list.insert(insert_at, ts)
    _insert_if_aligned(recorded, f"{cam_key}_task_id", insert_at, task_id, n)
    _insert_if_aligned(recorded, f"{cam_key}_state_timestamp", insert_at, state_ts, n)
    _insert_if_aligned(recorded, f"{cam_key}_action_step_index", insert_at, action_step_index, n)
    _insert_if_aligned(recorded, f"{cam_key}_frame_index", insert_at, inference_frame_index, n)
    if depth_u16 is not None:
        depth_key = f"{cam_key}_depth"
        depth_frames = recorded.get(depth_key)
        depth_ts = recorded.get(f"{depth_key}_timestamp")
        if (
            isinstance(depth_frames, list)
            and isinstance(depth_ts, list)
            and len(depth_frames) == n
            and len(depth_ts) == n
        ):
            depth_frames.insert(insert_at, np.copy(depth_u16))
            depth_ts.insert(insert_at, ts)
            _insert_if_aligned(recorded, f"{depth_key}_task_id", insert_at, task_id, n)
            _insert_if_aligned(recorded, f"{depth_key}_state_timestamp", insert_at, state_ts, n)
            _insert_if_aligned(recorded, f"{depth_key}_action_step_index", insert_at, action_step_index, n)
            _insert_if_aligned(recorded, f"{depth_key}_frame_index", insert_at, inference_frame_index, n)
    return insert_at, "inserted"


def save_infer_recorded_data(
    recorded: Dict[str, list],
    test_dir: Path,
    *,
    fps: float,
    save_videos: bool = True,
    save_trajectory: bool = True,
    save_third_cam: bool = True,
    save_fourth_cam: bool = True,
    save_depth_videos: bool = True,
    save_depth_raw: bool = True,
    on_event: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    将 recorded 字典写入 test_dir/videos 与 trajectory_data/。

    depth：
    - ``save_depth_raw``（默认开）：落盘 uint16 mm PNG 序列（可分析/进训）
    - ``save_depth_videos``：额外 JET 伪彩 mp4，仅预览，不可反算毫米
    """
    videos_dir = test_dir / "videos"
    traj_dir = test_dir / "trajectory_data"
    videos_dir.mkdir(parents=True, exist_ok=True)
    traj_dir.mkdir(parents=True, exist_ok=True)

    want_depth = bool(save_depth_videos or save_depth_raw)
    cam_keys = ["cam_left", "cam_mid"]
    if save_third_cam:
        cam_keys.append("cam_third")
    if save_fourth_cam:
        cam_keys.append("cam_fourth")
    if want_depth:
        cam_keys.extend(["cam_left_depth", "cam_mid_depth"])

    has_video = any(len(recorded.get(name, [])) > 0 for name in cam_keys)
    recorded_frames = len(recorded.get("state", []))
    if not has_video and recorded_frames == 0:
        return {"saved_videos": [], "saved_depth_raw": [], "traj_path": None, "recorded_frames": 0}

    saved_videos: List[str] = []
    saved_depth_raw: List[str] = []
    timestamp_index_path = videos_dir / "frame_timestamps.jsonl"
    timestamp_rows = 0
    if save_videos or save_depth_raw:
        if timestamp_index_path.exists():
            timestamp_index_path.unlink()
        seg_windows = _segment_action_windows(recorded)
        name_map = {**CAMERA_NAME_MAP, **DEPTH_CAMERA_NAME_MAP}
        for cam_name in cam_keys:
            frames = recorded.get(cam_name, [])
            if not frames:
                continue
            is_depth = cam_name in DEPTH_CAMERA_NAME_MAP
            cam_ts = [float(t) for t in recorded.get(f"{cam_name}_timestamp", [])]
            n_frames = len(frames)
            timestamps_aligned = len(cam_ts) == n_frames and len(cam_ts) >= 2
            cam_fps = estimate_video_fps(cam_ts, fps) if timestamps_aligned else float(fps)
            source_all = list(range(n_frames))
            stem = name_map[cam_name]

            task_groups = _frames_by_task(recorded, cam_name, n_frames)

            # --- metric depth raw (uint16 mm) ---
            if is_depth and save_depth_raw:
                if task_groups:
                    for seg_idx, idxs in sorted(task_groups.items()):
                        if len(idxs) < MIN_TASK_VIDEO_FRAMES:
                            logger.info(
                                "跳过过短任务 depth raw: %s task%d frames=%d (<%d)",
                                cam_name, seg_idx, len(idxs), MIN_TASK_VIDEO_FRAMES,
                            )
                            continue
                        info = _save_depth_raw_for_cam(
                            videos_dir=videos_dir,
                            cam_name=cam_name,
                            frames=frames,
                            source_indices=idxs,
                            name_stem=f"{stem}_task{seg_idx:02d}",
                            recorded=recorded,
                            timestamp_index_path=timestamp_index_path,
                        )
                        saved_depth_raw.append(info["dir"])
                        timestamp_rows += int(info["timestamp_rows"])
                else:
                    info = _save_depth_raw_for_cam(
                        videos_dir=videos_dir,
                        cam_name=cam_name,
                        frames=frames,
                        source_indices=source_all,
                        name_stem=stem,
                        recorded=recorded,
                        timestamp_index_path=timestamp_index_path,
                    )
                    saved_depth_raw.append(info["dir"])
                    timestamp_rows += int(info["timestamp_rows"])
                    if seg_windows and timestamps_aligned:
                        for seg_idx, (t0, t1) in sorted(seg_windows.items()):
                            idxs = [i for i, t in enumerate(cam_ts) if t0 <= t <= t1]
                            if not idxs:
                                continue
                            info = _save_depth_raw_for_cam(
                                videos_dir=videos_dir,
                                cam_name=cam_name,
                                frames=frames,
                                source_indices=idxs,
                                name_stem=f"{stem}_task{seg_idx:02d}",
                                recorded=recorded,
                                timestamp_index_path=timestamp_index_path,
                            )
                            saved_depth_raw.append(info["dir"])
                            timestamp_rows += int(info["timestamp_rows"])

            # --- RGB / optional depth JET preview mp4 ---
            write_preview = save_videos and ((not is_depth) or save_depth_videos)
            if not write_preview:
                continue
            if is_depth:
                vis_frames = [depth_u16_to_video_rgb(frame) for frame in frames]
                height, width, _ = vis_frames[0].shape
            else:
                vis_frames = frames
                height, width, _ = frames[0].shape

            if task_groups:
                for seg_idx, idxs in sorted(task_groups.items()):
                    if len(idxs) < MIN_TASK_VIDEO_FRAMES:
                        logger.info(
                            "跳过过短任务视频: %s task%d frames=%d (<%d)",
                            cam_name, seg_idx, len(idxs), MIN_TASK_VIDEO_FRAMES,
                        )
                        continue
                    seg_frames = [vis_frames[i] for i in idxs]
                    seg_ts = [cam_ts[i] for i in idxs] if timestamps_aligned else []
                    seg_fps = estimate_video_fps(seg_ts, cam_fps) if len(seg_ts) >= 2 else cam_fps
                    seg_target = videos_dir / f"{stem}_task{seg_idx:02d}.mp4"
                    seg_real = _write_rgb_video(seg_target, seg_frames, (width, height), seg_fps)
                    if seg_real is None:
                        logger.error("无法创建任务视频写入器: %s task%d", cam_name, seg_idx)
                        continue
                    saved_videos.append(str(seg_real))
                    timestamp_rows += _write_video_timestamp_index(
                        timestamp_index_path,
                        cam_name=cam_name,
                        video_path=seg_real,
                        recorded=recorded,
                        source_indices=idxs,
                    )
                    logger.info(
                        "任务视频已保存: %s (frames=%d, %.2ffps, task=%d)",
                        seg_real, len(seg_frames), seg_fps, seg_idx,
                    )
                continue

            target = videos_dir / f"{stem}.mp4"
            real_path = _write_rgb_video(target, vis_frames, (width, height), cam_fps)
            if real_path is None:
                logger.error("无法创建视频写入器: %s", cam_name)
                continue
            saved_videos.append(str(real_path))
            timestamp_rows += _write_video_timestamp_index(
                timestamp_index_path,
                cam_name=cam_name,
                video_path=real_path,
                recorded=recorded,
                source_indices=source_all,
            )
            logger.info("视频已保存: %s (frames=%d, %.2ffps)", real_path, len(vis_frames), cam_fps)

            if not (seg_windows and timestamps_aligned):
                continue
            for seg_idx, (t0, t1) in sorted(seg_windows.items()):
                idxs = [i for i, t in enumerate(cam_ts) if t0 <= t <= t1]
                if not idxs:
                    continue
                seg_frames = [vis_frames[i] for i in idxs]
                seg_target = videos_dir / f"{stem}_task{seg_idx:02d}.mp4"
                seg_real = _write_rgb_video(seg_target, seg_frames, (width, height), cam_fps)
                if seg_real is None:
                    logger.error("无法创建分段视频写入器: %s task%d", cam_name, seg_idx)
                    continue
                saved_videos.append(str(seg_real))
                timestamp_rows += _write_video_timestamp_index(
                    timestamp_index_path,
                    cam_name=cam_name,
                    video_path=seg_real,
                    recorded=recorded,
                    source_indices=idxs,
                )
                logger.info(
                    "分段视频已保存: %s (frames=%d, %.2ffps)",
                    seg_real, len(seg_frames), cam_fps,
                )

    traj_path = None
    if save_trajectory and recorded_frames > 0:
        try:
            import pandas as pd

            traj_path = traj_dir / "trajectory.parquet"
            df = pd.DataFrame({
                "timestamp": _series_values(recorded, "timestamp", recorded_frames),
                "state_timestamp": _series_values(recorded, "state_timestamp", recorded_frames),
                "image_timestamp": _series_values(recorded, "image_timestamp", recorded_frames),
                "inference_frame_index": _series_values(recorded, "frame_index", recorded_frames),
                "action_step_index": _series_values(recorded, "action_step_index", recorded_frames),
                "task_segment_index": _series_values(recorded, "task_segment_index", recorded_frames),
                "observation.state": [s.tolist() for s in recorded["state"]],
                "action": [a.tolist() for a in recorded["action"]],
            })
            df.to_parquet(traj_path)
            logger.info("轨迹已保存: %s", traj_path)
        except ImportError:
            logger.warning("未安装 pandas，跳过 trajectory.parquet")
        except Exception:
            logger.exception("保存 trajectory 失败")

    summary = {
        "saved_videos": saved_videos,
        "saved_depth_raw": saved_depth_raw,
        "video_timestamp_index": str(timestamp_index_path) if timestamp_rows else None,
        "video_timestamp_rows": timestamp_rows,
        "traj_path": str(traj_path) if traj_path else None,
        "recorded_frames": recorded_frames,
        "third_recorded_frames": len(recorded.get("cam_third", [])),
        "fourth_recorded_frames": len(recorded.get("cam_fourth", [])),
        "left_depth_recorded_frames": len(recorded.get("cam_left_depth", [])),
        "mid_depth_recorded_frames": len(recorded.get("cam_mid_depth", [])),
    }
    if on_event:
        on_event("recorded_data_saved", summary)
    return summary
