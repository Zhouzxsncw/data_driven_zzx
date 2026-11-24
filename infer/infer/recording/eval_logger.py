"""Evaluation-time data log: model inputs, images, outputs, and state feedback."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from infer.io.gripper_state_udp import empty_gripper_udp_feedback

logger = logging.getLogger(__name__)

try:
    import cv2
except ImportError:  # pragma: no cover - runtime inference env normally provides cv2.
    cv2 = None  # type: ignore


CAMERA_KEYS = ("cam_left", "cam_mid", "cam_third", "cam_fourth")


def _array_to_list(value: Any) -> Optional[List[float]]:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.size == 0:
        return []
    return arr.reshape(-1).astype(float).tolist()


def _plain_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _plain_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json_value(v) for v in value]
    if hasattr(value, "__len__") and not isinstance(value, (bytes, bytearray)):
        try:
            return [_plain_json_value(v) for v in value]
        except TypeError:
            pass
    return value


def normalize_gripper_udp_feedback(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return empty_gripper_udp_feedback()
    return {
        "source": str(value.get("source", "none")),
        "received": bool(value.get("received")),
        "left": _plain_json_value(value.get("left")),
        "right": _plain_json_value(value.get("right")),
        "left_current_mA": _plain_json_value(value.get("left_current_mA")),
        "right_current_mA": _plain_json_value(value.get("right_current_mA")),
        "packet_timestamp": _plain_json_value(value.get("packet_timestamp")),
        "age_sec": _plain_json_value(value.get("age_sec")),
    }


def _api_data(api_state: Optional[Dict[str, Any]], arm: str, key: str) -> Optional[Dict[str, Any]]:
    if not api_state:
        return None
    item = api_state.get(arm, {}).get(key)
    if not isinstance(item, dict) or not item.get("ok"):
        return None
    data = item.get("data")
    return data if isinstance(data, dict) else None


def _arm_field(api_state: Optional[Dict[str, Any]], key: str, *names: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for arm in ("left_arm", "right_arm"):
        data = _api_data(api_state, arm, key)
        value = None
        if data is not None:
            for name in names:
                if name in data:
                    value = _plain_json_value(data[name])
                    break
        out[arm] = value
    return out


def state_vector_to_dict(
    state_15d: Any,
    api_state: Optional[Dict[str, Any]] = None,
    *,
    gripper_udp_feedback: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Represent the available robot state and reserve interfaces for missing streams."""
    vec = _array_to_list(state_15d)
    udp_fb = (
        normalize_gripper_udp_feedback(gripper_udp_feedback)
        if gripper_udp_feedback is not None
        else None
    )
    if vec is None:
        return {
            "raw_15d": None,
            "joint_position": None,
            "end_effector_pose": None,
            "joint_velocity": None,
            "joint_current": None,
            "end_effector_wrench_6d": None,
            "gripper_state": None,
            "gripper_udp_feedback": udp_fb,
            "realman_api": _plain_json_value(api_state),
        }

    gripper_state: Dict[str, Any] = {
        "left_gripper": vec[14] if len(vec) >= 15 else None,
        "right_gripper": vec[15] if len(vec) >= 16 else None,
        "raw": vec[14:16] if len(vec) >= 16 else (vec[14:15] if len(vec) >= 15 else None),
    }
    if udp_fb is not None:
        gripper_state["observation_source"] = (
            "udp" if udp_fb.get("received") else "udp_no_fresh_packet"
        )
        gripper_state["udp_feedback"] = udp_fb

    return {
        "raw_15d": vec,
        "joint_position": {
            "left_arm": vec[0:7] if len(vec) >= 7 else None,
            "right_arm": vec[7:14] if len(vec) >= 14 else None,
        },
        "end_effector_pose": _arm_field(api_state, "current_arm_state", "pose"),
        "joint_velocity": _arm_field(api_state, "joint_speed", "joint_speed"),
        "joint_current": _arm_field(api_state, "arm_all_state", "joint_current"),
        "joint_temperature": _arm_field(api_state, "arm_all_state", "joint_temperature"),
        "joint_voltage": _arm_field(api_state, "arm_all_state", "joint_voltage"),
        "joint_error_code": _arm_field(api_state, "arm_all_state", "joint_err_code"),
        "joint_enable_state": _arm_field(api_state, "arm_all_state", "joint_en_flag"),
        "controller_voltage": _arm_field(api_state, "controller_state", "voltage"),
        "controller_current": _arm_field(api_state, "controller_state", "current"),
        "controller_temperature": _arm_field(api_state, "controller_state", "temperature"),
        "controller_error": _arm_field(api_state, "controller_state", "system_error"),
        "joint_brake_state": _arm_field(api_state, "brake_state", "brake_state"),
        "end_effector_wrench_6d": _arm_field(
            api_state,
            "force_data",
            "force_data",
            "zero_force_data",
            "work_zero_force_data",
            "tool_zero_force_data",
        ),
        "gripper_state": gripper_state,
        "gripper_udp_feedback": udp_fb,
        "realman_api": _plain_json_value(api_state),
    }


def model_state_to_dict(model_state: Any, keep_dims: Iterable[int]) -> Dict[str, Any]:
    dims = [int(x) for x in keep_dims]
    return {
        "vector": _array_to_list(model_state),
        "source": "robot_state.raw_15d",
        "keep_dims": dims,
        "fields_note": "Current checkpoint input vector after applying policy.keep_dims.",
    }


class EvalLogger:
    """Append-only eval log with camera frames stored as image files."""

    def __init__(
        self,
        test_dir: Path,
        *,
        enabled: bool = True,
        filename: str = "eval_log.jsonl",
        save_images: bool = False,
        image_dirname: str = "eval_images",
        image_format: str = "jpg",
        save_videos: bool = False,
        video_dirname: str = "eval_videos",
        video_fps: float = 15.0,
        generate_plots: bool = True,
        plot_dirname: str = "eval_plots",
        plot_dpi: int = 150,
    ) -> None:
        self.enabled = enabled
        self.test_dir = test_dir
        self.path = test_dir / filename
        self.image_dir = test_dir / image_dirname
        self.video_dir = test_dir / video_dirname
        self.image_format = image_format.lower().lstrip(".") or "jpg"
        if self.image_format == "jpeg":
            self.image_format = "jpg"
        self.save_images = bool(save_images)
        self.save_videos = bool(save_videos)
        self.video_fps = float(video_fps)
        self.generate_plots = bool(generate_plots)
        self.plot_dirname = str(plot_dirname)
        self.plot_dpi = int(plot_dpi)
        self._plot_summary: Optional[Dict[str, Any]] = None
        self.video_timestamp_index = self.video_dir / "frame_timestamps.jsonl"
        self._fp = None
        self._video_writers: Dict[str, Any] = {}
        self._video_paths: Dict[str, Path] = {}
        self._video_frame_counts: Dict[str, int] = {}
        self._t0 = time.time()

        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.save_images:
                self.image_dir.mkdir(parents=True, exist_ok=True)
            if self.save_videos:
                self.video_dir.mkdir(parents=True, exist_ok=True)
                if self.video_timestamp_index.exists():
                    self.video_timestamp_index.unlink()
            self._fp = self.path.open("a", encoding="utf-8")

    def close(self) -> None:
        for writer in self._video_writers.values():
            writer.release()
        self._video_writers.clear()
        if self.enabled and self.save_videos:
            self._write_summary()
        if self.enabled and self.generate_plots and self.path.is_file():
            try:
                from infer.recording.eval_log_plots import generate_eval_log_plots_for_test_dir

                self._plot_summary = generate_eval_log_plots_for_test_dir(
                    self.test_dir,
                    eval_log_filename=self.path.name,
                    output_dirname=self.plot_dirname,
                    dpi=self.plot_dpi,
                )
                if self._plot_summary is not None:
                    self._write_plot_summary()
            except Exception:
                logger.exception("生成 eval_log 对比图失败")
        if self._fp is not None:
            self._fp.close()
            self._fp = None

    def _write_summary(self) -> None:
        summary = {
            "eval_log": str(self.path.relative_to(self.test_dir)),
            "image_dir": str(self.image_dir.relative_to(self.test_dir)),
            "video_dir": str(self.video_dir.relative_to(self.test_dir)),
            "video_timestamp_index": str(self.video_timestamp_index.relative_to(self.test_dir)),
            "videos": [
                {
                    "key": key,
                    "path": str(path.relative_to(self.test_dir)),
                    "frames": self._video_frame_counts.get(key, 0),
                    "fps": self.video_fps,
                }
                for key, path in sorted(self._video_paths.items())
            ],
        }
        (self.test_dir / "eval_log_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    def _write_plot_summary(self) -> None:
        if self._plot_summary is None:
            return
        (self.test_dir / "eval_log_plots_summary.json").write_text(
            json.dumps(self._plot_summary, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    def _write_jsonl(self, row: Dict[str, Any]) -> None:
        if not self.enabled or self._fp is None:
            return
        self._fp.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        self._fp.flush()

    def _save_images(self, frame_index: int, images: list) -> List[Dict[str, Any]]:
        if not self.save_images:
            return []
        saved: List[Dict[str, Any]] = []
        for i, image in enumerate(images):
            if image is None:
                continue
            key = CAMERA_KEYS[i] if i < len(CAMERA_KEYS) else f"cam_{i}"
            rel = Path(self.image_dir.name) / f"frame_{frame_index:06d}_{key}.{self.image_format}"
            path = self.test_dir / rel
            arr = np.asarray(image)
            if arr.ndim != 3:
                continue
            if cv2 is None:
                saved.append(
                    {
                        "key": key,
                        "index": i,
                        "path": str(rel),
                        "shape": list(arr.shape),
                        "saved": False,
                        "error": "cv2_not_available",
                    }
                )
                continue
            ok = cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
            saved.append(
                {
                    "key": key,
                    "index": i,
                    "path": str(rel),
                    "shape": list(arr.shape),
                    "saved": bool(ok),
                }
            )
        return saved

    def _video_writer_for(self, key: str, arr: np.ndarray) -> Optional[Any]:
        if not self.save_videos or cv2 is None:
            return None
        if key in self._video_writers:
            return self._video_writers[key]
        height, width = arr.shape[:2]
        path = self.video_dir / f"{key}.mp4"
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.video_fps,
            (width, height),
        )
        if not writer.isOpened():
            writer.release()
            path = self.video_dir / f"{key}.avi"
            writer = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                self.video_fps,
                (width, height),
            )
        if not writer.isOpened():
            writer.release()
            return None
        self._video_writers[key] = writer
        self._video_paths[key] = path
        self._video_frame_counts[key] = 0
        return writer

    def _write_video_timestamp_row(
        self,
        *,
        key: str,
        path: Path,
        video_frame_index: int,
        frame_index: int,
        state_timestamp: Optional[float],
        image_timestamp: Optional[float],
        image_diff: Optional[float],
    ) -> None:
        row = {
            "type": "eval_video_frame_timestamp",
            "camera": key,
            "video_path": str(path.relative_to(self.test_dir)),
            "video_frame_index": int(video_frame_index),
            "inference_frame_index": int(frame_index),
            "state_timestamp": state_timestamp,
            "image_timestamp": image_timestamp,
            "image_state_diff_sec": image_diff,
            "source": "aligned_model_input",
        }
        with self.video_timestamp_index.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    def _append_video_frames(
        self,
        images: list,
        *,
        frame_index: int,
        state_timestamp: Optional[float],
        image_timestamp: Optional[float],
        image_diff: Optional[float],
    ) -> List[Dict[str, Any]]:
        videos: List[Dict[str, Any]] = []
        for i, image in enumerate(images):
            if image is None:
                continue
            key = CAMERA_KEYS[i] if i < len(CAMERA_KEYS) else f"cam_{i}"
            arr = np.asarray(image)
            if arr.ndim != 3:
                continue
            if cv2 is None:
                videos.append({"key": key, "saved": False, "error": "cv2_not_available"})
                continue
            writer = self._video_writer_for(key, arr)
            if writer is None:
                videos.append({"key": key, "saved": False, "error": "video_writer_unavailable"})
                continue
            video_frame_index = self._video_frame_counts.get(key, 0)
            writer.write(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
            self._write_video_timestamp_row(
                key=key,
                path=self._video_paths[key],
                video_frame_index=video_frame_index,
                frame_index=frame_index,
                state_timestamp=state_timestamp,
                image_timestamp=image_timestamp,
                image_diff=image_diff,
            )
            self._video_frame_counts[key] = video_frame_index + 1
            videos.append(
                {
                    "key": key,
                    "path": str(self._video_paths[key].relative_to(self.test_dir)),
                    "video_frame_index": video_frame_index,
                    "frame_count": self._video_frame_counts[key],
                    "saved": True,
                }
            )
        return videos

    def log_inference_frame(
        self,
        *,
        frame_index: int,
        state_timestamp: Optional[float],
        image_timestamp: Optional[float],
        image_diff: Optional[float],
        state_15d: Any,
        model_state: Any,
        keep_dims: Iterable[int],
        images: list,
        task_prompt: str,
        raw_action_chunk: Any,
        post_action_chunk: Any,
        infer_timing: Dict[str, Any],
        api_state: Optional[Dict[str, Any]] = None,
        rtc_info: Optional[Dict[str, Any]] = None,
        action_steps: Optional[List[Dict[str, Any]]] = None,
        gripper_udp_feedback: Optional[Dict[str, Any]] = None,
        gripper_udp_timeline: Optional[List[Dict[str, Any]]] = None,
        ik: Optional[Dict[str, Any]] = None,
    ) -> None:
        frame_gripper_udp = normalize_gripper_udp_feedback(gripper_udp_feedback)
        timeline = [
            {
                "event": str(item.get("event")),
                "wall_time": _plain_json_value(item.get("wall_time")),
                "chunk_step_index": _plain_json_value(item.get("chunk_step_index")),
                "gripper_udp_feedback": normalize_gripper_udp_feedback(
                    item.get("gripper_udp_feedback")
                ),
            }
            for item in (gripper_udp_timeline or [])
        ]
        row = {
            "type": "eval_inference_frame",
            "frame_index": int(frame_index),
            "wall_time": time.time(),
            "elapsed_sec": time.time() - self._t0,
            "timestamps": {
                "state": state_timestamp,
                "image": image_timestamp,
                "image_state_diff_sec": image_diff,
            },
            "gripper_udp_feedback": frame_gripper_udp,
            "gripper_udp_timeline": timeline,
            "checkpoint_input": {
                "robot_state": state_vector_to_dict(
                    state_15d,
                    api_state,
                    gripper_udp_feedback=frame_gripper_udp,
                ),
                "model_state": model_state_to_dict(model_state, keep_dims),
                "task_prompt": task_prompt,
                "cameras": self._save_images(frame_index, images),
                "camera_videos": (
                    self._append_video_frames(
                        images,
                        frame_index=frame_index,
                        state_timestamp=state_timestamp,
                        image_timestamp=image_timestamp,
                        image_diff=image_diff,
                    )
                    if self.save_videos
                    else []
                ),
            },
            "network_output": {
                "raw_action_chunk": None
                if raw_action_chunk is None
                else np.asarray(raw_action_chunk).tolist(),
                "post_action_chunk": None
                if post_action_chunk is None
                else np.asarray(post_action_chunk).tolist(),
                "action_space": {
                    "type": "policy_action",
                    "keep_dims": [int(x) for x in keep_dims],
                    "note": "post_action_chunk is expanded to desired_state.raw_15d before UDP control.",
                },
            },
            "infer_timing": infer_timing,
            "rtc_info": rtc_info,
            "action_steps": action_steps or [],
        }
        if isinstance(ik, dict) and ik:
            row["ik"] = ik
        self._write_jsonl(row)


def build_action_step_eval_record(
    *,
    chunk_step_index: int,
    action_model: Any,
    desired_state_15d: Any,
    pre_send_state: Any,
    response_state: Any,
    feedback: Dict[str, Any],
    send_duration_sec: float,
    tracking_error_l2: Optional[float],
    tracking_error_max: Optional[float],
    pre_send_api_state: Optional[Dict[str, Any]] = None,
    response_api_state: Optional[Dict[str, Any]] = None,
    gripper_udp_pre: Optional[Dict[str, Any]] = None,
    gripper_udp_post: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    pre_udp = normalize_gripper_udp_feedback(gripper_udp_pre)
    post_udp = normalize_gripper_udp_feedback(gripper_udp_post)
    return {
        "chunk_step_index": int(chunk_step_index),
        "network_action_step": _array_to_list(action_model),
        "current_state": state_vector_to_dict(pre_send_state, pre_send_api_state),
        "desired_state": state_vector_to_dict(desired_state_15d),
        "response_state": state_vector_to_dict(response_state, response_api_state),
        "gripper_udp_pre": pre_udp,
        "gripper_udp_post": post_udp,
        "state_timestamps": {
            "pre_send": feedback.get("pre_send_state_timestamp"),
            "response": feedback.get("response_state_timestamp"),
        },
        "feedback": {
            "state_feedback_received": feedback.get("state_feedback_received"),
            "robot_response_time_sec": feedback.get("robot_response_time_sec"),
            "state_feedback_timeout_sec": feedback.get("state_feedback_timeout_sec"),
            "action_send_duration_sec": send_duration_sec,
            "response_tracking_error_l2": tracking_error_l2,
            "response_tracking_error_max": tracking_error_max,
        },
    }
