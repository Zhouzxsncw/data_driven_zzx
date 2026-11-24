#!/usr/bin/env python3
"""查询 RealSense 相机支持的采集分辨率（默认 mid / devices[1]）。

用法示例::

  # 从 infer 配置读 mid 相机 serial，列出 RGB 支持分辨率
  python3 tools/camera/query_camera_resolutions.py \\
      --config infer/config/lerobot/act_left_arm_direct.yaml

  # 从 data_collect 配置读 observation.cam_mid
  python3 tools/camera/query_camera_resolutions.py \\
      --config data_collect/config/left_arm_gripper_2cam.yaml

  # 直接指定 serial
  python3 tools/query_camera_resolutions.py --serial f1420947

  # 列出本机已连接的 RealSense
  python3 tools/query_camera_resolutions.py --list-devices

  # 同时看 depth 流
  python3 tools/query_camera_resolutions.py --config infer/config/lerobot/act_left_arm_direct.yaml --stream all
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class StreamProfileRow:
    stream: str
    width: int
    height: int
    fps: int
    format: str

    @property
    def label(self) -> str:
        return f"{self.width}x{self.height} @ {self.fps}fps ({self.format})"


def _parse_policy_camera_keys_from_yaml(raw: Dict[str, Any]) -> Optional[List[str]]:
    policy = raw.get("policy")
    if not isinstance(policy, dict):
        return None
    keys = policy.get("camera_keys")
    if keys is None:
        return None
    if not isinstance(keys, list):
        return None
    parsed = [str(k).strip() for k in keys if str(k).strip()]
    return parsed


def _camera_key_for_query(camera: str) -> str:
    key_map = {
        "mid": "observation.cam_mid",
        "cam_mid": "observation.cam_mid",
        "left": "observation.cam_left",
        "cam_left": "observation.cam_left",
    }
    return key_map.get(camera, camera)


def _load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"配置根节点须为 dict: {path}")
    return data


def _device_serial_from_infer_config(raw: Dict[str, Any], camera: str) -> Tuple[str, str]:
    devices = (raw.get("camera_sender") or {}).get("devices") or []
    if not devices:
        raise ValueError("infer 配置缺少 camera_sender.devices")

    target_key = _camera_key_for_query(camera)
    for idx, dev in enumerate(devices):
        if not isinstance(dev, dict):
            continue
        dev_key = str(dev.get("key", "")).strip()
        if dev_key and (dev_key == target_key or dev_key == camera):
            serial = str(dev.get("serial", "")).strip()
            if not serial:
                raise ValueError(f"devices[{idx}] 未配置 serial")
            label = dev_key or f"devices[{idx}]"
            return serial, label

    if camera in ("mid", "cam_mid", "observation.cam_mid"):
        idx = 1
        label = "observation.cam_mid (devices[1])"
    elif camera in ("left", "cam_left", "observation.cam_left"):
        idx = 0
        label = "observation.cam_left (devices[0])"
    else:
        idx = int(camera)
        label = f"devices[{idx}]"

    if idx < 0 or idx >= len(devices):
        raise ValueError(f"camera_sender.devices 只有 {len(devices)} 路，无法取 index={idx}")

    dev = devices[idx]
    if not isinstance(dev, dict):
        raise ValueError(f"devices[{idx}] 格式无效")
    serial = str(dev.get("serial", "")).strip()
    if not serial:
        raise ValueError(f"devices[{idx}] 未配置 serial")
    return serial, label


def _device_serial_from_collect_config(raw: Dict[str, Any], camera: str) -> Tuple[str, str]:
    devices = (raw.get("cameras") or {}).get("devices") or []
    if not devices:
        raise ValueError("data_collect 配置缺少 cameras.devices")

    key_map = {
        "mid": "observation.cam_mid",
        "cam_mid": "observation.cam_mid",
        "left": "observation.cam_left",
        "cam_left": "observation.cam_left",
    }
    target_key = key_map.get(camera, camera)

    for dev in devices:
        if not isinstance(dev, dict):
            continue
        if str(dev.get("key", "")).strip() == target_key:
            serial = str(dev.get("serial", "")).strip()
            if not serial:
                raise ValueError(f"{target_key} 未配置 serial")
            return serial, target_key

    if camera.isdigit():
        idx = int(camera)
        if 0 <= idx < len(devices):
            dev = devices[idx]
            serial = str(dev.get("serial", "")).strip()
            key = str(dev.get("key", f"devices[{idx}]"))
            if not serial:
                raise ValueError(f"devices[{idx}] 未配置 serial")
            return serial, key

    raise ValueError(f"cameras.devices 中未找到 {target_key!r}")


def resolve_camera_serial(
    config_path: Path,
    *,
    camera: str = "mid",
) -> Tuple[str, str]:
    raw = _load_yaml(config_path)
    if "camera_sender" in raw:
        return _device_serial_from_infer_config(raw, camera)
    if "cameras" in raw:
        return _device_serial_from_collect_config(raw, camera)
    raise ValueError(
        f"无法识别配置类型（需 camera_sender 或 cameras）: {config_path}"
    )


def _stream_name(stream_type: Any) -> str:
    import pyrealsense2 as rs

    mapping = {
        rs.stream.color: "color",
        rs.stream.depth: "depth",
        rs.stream.infrared: "infrared",
    }
    return mapping.get(stream_type, str(stream_type))


def _format_name(fmt: Any) -> str:
    import pyrealsense2 as rs

    try:
        return str(fmt).split(".")[-1]
    except Exception:
        return str(fmt)


def _wanted_stream_types(stream: str) -> set:
    import pyrealsense2 as rs

    if stream == "color":
        return {rs.stream.color}
    if stream == "depth":
        return {rs.stream.depth}
    return {rs.stream.color, rs.stream.depth, rs.stream.infrared}


def list_connected_devices() -> List[Dict[str, str]]:
    import pyrealsense2 as rs

    ctx = rs.context()
    rows: List[Dict[str, str]] = []
    for dev in ctx.query_devices():
        rows.append(
            {
                "name": dev.get_info(rs.camera_info.name),
                "serial": dev.get_info(rs.camera_info.serial_number),
                "product_line": dev.get_info(rs.camera_info.product_line),
                "firmware": dev.get_info(rs.camera_info.firmware_version),
            }
        )
    return rows


def query_profiles_for_serial(
    serial: str,
    *,
    stream: str = "color",
) -> List[StreamProfileRow]:
    import pyrealsense2 as rs

    ctx = rs.context()
    device = None
    for dev in ctx.query_devices():
        if dev.get_info(rs.camera_info.serial_number) == serial:
            device = dev
            break
    if device is None:
        connected = [
            d.get_info(rs.camera_info.serial_number) for d in ctx.query_devices()
        ]
        raise RuntimeError(
            f"未找到 serial={serial!r} 的 RealSense。"
            f" 当前已连接: {connected or '（无）'}"
        )

    wanted = _wanted_stream_types(stream)
    rows: List[StreamProfileRow] = []
    for sensor in device.query_sensors():
        for profile in sensor.get_stream_profiles():
            if profile.stream_type() not in wanted:
                continue
            try:
                vp = profile.as_video_stream_profile()
            except Exception:
                continue
            rows.append(
                StreamProfileRow(
                    stream=_stream_name(profile.stream_type()),
                    width=int(vp.width()),
                    height=int(vp.height()),
                    fps=int(vp.fps()),
                    format=_format_name(vp.format()),
                )
            )

    # 去重后按分辨率、帧率排序
    unique = sorted(
        {(r.stream, r.width, r.height, r.fps, r.format): r for r in rows}.values(),
        key=lambda r: (r.stream, r.width * r.height, r.fps, r.height, r.width),
        reverse=True,
    )
    return unique


def _group_by_resolution(rows: Sequence[StreamProfileRow]) -> List[Tuple[str, List[str]]]:
    grouped: Dict[Tuple[str, int, int], List[str]] = {}
    for row in rows:
        key = (row.stream, row.width, row.height)
        grouped.setdefault(key, []).append(f"{row.fps}fps ({row.format})")
    out: List[Tuple[str, List[str]]] = []
    for (stream_name, width, height), fps_list in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], item[0][1] * item[0][2], item[0][2], item[0][1]),
        reverse=True,
    ):
        uniq_fps = sorted(set(fps_list))
        out.append((f"{stream_name} {width}x{height}", uniq_fps))
    return out


def _print_human(
    *,
    serial: str,
    camera_label: str,
    rows: Sequence[StreamProfileRow],
    configured: Optional[Tuple[int, int, int]] = None,
) -> None:
    print(f"相机: {camera_label}")
    print(f"serial: {serial}")
    if configured is not None:
        w, h, fps = configured
        print(f"配置中当前值: {w}x{h} @ {fps}fps")
    print()

    if not rows:
        print("（无匹配流配置）")
        return

    print("支持的分辨率（按像素从高到低）:")
    for res_label, fps_list in _group_by_resolution(rows):
        print(f"  {res_label}")
        for item in fps_list:
            print(f"    - {item}")


def _parse_configured_resolution(
    config_path: Optional[Path],
    camera: str,
) -> Optional[Tuple[int, int, int]]:
    if config_path is None:
        return None
    raw = _load_yaml(config_path)
    if "camera_sender" in raw:
        devices = (raw.get("camera_sender") or {}).get("devices") or []
        idx = 1 if camera in ("mid", "cam_mid", "observation.cam_mid") else 0
        if camera.isdigit():
            idx = int(camera)
        if idx >= len(devices):
            return None
        dev = devices[idx]
        w, h, fps = dev.get("width"), dev.get("height"), dev.get("fps", 30)
        if w is None or h is None:
            return None
        return int(w), int(h), int(fps)

    devices = (raw.get("cameras") or {}).get("devices") or []
    key_map = {
        "mid": "observation.cam_mid",
        "cam_mid": "observation.cam_mid",
        "left": "observation.cam_left",
        "cam_left": "observation.cam_left",
    }
    target_key = key_map.get(camera, camera)
    for dev in devices:
        if str(dev.get("key", "")).strip() == target_key:
            return int(dev["width"]), int(dev["height"]), int(dev.get("fps", 30))
    return None


def _resolve_color_target(
    config_path: Optional[Path],
    camera: str,
) -> Optional[Tuple[int, int, int]]:
    """返回训练/配置目标 color 分辨率 (width, height, fps)。"""
    if config_path is None:
        return None

    configured = _parse_configured_resolution(config_path, camera)
    if configured is not None:
        return configured

    raw = _load_yaml(config_path)
    policy = raw.get("policy") if isinstance(raw.get("policy"), dict) else None
    if policy is None:
        return None

    ckpt_dir = policy.get("ckpt_dir")
    policy_type = policy.get("type", "pi05")
    if not ckpt_dir:
        return None

    from tools.camera.checkpoint_camera_resolution import (
        read_ckpt_config,
        resolve_checkpoint_camera_resolutions,
    )

    ckpt_cfg = read_ckpt_config(Path(str(ckpt_dir)).expanduser())
    if not ckpt_cfg:
        return None

    resolutions = resolve_checkpoint_camera_resolutions(
        ckpt_cfg,
        str(policy_type),
        policy_camera_keys=_parse_policy_camera_keys_from_yaml(raw),
    )
    if not resolutions:
        return None

    cam_keys = [item[0] for item in resolutions]
    if camera.isdigit():
        cam_idx = int(camera)
    else:
        target_key = _camera_key_for_query(camera)
        if target_key in cam_keys:
            cam_idx = cam_keys.index(target_key)
        else:
            # 兼容 observation.images.* 等同 slot 后缀
            suffix = ".cam_mid" if "mid" in camera else ".cam_left"
            matches = [i for i, k in enumerate(cam_keys) if k.endswith(suffix)]
            cam_idx = matches[0] if matches else (1 if "mid" in camera else 0)
    if cam_idx >= len(resolutions):
        return None
    _, height, width = resolutions[cam_idx]
    if height is None or width is None:
        return None

    devices = (raw.get("camera_sender") or {}).get("devices") or []
    fps = 30
    if cam_idx < len(devices) and isinstance(devices[cam_idx], dict):
        fps = int(devices[cam_idx].get("fps", 30))
    return int(width), int(height), fps


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="查询 RealSense 相机支持的采集分辨率（默认 mid 相机）",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="infer 或 data_collect YAML；自动读取 mid 相机 serial",
    )
    parser.add_argument(
        "--serial",
        help="直接指定 RealSense serial（优先级高于 --config）",
    )
    parser.add_argument(
        "--camera",
        default="mid",
        help="配置中的相机：mid / left / 0 / 1 / observation.cam_mid（默认 mid）",
    )
    parser.add_argument(
        "--stream",
        choices=("color", "depth", "all"),
        default="color",
        help="查询的流类型（默认 color / RGB）",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="列出本机已连接的 RealSense 后退出",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="以 JSON 输出",
    )
    parser.add_argument(
        "--recommend-depth",
        action="store_true",
        help="根据配置/checkpoint 的 color 分辨率，推荐最接近的 depth profile",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        import pyrealsense2 as rs  # noqa: F401
    except ImportError:
        print(
            "需要安装 pyrealsense2。建议在 lerobot 环境运行:\n"
            "  conda activate lerobot\n"
            "  python3 tools/query_camera_resolutions.py --list-devices",
            file=sys.stderr,
        )
        return 1

    if args.list_devices:
        devices = list_connected_devices()
        if args.json:
            print(json.dumps(devices, ensure_ascii=False, indent=2))
        else:
            if not devices:
                print("未检测到已连接的 RealSense 相机")
            for idx, dev in enumerate(devices):
                print(
                    f"[{idx}] {dev['name']} | serial={dev['serial']} | "
                    f"{dev['product_line']} | firmware={dev['firmware']}"
                )
        return 0

    serial = str(args.serial).strip() if args.serial else ""
    camera_label = "manual"
    if not serial:
        if args.config is None:
            parser.error("请指定 --config 或 --serial（或用 --list-devices）")
        config_path = args.config if args.config.is_absolute() else _REPO_ROOT / args.config
        if not config_path.is_file():
            raise SystemExit(f"配置文件不存在: {config_path}")
        serial, camera_label = resolve_camera_serial(config_path, camera=args.camera)
        configured = _parse_configured_resolution(config_path, args.camera)
    else:
        configured = _parse_configured_resolution(args.config, args.camera) if args.config else None

    config_path = None
    if args.config is not None:
        config_path = args.config if args.config.is_absolute() else _REPO_ROOT / args.config

    if args.recommend_depth:
        color_target = _resolve_color_target(config_path, args.camera)
        if color_target is None:
            raise SystemExit(
                "无法确定 color 训练分辨率：请在配置中写 width/height，"
                "或确保 policy.ckpt_dir 有效"
            )
        cw, ch, cfps = color_target
        from tools.camera.realsense_profile_utils import (
            enumerate_depth_profiles,
            pick_depth_profiles_for_color,
            recommend_depth_profile,
        )

        recommended = recommend_depth_profile(serial, cw, ch, cfps)
        ranked = pick_depth_profiles_for_color(
            enumerate_depth_profiles(serial),
            cw,
            ch,
            cfps,
        )
        if args.json:
            print(
                json.dumps(
                    {
                        "serial": serial,
                        "camera": camera_label,
                        "color_target": {"width": cw, "height": ch, "fps": cfps},
                        "recommended_depth": (
                            {
                                "width": recommended.width,
                                "height": recommended.height,
                                "fps": recommended.fps,
                                "format": recommended.format,
                            }
                            if recommended is not None
                            else None
                        ),
                        "ranked_depth": [
                            {
                                "width": p.width,
                                "height": p.height,
                                "fps": p.fps,
                                "format": p.format,
                            }
                            for p in ranked[:5]
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(f"相机: {camera_label}")
            print(f"serial: {serial}")
            print(f"训练 color 目标: {cw}x{ch} @ {cfps}fps")
            if recommended is None:
                print("（无可用 depth profile）")
            else:
                print(
                    f"推荐 depth: {recommended.label}"
                    "（最接近训练 color，推理时 rs.align 对齐）"
                )
                if len(ranked) > 1:
                    print("备选 depth:")
                    for p in ranked[1:5]:
                        print(f"  - {p.label}")
        return 0

    rows = query_profiles_for_serial(serial, stream=args.stream)

    if args.json:
        payload = {
            "serial": serial,
            "camera": camera_label,
            "stream": args.stream,
            "configured": (
                {
                    "width": configured[0],
                    "height": configured[1],
                    "fps": configured[2],
                }
                if configured is not None
                else None
            ),
            "profiles": [
                {
                    "stream": r.stream,
                    "width": r.width,
                    "height": r.height,
                    "fps": r.fps,
                    "format": r.format,
                }
                for r in rows
            ],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        _print_human(
            serial=serial,
            camera_label=camera_label,
            rows=rows,
            configured=configured,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
