"""YAML 配置加载与校验。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from data_collect.modalities import (
    GripperValueFilterConfig,
    ModalityGroupConfig,
    ModalityKind,
    ModalityToggle,
    StreamKind,
)


@dataclass
class DatasetConfig:
    repo_id: str
    name_prefix: str
    root: Path
    task: str
    fps: int
    robot_type: Optional[str]
    table_height: Optional[int] = None  # 桌子高度(cm)，写入数据集路径


@dataclass
class NetworkConfig:
    high_host: str
    low_host: str
    ports: Dict[str, int]


@dataclass
class CameraDeviceConfig:
    key: str
    serial: str
    width: int
    height: int
    fps: int
    # 是否采集该相机的 depth。建议只给 L515 对应设备设 true。
    save_depth: bool = False
    # depth 流分辨率/帧率；不填时默认跟 color 相同。
    depth_width: Optional[int] = None
    depth_height: Optional[int] = None
    depth_fps: Optional[int] = None


@dataclass
class CamerasConfig:
    enabled: bool
    use_host_timestamp: bool
    get_timeout_sec: float
    devices: List[CameraDeviceConfig]
    enable_display: bool = True  # 是否在采集时实时显示相机画面，默认开启
    # depth 输出开关。开启后会在数据集根目录创建 depth/，与 videos/ 同级。
    save_depth: bool = False
    depth_dir_name: str = "depth"
    depth_format: str = "png"
    # 若非空，只保存这些 camera key 的 depth；否则保存 devices[].save_depth=true 的设备。
    depth_devices: List[str] = None
    # 将 depth 通过 RealSense 标定外参重投影到 RGB/color 像素坐标系。
    align_depth_to_color: bool = True
    # 写入 LeRobot meta/info.json 的深度字段前缀。
    depth_feature_prefix: str = "observation.depth"


@dataclass
class SyncConfig:
    max_time_diff: Optional[float]
    sync_delay: float
    buffer_size: int
    prefill_sec: float
    min_anchor_buffer: int
    cleanup_interval_frames: int
    cleanup_keep_sec: float
    clock_offset: Dict[str, float]

    def resolved_max_time_diff(self, fps: int) -> float:
        if self.max_time_diff is not None:
            return float(self.max_time_diff)
        return max(0.1, 2.0 / fps)

    def offset_for(self, stream: StreamKind) -> float:
        return float(self.clock_offset.get(stream.value, 0.0))


@dataclass
class EpisodeConfig:
    image_writer_processes: int
    image_writer_threads: int
    prompt_save: bool
    prompt_continue: bool


@dataclass
class QaConfig:
    """
    采后质量评估（evaluation.py）钩子配置。

    evaluation.py 在本仓库为离线 CLI 工具，默认不在采集流程中自动调用；
    将 enabled 设为 true 可在每次 save_episode 成功后评估刚保存的那一集。
    """

    enabled: bool = False
    threshold: float = 0.65
    warn_if_below_threshold: bool = True
    save_report: bool = True
    report_subdir: str = "qa_reports"
    report_path: Optional[Path] = None  # 非 null 时覆盖 report_subdir


@dataclass
class RealtimeGripperPlotConfig:
    """实时夹爪曲线显示配置。

    enabled=true 时，run_collect.py 启动后使用一个 OpenCV 窗口显示左右手夹爪实时曲线；
    enabled=false 时，完全不启动绘图，采集流程保持原样。
    """

    enabled: bool = True
    max_points: int = 3000
    update_interval_sec: float = 0.0


@dataclass
class RobotArmControlConfig:
    """外骨骼控制与 RM75 实际状态采集配置。"""

    enabled: bool = False
    sdk_path: str = "~/RM_API2-main"
    robot_r_host: str = "192.168.1.17"
    robot_l_host: str = "192.168.1.18"
    port_r: int = 8080
    port_l: int = 8080
    control_hz: float = 30.0
    state_hz: float = 30.0
    gripper_host: str = "192.168.1.13"
    gripper_port: int = 30015
    # 外骨骼夹爪命令防抖。1.0 表示全开，短促周期脉冲不会驱动真实夹爪。
    gripper_filter_enabled: bool = True
    gripper_open_value: float = 1.0
    gripper_open_deadband: float = 0.06
    gripper_change_threshold: float = 0.08
    gripper_stable_samples: int = 2
    gripper_ema_alpha: float = 1.0
    # Episode 启动时经 30015 初始化左右真机夹爪开口比（不 movej 机械臂）
    init_grippers_on_start: bool = True
    gripper_initial_position: Optional[List[float]] = None


@dataclass
class LoggingConfig:
    level: str
    log_sync_every_n_frames: int
    log_file: Optional[Path]


@dataclass
class CollectConfig:
    """数据采集完整配置。"""

    dataset: DatasetConfig
    network: NetworkConfig
    state_modalities: ModalityGroupConfig
    action_modalities: ModalityGroupConfig
    gripper_source: int
    gripper_value_filter: GripperValueFilterConfig
    cameras: CamerasConfig
    sync: SyncConfig
    episode: EpisodeConfig
    qa: QaConfig
    realtime_gripper_plot: RealtimeGripperPlotConfig
    robot_arm: RobotArmControlConfig
    logging: LoggingConfig
    config_path: Path

    def resolve_repo_id(self) -> str:
        if self.dataset.repo_id != "auto":
            return self.dataset.repo_id
        ts = time.strftime("%Y%m%d_%H%M%S")
        prefix = self.dataset.name_prefix
        if self.dataset.table_height is not None:
            prefix = f"{prefix}_h{self.dataset.table_height}"
        return f"{prefix}_{ts}"

    def dataset_root(self, repo_id: str) -> Path:
        return self.dataset.root / repo_id


def _parse_modality_group(raw: Dict[str, Any]) -> ModalityGroupConfig:
    def one(name: str) -> ModalityToggle:
        item = raw.get(name, {})
        return ModalityToggle(
            enabled=bool(item.get("enabled", False)),
            required=bool(item.get("required", True)),
            fallback=str(item.get("fallback", "drop_frame")),
        )

    return ModalityGroupConfig(
        left_arm=one("left_arm"),
        right_arm=one("right_arm"),
        left_gripper=one("left_gripper"),
        right_gripper=one("right_gripper"),
        right_hand=one("right_hand"),
    )


def _require_section(data: Dict[str, Any], key: str) -> Dict[str, Any]:
    if key not in data:
        raise ValueError(f"配置缺少必填节: '{key}'")
    return data[key]


def _merge_sync_clock_offset(sync: Dict[str, Any]) -> Dict[str, float]:
    from common.sync_calibration import load_sync_calibration, merge_clock_offsets

    base = {str(k): float(v) for k, v in sync.get("clock_offset", {}).items()}
    cal_file = sync.get("calibration_file")
    shared = load_sync_calibration(cal_file) if cal_file else load_sync_calibration()
    return merge_clock_offsets(base, shared)


def load_collect_config(path: str | Path) -> CollectConfig:
    """从 YAML 加载 CollectConfig。"""
    config_path = Path(path).resolve()
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    ds = _require_section(raw, "dataset")
    net = _require_section(raw, "network")
    mods = _require_section(raw, "modalities")
    gripper_value_filter_raw = mods.get("gripper_value_filter", {}) or {}
    cams = raw.get("cameras", {"enabled": False, "devices": []})
    sync = _require_section(raw, "sync")
    merged_clock_offset = _merge_sync_clock_offset(sync)
    ep = raw.get("episode", {})
    qa = raw.get("qa", {})
    # 兼容两个名字：推荐使用 realtime_gripper_plot；旧/简称 gripper_plot 也可用。
    realtime_plot = raw.get("realtime_gripper_plot", raw.get("gripper_plot", {}))
    robot_arm_raw = raw.get("robot_arm", {})
    log = raw.get("logging", {})

    ports = net.get("ports", {})
    for required_port in ("robot_action",):
        if required_port not in ports:
            raise ValueError(f"network.ports 缺少 {required_port}")

    devices = []
    for dev in cams.get("devices", []):
        devices.append(
            CameraDeviceConfig(
                key=str(dev["key"]),
                serial=str(dev["serial"]),
                width=int(dev["width"]),
                height=int(dev["height"]),
                fps=int(dev.get("fps", 30)),
                save_depth=bool(dev.get("save_depth", dev.get("depth", False))),
                depth_width=int(dev["depth_width"]) if dev.get("depth_width") is not None else None,
                depth_height=int(dev["depth_height"]) if dev.get("depth_height") is not None else None,
                depth_fps=int(dev["depth_fps"]) if dev.get("depth_fps") is not None else None,
            )
        )

    if cams.get("enabled", True) and not devices:
        raise ValueError("cameras.enabled=true 但未配置 cameras.devices")

    depth_raw = cams.get("depth", {}) or {}
    depth_devices = [str(x) for x in depth_raw.get("devices", [])]
    global_save_depth = bool(depth_raw.get("enabled", cams.get("save_depth", False)))
    if depth_devices:
        # 配置了 cameras.depth.devices 时，自动把对应 device 标记为 save_depth。
        depth_device_set = set(depth_devices)
        for dev in devices:
            if dev.key in depth_device_set:
                dev.save_depth = True
    save_depth = global_save_depth or any(d.save_depth for d in devices)

    return CollectConfig(
        dataset=DatasetConfig(
            repo_id=str(ds.get("repo_id", "auto")),
            name_prefix=str(ds.get("name_prefix", "collect")),
            root=Path(ds["root"]).expanduser(),
            task=str(ds["task"]),
            fps=int(ds.get("fps", 15)),
            robot_type=ds.get("robot_type"),
            table_height=ds.get("table_height"),
        ),
        network=NetworkConfig(
            high_host=str(net["high_host"]),
            low_host=str(net.get("low_host", "192.168.1.13")),
            ports={k: int(v) for k, v in ports.items()},
        ),
        state_modalities=_parse_modality_group(mods.get("state", {})),
        action_modalities=_parse_modality_group(mods.get("action", {})),
        gripper_source=int(mods.get("gripper_source", 0)),
        gripper_value_filter=GripperValueFilterConfig(
            enabled=bool(gripper_value_filter_raw.get("enabled", True)),
            open_value=float(gripper_value_filter_raw.get("open_value", 1.0)),
            open_deadband=float(gripper_value_filter_raw.get("open_deadband", 0.08)),
            close_threshold=float(gripper_value_filter_raw.get("close_threshold", 0.85)),
            stable_samples=int(gripper_value_filter_raw.get("stable_samples", 2)),
            output_mode=str(gripper_value_filter_raw.get("output_mode", "binary")),
            ema_alpha=float(gripper_value_filter_raw.get("ema_alpha", 1.0)),
        ),
        cameras=CamerasConfig(
            enabled=bool(cams.get("enabled", True)),
            use_host_timestamp=bool(cams.get("use_host_timestamp", True)),
            get_timeout_sec=float(cams.get("get_timeout_sec", 3.0)),
            devices=devices,
            enable_display=bool(cams.get("enable_display", True)),
            save_depth=save_depth,
            depth_dir_name=str(depth_raw.get("dir_name", cams.get("depth_dir_name", "depth"))),
            depth_format=str(depth_raw.get("format", cams.get("depth_format", "png"))).lower(),
            depth_devices=depth_devices,
            align_depth_to_color=bool(depth_raw.get("align_to_color", True)),
            depth_feature_prefix=str(depth_raw.get("feature_prefix", "observation.depths")).rstrip("."),
        ),
        sync=SyncConfig(
            max_time_diff=sync.get("max_time_diff"),
            sync_delay=float(sync.get("sync_delay", 0.05)),
            buffer_size=int(sync.get("buffer_size", 200)),
            prefill_sec=float(sync.get("prefill_sec", 2.0)),
            min_anchor_buffer=int(sync.get("min_anchor_buffer", 5)),
            cleanup_interval_frames=int(sync.get("cleanup_interval_frames", 15)),
            cleanup_keep_sec=float(sync.get("cleanup_keep_sec", 1.0)),
            clock_offset=merged_clock_offset,
        ),
        episode=EpisodeConfig(
            image_writer_processes=int(ep.get("image_writer_processes", 0)),
            image_writer_threads=int(ep.get("image_writer_threads", 12)),
            prompt_save=bool(ep.get("prompt_save", True)),
            prompt_continue=bool(ep.get("prompt_continue", True)),
        ),
        qa=QaConfig(
            enabled=bool(qa.get("enabled", False)),
            threshold=float(qa.get("threshold", 0.65)),
            warn_if_below_threshold=bool(qa.get("warn_if_below_threshold", True)),
            save_report=bool(qa.get("save_report", True)),
            report_subdir=str(qa.get("report_subdir", "qa_reports")),
            report_path=Path(qa["report_path"]).expanduser() if qa.get("report_path") else None,
        ),
        realtime_gripper_plot=RealtimeGripperPlotConfig(
            enabled=bool(realtime_plot.get("enabled", True)),
            max_points=int(realtime_plot.get("max_points", 3000)),
            update_interval_sec=float(realtime_plot.get("update_interval_sec", 0.0)),
        ),
        robot_arm=RobotArmControlConfig(
            enabled=bool(robot_arm_raw.get("enabled", False)),
            sdk_path=str(robot_arm_raw.get("sdk_path", "~/RM_API2-main")),
            robot_r_host=str(robot_arm_raw.get("robot_r_host", "192.168.1.17")),
            robot_l_host=str(robot_arm_raw.get("robot_l_host", "192.168.1.18")),
            port_r=int(robot_arm_raw.get("port_r", 8080)),
            port_l=int(robot_arm_raw.get("port_l", 8080)),
            control_hz=float(robot_arm_raw.get("control_hz", 30.0)),
            state_hz=float(robot_arm_raw.get("state_hz", 30.0)),
            gripper_host=str(robot_arm_raw.get("gripper_host", "192.168.1.13")),
            gripper_port=int(robot_arm_raw.get("gripper_port", 30015)),
            gripper_filter_enabled=bool((robot_arm_raw.get("gripper_filter", {}) or {}).get("enabled", True)),
            gripper_open_value=float((robot_arm_raw.get("gripper_filter", {}) or {}).get("open_value", 1.0)),
            gripper_open_deadband=float((robot_arm_raw.get("gripper_filter", {}) or {}).get("open_deadband", 0.06)),
            gripper_change_threshold=float((robot_arm_raw.get("gripper_filter", {}) or {}).get("change_threshold", 0.08)),
            gripper_stable_samples=int((robot_arm_raw.get("gripper_filter", {}) or {}).get("stable_samples", 2)),
            gripper_ema_alpha=float((robot_arm_raw.get("gripper_filter", {}) or {}).get("ema_alpha", 1.0)),
            init_grippers_on_start=bool(
                robot_arm_raw.get(
                    "init_grippers_on_start",
                    robot_arm_raw.get("reset_to_init_on_start", True),
                )
            ),
            gripper_initial_position=(
                [float(x) for x in robot_arm_raw["gripper_initial_position"]]
                if robot_arm_raw.get("gripper_initial_position") is not None
                else [0.7, 0.7]
            ),
        ),
        logging=LoggingConfig(
            level=str(log.get("level", "INFO")),
            log_sync_every_n_frames=int(log.get("log_sync_every_n_frames", 30)),
            log_file=Path(log["log_file"]) if log.get("log_file") else None,
        ),
        config_path=config_path,
    )


def validate_collect_config(cfg: CollectConfig) -> None:
    """启动前校验：至少一个 state/action 模态、相机 key 唯一。"""
    if not cfg.state_modalities.iter_enabled():
        raise ValueError("modalities.state 至少启用一个模态")
    if not cfg.action_modalities.iter_enabled():
        raise ValueError("modalities.action 至少启用一个模态")

    if cfg.gripper_source not in (0, 1):
        raise ValueError("modalities.gripper_source 只能为 0（真实夹爪）或 1（外骨骼夹爪）")
    gvf = cfg.gripper_value_filter
    if not 0.0 <= gvf.open_value <= 1.0:
        raise ValueError("modalities.gripper_value_filter.open_value 必须在 [0,1]")
    if not 0.0 <= gvf.open_deadband < 1.0:
        raise ValueError("modalities.gripper_value_filter.open_deadband 必须在 [0,1)")
    if not 0.0 <= gvf.close_threshold <= 1.0:
        raise ValueError("modalities.gripper_value_filter.close_threshold 必须在 [0,1]")
    if gvf.stable_samples < 1:
        raise ValueError("modalities.gripper_value_filter.stable_samples 必须 >= 1")
    if str(gvf.output_mode).lower() not in ("binary", "continuous"):
        raise ValueError("modalities.gripper_value_filter.output_mode 只能是 binary 或 continuous")
    if not 0.0 < gvf.ema_alpha <= 1.0:
        raise ValueError("modalities.gripper_value_filter.ema_alpha 必须在 (0,1]")
    if not 0.0 <= cfg.robot_arm.gripper_open_value <= 1.0:
        raise ValueError("robot_arm.gripper_filter.open_value 必须在 [0,1]")
    if not 0.0 <= cfg.robot_arm.gripper_open_deadband < 1.0:
        raise ValueError("robot_arm.gripper_filter.open_deadband 必须在 [0,1)")
    if cfg.robot_arm.gripper_stable_samples < 1:
        raise ValueError("robot_arm.gripper_filter.stable_samples 必须 >= 1")
    if not 0.0 < cfg.robot_arm.gripper_ema_alpha <= 1.0:
        raise ValueError("robot_arm.gripper_filter.ema_alpha 必须在 (0,1]")
    if cfg.robot_arm.gripper_initial_position is not None:
        if len(cfg.robot_arm.gripper_initial_position) not in (1, 2):
            raise ValueError("robot_arm.gripper_initial_position 需要 1 或 2 维")
        if any(not 0.0 <= float(v) <= 1.0 for v in cfg.robot_arm.gripper_initial_position):
            raise ValueError("robot_arm.gripper_initial_position 各维必须在 [0,1]")
    if any(k in cfg.state_modalities.iter_enabled() for k in (ModalityKind.LEFT_ARM, ModalityKind.RIGHT_ARM)) and not cfg.robot_arm.enabled:
        raise ValueError("observation.state 的机械臂关节来自 RM75；启用 state.left_arm/right_arm 时必须设置 robot_arm.enabled=true")

    keys = [d.key for d in cfg.cameras.devices]
    if len(keys) != len(set(keys)):
        raise ValueError(f"cameras.devices 存在重复 key: {keys}")

    if cfg.cameras.save_depth:
        if cfg.cameras.depth_format != "png":
            raise ValueError("当前 depth_format 只支持 png，因为需要保存 L515 16-bit 深度值")
        depth_keys = [d.key for d in cfg.cameras.devices if d.save_depth]
        if not depth_keys:
            raise ValueError("cameras.depth.enabled=true，但没有任何 cameras.devices[].save_depth=true 或 cameras.depth.devices")
        if not cfg.cameras.align_depth_to_color:
            raise ValueError("当前 LeRobot depth 字段要求与 RGB 像素对齐，请设置 cameras.depth.align_to_color=true")
        if not cfg.cameras.depth_feature_prefix:
            raise ValueError("cameras.depth.feature_prefix 不能为空")

    for group in (cfg.state_modalities, cfg.action_modalities):
        for kind in ModalityKind:
            toggle = group.get_toggle(kind)
            if toggle.fallback not in ("drop_frame", "hold_last"):
                raise ValueError(f"未知 fallback 策略: {toggle.fallback}")
