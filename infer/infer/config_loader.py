"""推理 YAML 配置加载。"""

from __future__ import annotations

import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import yaml

from infer.core.types import EngineMode, PolicyType
from infer.ui.pedal_input import PedalKeyBinding, user_hotkey_to_pynput
from infer.io.robot_init_pose import STANDARD_INIT_JOINT_17D_DEG

# 旧扁平 config/*.yaml → config/{lerobot|aloha}/*.yaml（仅文件名匹配时生效）
_LEGACY_FLAT_CONFIG_REMAP: Dict[str, str] = {
    "pi05_left_arm_direct.yaml": "lerobot/pi05_left_arm_direct.yaml",
    "pi05_right_arm_direct.yaml": "lerobot/pi05_right_arm_direct.yaml",
    "pi05_dual_arm_gripper_direct.yaml": "lerobot/pi05_dual_arm_gripper_direct.yaml",
    "pi05_left_arm.yaml": "lerobot/pi05_left_arm.yaml",
    "pi05_openpi_left_arm_direct.yaml": "lerobot/pi05_openpi_left_arm_direct.yaml",
    "pi05_openpi_dual_arm_gripper_direct.yaml": "lerobot/pi05_openpi_dual_arm_gripper_direct.yaml",
    "act_left_arm_direct.yaml": "lerobot/act_left_arm_direct.yaml",
    "act_left_arm.yaml": "lerobot/act_left_arm.yaml",
    "dp_left_arm_direct.yaml": "lerobot/dp_left_arm_direct.yaml",
    "ablation_nas.yaml": "lerobot/ablation_nas.yaml",
    "act_aloha_left_arm_direct.yaml": "aloha/act_aloha_left_arm_direct.yaml",
    "act_aloha_dual_arm_gripper_direct.yaml": "aloha/act_aloha_dual_arm_gripper_direct.yaml",
}

_CONFIG_ROOT = Path(__file__).resolve().parent / "config"

# Pi0.5 Real-Time Chunking（推理专用；训练权重不用重训）
RTC_SCHEDULES = frozenset({"exp", "linear", "ones", "zeros"})
RTC_BLEND_SCHEDULES = frozenset({"linear", "exp"})
RTC_ALGORITHMS = frozenset({"pi05", "gr00t"})
RTC_CONFIG_DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    "algorithm": "pi05",  # pi05：去噪前缀引导；gr00t：噪声前缀替换 + 速度掩码
    "execution_horizon": 10,
    "max_guidance_weight": 10.0,
    "prefix_attention_schedule": "exp",
    "inference_delay": 0,  # 同步环 leftover 未重叠时强制 0；"auto" 仅在推理期间消耗 leftover 时按耗时×fps
    "async_inference": True,  # DEFAULT：第一包起 GPU 持续后台推理，不等 committed
    "async_blending": False,  # 执行 leftover 时后台推下一包，重叠段融合
    "blend_horizon": 8,  # 融合步数；null = 用全部 leftover
    "blend_schedule": "linear",  # linear | exp
    "rtc_ramp_rate": 1.0,  # 仅 algorithm=gr00t：重叠段速度掩码的指数斜率
    "rtc_overlap_steps": None,  # 仅 gr00t；null = 用上一包未执行步数
    "rtc_frozen_steps": None,  # 仅 gr00t；null = 用 inference_delay
    "debug": False,
}

# 终端 PI05_RTC=1 时套用的真机档。这一次启动以环境变量为准，不改 YAML 文件。
RTC_TERMINAL_PROFILE: Dict[str, Any] = {
    "enabled": True,
    "algorithm": "pi05",
    "execution_horizon": 5,
    "max_guidance_weight": 5.0,
    "prefix_attention_schedule": "exp",
    "inference_delay": "auto",
    "async_inference": True,
    "async_blending": False,
    "blend_horizon": 8,
    "blend_schedule": "linear",
    "debug": False,
}

# DiT / flow action generator：YAML ``rtc_enabled`` 可开可关（默认关）。
# 非此类模型开 RTC 会在校验期拒绝。服务端还要握手 ``rtc_capable``。
RTC_OPTIONAL_POLICY_TYPES = frozenset(
    {
        PolicyType.PI05,
        PolicyType.FASTWAM,
        PolicyType.DREAMZERO,
        PolicyType.COSMOS3,
        PolicyType.MOTUS,
        PolicyType.GIGAWORLD,
        PolicyType.WALLOSS,
        PolicyType.TAU0WM,
        PolicyType.GR00T,
        PolicyType.LINGBOT_VLA,
    }
)


def policy_supports_rtc_option(ptype: PolicyType) -> bool:
    return ptype in RTC_OPTIONAL_POLICY_TYPES


def rtc_wants_async_pool(rtc: Optional[Dict[str, Any]]) -> bool:
    """持续异步推理或 leftover blending 都需要单 worker 推理线程池。"""
    if not rtc:
        return False
    return bool(rtc.get("async_inference") or rtc.get("async_blending"))


def parse_async_blending(raw: Dict[str, Any]) -> bool:
    """YAML 开关：``policy.async_blending`` 优先，否则看 ``rtc_config.async_blending``。默认关。"""
    if "async_blending" in raw:
        return bool(raw.get("async_blending"))
    rtc = raw.get("rtc_config")
    if isinstance(rtc, dict) and "async_blending" in rtc:
        return bool(rtc.get("async_blending"))
    return False


def parse_blend_horizon(raw: Dict[str, Any]) -> Optional[int]:
    if "blend_horizon" in raw:
        src = raw.get("blend_horizon")
    elif isinstance(raw.get("rtc_config"), dict) and "blend_horizon" in raw["rtc_config"]:
        src = raw["rtc_config"].get("blend_horizon")
    else:
        src = 8
    if src is None:
        return None
    n = int(src)
    if n < 0:
        raise ValueError(f"policy.blend_horizon 必须 >= 0 或 null，得到 {src!r}")
    return n


def parse_blend_schedule(raw: Dict[str, Any]) -> str:
    src = raw.get("blend_schedule")
    if src is None and isinstance(raw.get("rtc_config"), dict):
        src = raw["rtc_config"].get("blend_schedule")
    sched = str(src or "linear").strip().lower()
    if sched not in RTC_BLEND_SCHEDULES:
        raise ValueError(
            f"policy.blend_schedule 只能是 linear / exp，得到 {sched!r}"
        )
    return sched


def normalize_rtc_config(
    raw: Any,
    *,
    enabled: bool,
) -> Optional[Dict[str, Any]]:
    """把 YAML ``policy.rtc_config`` 收成统一 dict；未开 RTC 时返回 None。"""
    if not enabled:
        return None
    src = dict(raw) if isinstance(raw, dict) else {}
    merged = dict(RTC_CONFIG_DEFAULTS)
    merged.update({k: v for k, v in src.items() if v is not None})
    merged["enabled"] = True

    algorithm = str(merged.get("algorithm", "pi05")).strip().lower()
    if algorithm in {"pi0.5", "pi0_5", "openpi"}:
        algorithm = "pi05"
    if algorithm in {"groot", "gr00t_n1", "n1.7"}:
        algorithm = "gr00t"
    if algorithm not in RTC_ALGORITHMS:
        raise ValueError(
            "policy.rtc_config.algorithm 只能是 pi05 / gr00t，得到 "
            f"{algorithm!r}"
        )
    merged["algorithm"] = algorithm

    schedule = str(merged.get("prefix_attention_schedule", "exp")).strip().lower()
    if schedule in {"exponential", "exponential_decay"}:
        schedule = "exp"
    if schedule not in RTC_SCHEDULES:
        raise ValueError(
            "policy.rtc_config.prefix_attention_schedule 只能是 "
            f"exp / linear / ones / zeros，得到 {schedule!r}"
        )
    merged["prefix_attention_schedule"] = schedule

    horizon = int(merged.get("execution_horizon", 10))
    if horizon < 1:
        raise ValueError(f"policy.rtc_config.execution_horizon 必须 >= 1，得到 {horizon}")
    merged["execution_horizon"] = horizon

    weight = float(merged.get("max_guidance_weight", 10.0))
    if weight <= 0:
        raise ValueError(
            f"policy.rtc_config.max_guidance_weight 必须 > 0，得到 {weight}"
        )
    merged["max_guidance_weight"] = weight

    delay_raw = merged.get("inference_delay", 0)
    if delay_raw is None:
        delay: Union[int, str] = 0
    elif isinstance(delay_raw, str) and delay_raw.strip().lower() == "auto":
        delay = "auto"
    else:
        delay = int(delay_raw)
        if delay < 0:
            raise ValueError(
                f"policy.rtc_config.inference_delay 必须 >= 0 或 auto，得到 {delay_raw!r}"
            )
    merged["inference_delay"] = delay
    merged["async_inference"] = bool(merged.get("async_inference", True))
    merged["async_blending"] = bool(merged.get("async_blending", False))
    blend_horizon_raw = merged.get("blend_horizon", 8)
    if blend_horizon_raw is None:
        merged["blend_horizon"] = None
    else:
        blend_horizon = int(blend_horizon_raw)
        if blend_horizon < 0:
            raise ValueError(
                f"policy.rtc_config.blend_horizon 必须 >= 0 或 null，得到 {blend_horizon_raw!r}"
            )
        merged["blend_horizon"] = blend_horizon
    blend_sched = str(merged.get("blend_schedule", "linear")).strip().lower()
    if blend_sched not in RTC_BLEND_SCHEDULES:
        raise ValueError(
            "policy.rtc_config.blend_schedule 只能是 linear / exp，"
            f"得到 {blend_sched!r}"
        )
    merged["blend_schedule"] = blend_sched
    merged["debug"] = bool(merged.get("debug", False))
    ramp_rate = float(merged.get("rtc_ramp_rate", 1.0))
    if ramp_rate <= 0:
        raise ValueError(
            f"policy.rtc_config.rtc_ramp_rate 必须 > 0，得到 {ramp_rate}"
        )
    merged["rtc_ramp_rate"] = ramp_rate
    for key in ("rtc_overlap_steps", "rtc_frozen_steps"):
        raw_steps = merged.get(key)
        if raw_steps is None:
            merged[key] = None
            continue
        steps = int(raw_steps)
        if steps < 0:
            raise ValueError(f"policy.rtc_config.{key} 必须 >= 0 或 null，得到 {raw_steps!r}")
        merged[key] = steps
    return merged


def _env_flag(raw: str, name: str) -> bool:
    value = str(raw).strip().lower()
    if value in {"1", "true", "on", "yes", "y"}:
        return True
    if value in {"0", "false", "off", "no", "n"}:
        return False
    raise ValueError(f"{name} 只能是 1/0、on/off、true/false，得到 {raw!r}")


def _env_text(env: Mapping[str, str], name: str) -> Optional[str]:
    raw = env.get(name)
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def apply_terminal_rtc_config(
    cfg: Any,
    environ: Optional[Mapping[str, str]] = None,
) -> List[str]:
    """用终端环境变量覆盖这一次启动的 RTC。未设置 ``PI05_RTC`` 时不改 YAML。"""
    env = os.environ if environ is None else environ
    raw_switch = _env_text(env, "PI05_RTC")
    if raw_switch is None:
        return []

    policy = cfg.policy
    if not _env_flag(raw_switch, "PI05_RTC"):
        policy.rtc_enabled = False
        policy.rtc_config = None
        return ["[RTC] PI05_RTC=0，这次启动关闭 RTC，走普通去噪"]

    profile = dict(RTC_TERMINAL_PROFILE)
    schedule = _env_text(env, "PI05_RTC_SCHEDULE")
    if schedule is not None:
        profile["prefix_attention_schedule"] = schedule
    weight = _env_text(env, "PI05_RTC_MAX_GUIDANCE_WEIGHT")
    if weight is not None:
        profile["max_guidance_weight"] = float(weight)
    execution_horizon = _env_text(env, "PI05_RTC_EXECUTION_HORIZON")
    if execution_horizon is not None:
        profile["execution_horizon"] = int(execution_horizon)
    delay = _env_text(env, "PI05_RTC_INFERENCE_DELAY")
    if delay is not None:
        profile["inference_delay"] = delay if delay.lower() == "auto" else int(delay)
    async_inference = _env_text(env, "PI05_RTC_ASYNC_INFERENCE")
    if async_inference is not None:
        profile["async_inference"] = _env_flag(async_inference, "PI05_RTC_ASYNC_INFERENCE")
    async_blending = _env_text(env, "PI05_RTC_ASYNC_BLENDING")
    if async_blending is not None:
        profile["async_blending"] = _env_flag(async_blending, "PI05_RTC_ASYNC_BLENDING")
    debug = _env_text(env, "PI05_RTC_DEBUG")
    if debug is not None:
        profile["debug"] = _env_flag(debug, "PI05_RTC_DEBUG")
    algorithm = _env_text(env, "PI05_RTC_ALGORITHM")
    if algorithm is not None:
        profile["algorithm"] = algorithm

    action_horizon = int(getattr(policy, "openpi_action_horizon", 50) or 50)
    steps_text = _env_text(env, "PI05_RTC_N_ACTION_STEPS")
    if steps_text is not None:
        nas = int(steps_text)
    else:
        current = getattr(policy, "n_action_steps_override", None)
        if current is None or int(current) >= action_horizon:
            nas = min(15, max(1, action_horizon - 1))
        else:
            nas = int(current)

    policy.rtc_enabled = True
    policy.rtc_config = normalize_rtc_config(profile, enabled=True)
    policy.async_blending = bool(policy.rtc_config["async_blending"])
    policy.blend_horizon = policy.rtc_config["blend_horizon"]
    policy.blend_schedule = str(policy.rtc_config["blend_schedule"])
    policy.n_action_steps_override = nas
    batch = getattr(cfg, "batch_infer", None)
    if batch is not None:
        batch.action_steps = [nas]

    rtc = policy.rtc_config
    return [
        (
            "[RTC] PI05_RTC=1，这次启动打开前缀引导："
            f"n_action_steps={nas} schedule={rtc['prefix_attention_schedule']} "
            f"max_guidance_weight={rtc['max_guidance_weight']} "
            f"execution_horizon={rtc['execution_horizon']} "
            f"inference_delay={rtc['inference_delay']}"
        ),
        "[RTC] 去噪引导在服务端。客户端把上一包剩余动作和这份配置发给服务端。",
    ]


def _resolve_config_path(path: str | Path) -> Path:
    """解析配置路径；旧扁平路径不存在时自动映射到 lerobot/ 或 aloha/。"""
    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    config_path = config_path.resolve()
    if config_path.is_file():
        return config_path

    remapped = _LEGACY_FLAT_CONFIG_REMAP.get(config_path.name)
    if remapped is None:
        return config_path

    # 仅当调用方仍指向旧扁平位置（config/ 根或任意同名缺失路径）时重映射
    candidate = (_CONFIG_ROOT / remapped).resolve()
    if not candidate.is_file():
        return config_path

    warnings.warn(
        f"配置路径已迁移: {config_path.name} → infer/config/{remapped}。"
        f"请改用新路径（当前仍兼容旧路径）。",
        DeprecationWarning,
        stacklevel=3,
    )
    return candidate


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base without mutating either input."""
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML 顶层必须是字典: {path}")
    return data


def _resolve_sidecar_path(config_path: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def _resolve_optional_sidecar_path(
    config_path: Path,
    raw_path: str | Path | None,
) -> Optional[Path]:
    if raw_path is None or str(raw_path).strip() == "":
        return None
    return _resolve_sidecar_path(config_path, raw_path)


def controller_config_path() -> Path:
    """机械臂控制器配置。不跟实验 YAML 走。"""
    return Path(__file__).resolve().parent / "config" / "direct_robot_controller.yaml"


def _apply_controller_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    """已回撤到 10/27 上午行为：不再用 direct_robot_controller.yaml 覆盖 joint_stream。

    保留函数名以免打断 load_infer_config 调用链；直接返回原配置。
    """
    return raw


def _apply_local_env_config(config_path: Path, raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply optional machine-local overrides.

    Main experiment YAMLs can set:

      local_env: local_env.yaml

    The referenced YAML uses the same section names as the main config and is
    deep-merged on top of the main config. This keeps paths/IPs local while
    preserving model/task/session defaults in versioned configs.
    """
    local_env = raw.get("local_env")
    if not local_env:
        return raw
    local_path = _resolve_sidecar_path(config_path, local_env)
    if not local_path.is_file():
        raise FileNotFoundError(f"local_env 配置不存在: {local_path}")
    local_raw = _load_yaml(local_path)
    return _deep_merge(raw, local_raw)


@dataclass
class PolicyConfig:
    type: PolicyType
    ckpt_dir: Path
    task_prompt: str
    # ACT / Diffusion 专用
    dataset_repo_id: Optional[str] = None
    dataset_root: Optional[Path] = None
    n_action_steps_override: Optional[int] = None
    num_inference_steps_override: Optional[int] = None
    # None=适配器默认（Cosmos 钉死 16）；false=允许菜单/网格扫 action-step
    pin_exec_horizon: Optional[bool] = None
    use_action_chunk: bool = False
    # True: execute the current action chunk while the next chunk is inferred
    # in a background worker (π0.5-style async inference).
    async_inference: bool = True
    keep_dims: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 14])
    # YAML 开关：默认关。true 仅 DiT/flow 允许，且服务端须 rtc_capable。
    rtc_enabled: bool = False
    rtc_config: Optional[Dict[str, Any]] = None
    # WALL-OSS 服务端 IK（相对 EE → 关节）。其它策略忽略。
    ik: Optional[Dict[str, Any]] = None
    # YAML 开关：默认关。开则客户端把 leftover 与新 chunk 重叠融合（不依赖 RTC）。
    async_blending: bool = False
    blend_horizon: Optional[int] = 8
    blend_schedule: str = "linear"
    # Action output postprocess before expanding/sending robot commands.
    action_scale: float = 1.0
    action_clip_abs: Optional[float] = None
    # 批量推理 / 脚踏板模式：每组独立任务数（任务脚踏板开始+结束各算 1 个）
    tasks_per_group: Optional[int] = None
    # 视觉输入：null=从 checkpoint 自动推断；[]=纯 state；非空=仅用列出的 feature key
    camera_keys: Optional[List[str]] = None
    # LingBot-VA：训练 VAE 抽帧 fps（H100/江算 TARGET_FPS=5，间隔 200ms）
    va_obs_fps: Optional[float] = None
    # LingBot-VA / DreamZero websocket（VA 默认 1234，DreamZero 默认 1235）
    lingbot_backend: str = "websocket"
    lingbot_host: str = "127.0.0.1"
    lingbot_port: int = 1234
    lingbot_connect_timeout_sec: float = 300.0
    # 握手 metadata.model；设了则连错服务直接退出
    lingbot_expected_model: Optional[str] = None
    # LingBot-VLA：须与 lingbot-vla-v2/configs/robot_configs/{name}.yaml 一致
    lingbot_robo_name: str = "tongbot_16d"
    lingbot_action_key: str = "action"
    lingbot_image_key_map: Optional[Dict[str, str]] = None
    lingbot_robot_config: Optional[Path] = None
    # joint | ee | auto/None。None=读 robot_config actions（end.position→EE IK）
    action_space: Optional[str] = None
    ee_ik_sdk_path: Optional[Path] = None
    # LingBot-VA 服务端 yaml（websocket 模式下权重/采样参数在那边）
    va_server_config: Optional[Path] = None
    # LingBot-VA 训练记录（ckpt 旁 train_record.json，或显式路径）
    va_train_config: Optional[Path] = None
    # ACT implementation: lerobot (pretrained_model directory) or aloha
    act_backend: str = "lerobot"
    # ALOHA native checkpoint/statistics/model arguments.
    aloha_checkpoint_name: str = "policy_best.ckpt"
    aloha_stats_path: Optional[Path] = None
    aloha_model_args: Dict[str, Any] = field(default_factory=dict)
    # Pi0.5 implementation: lerobot (pretrained_model) or openpi (vendored infer/policies/openpi)
    pi05_backend: str = "lerobot"
    # Deprecated: openpi is vendored under infer/policies/openpi. Optional override for tests only.
    openpi_repo_path: Optional[Path] = None
    openpi_config_name: str = "pi05_aloha"
    # tongbot | aloha | libero；空=从 openpi_config_name 推断（含 libero → libero，否则 tongbot）
    openpi_embodiment: Optional[str] = None
    # None=与 openpi_action_dim 相同；LIBERO 为 8
    openpi_state_dim: Optional[int] = None
    # AgileX-style short names → openpi image keys (left_wrist / base / right_wrist)
    openpi_camera_names: List[str] = field(
        default_factory=lambda: ["left", "front", "right"]
    )
    openpi_action_dim: int = 14
    openpi_action_horizon: int = 50
    openpi_image_size: int = 224
    # null=沿用 TrainConfig；TongBot 真机通常 false（避免 Aloha 关节翻转/夹爪角变换）
    openpi_adapt_to_pi: Optional[bool] = None
    # null=沿用 TrainConfig.data.delta_action_dims；TongBot 16D 用 [7,7,-1,-1]
    # 训练槽 [7,-1,7,-1] 时同时 openpi_interleave_grippers: true
    openpi_delta_action_dims: Optional[List[int]] = None
    # 落盘 [左7|右7|左爪|右爪] 重排成 [左7|左爪|右7|右爪] 再做 delta。真机下发仍是原顺序。
    openpi_interleave_grippers: bool = False
    # null=沿用模型默认（π0.5 为 quantile）。quantile | zscore：state 输入与 action 输出同一套。
    openpi_vector_norm: Optional[str] = None
    # GR00T N1.7：ZMQ 客户端 → PolicyServer（权重在服务端）
    gr00t_action_space: str = "joint16"  # joint16 / ee20 (client FK + IK)
    gr00t_host: str = "127.0.0.1"
    gr00t_port: int = 5555
    gr00t_timeout_ms: int = 300000
    gr00t_action_key: str = "observation.state"
    gr00t_state_key: str = "observation.state"
    gr00t_language_key: str = "annotation.human.task_description"
    gr00t_video_keys: Optional[List[str]] = None
    gr00t_image_key_map: Optional[Dict[str, str]] = None
    # DM0.5 HTTP。当前图用最新帧；头相机历史在客户端按 dm05_history_hz 取样。
    dm05_host: str = "127.0.0.1"
    dm05_port: int = 7891
    dm05_timeout_sec: float = 120.0
    dm05_history_slots: int = 32
    dm05_history_hz: float = 1.0
    dm05_history_tolerance_sec: float = 0.2
    dm05_robot_type: str = "tong_dual_arm"
    dm05_speed: str = "0.5"
    dm05_send_history: bool = True
    # 训练用了 prompt_from_subtask 时：推理走两层（VLM 拆 subtask → 动作模型）
    prompt_from_subtask: bool = False
    vlm_planner: Optional["VlmPlannerConfig"] = None


@dataclass
class VlmPlannerConfig:
    """上层 VLM。enabled=None 表示跟随 policy.prompt_from_subtask。"""

    enabled: Optional[bool] = None
    weight_dir: Path = field(
        default_factory=lambda: Path("/home/xk/VLA/openpi/models/e2_subtask_ce_best")
    )
    tokenizer_path: Path = field(
        default_factory=lambda: Path("/home/xk/VLA/openpi/models/pi05_fixed/tokenizer.model")
    )
    device: str = "cuda:0"
    camera: str = "front"
    interval_sec: float = 1.0
    max_new_tokens: int = 32
    prompt_template: str = (
        "Task: {task}\nWhat is the current subtask?\nSubtask:"
    )


def vlm_planner_enabled(pol: "PolicyConfig") -> bool:
    planner = getattr(pol, "vlm_planner", None)
    if planner is not None and planner.enabled is False:
        return False
    if planner is not None and planner.enabled is True:
        return True
    return bool(getattr(pol, "prompt_from_subtask", False))


@dataclass
class NetworkConfig:
    high_host: str
    low_host: str
    state_port: int
    control_port: int
    gripper_port: int
    camera_port: int = 18888
    gripper_state_port: int = 30016
    gripper_state_stale_sec: float = 1.0


@dataclass
class CameraDeviceConfig:
    serial: str
    # 与 checkpoint input_features 一致；省略时按 devices 下标对应 infer 相机 key 顺序
    key: Optional[str] = None
    # 真机采集分辨率（RealSense 硬件）；省略时由启动校验自动填充
    width: Optional[int] = None
    height: Optional[int] = None
    # 模型/checkpoint 目标分辨率；与采集不一致时在相机输出前 resize
    model_width: Optional[int] = None
    model_height: Optional[int] = None
    fps: int = 30


@dataclass
class CameraSenderConfig:
    enabled: bool = False
    transport: str = "local"  # local | tcp
    target_host: Optional[str] = None  # null -> network.high_host
    target_port: Optional[int] = None  # null -> network.camera_port
    jpeg_quality: int = 95
    reconnect_delay_sec: float = 0.5
    get_timeout_sec: float = 3.0
    use_host_timestamp: bool = True
    # local 模式：同时采集 RealSense depth 并对齐到 color（用于 videos/*_depth.mp4）
    enable_depth: bool = True
    devices: List[CameraDeviceConfig] = field(default_factory=list)


@dataclass
class JointStreamConfig:
    """15Hz 目标经本地插值后，按固定周期透传给关节。

    high_follow 为真时，睿尔曼要求周期不超过 10ms，即 hz 至少 100。
    trajectory_mode 0 是完全透传。换块阶跃由 slew_deg_per_sec 摊开。
    """

    enabled: bool = False
    hz: float = 100.0
    high_follow: bool = False
    trajectory_mode: int = 0
    radio: int = 0
    slew_deg_per_sec: float = 90.0


@dataclass
class DirectRobotConfig:
    enabled: bool = False
    sdk_path: Optional[Path] = None
    gripper_mode: str = "udp"  # udp | local | none
    # 模型 observation.state 夹爪维来源：udp=推理帧最新 UDP；buffer=同步缓冲池对齐值
    gripper_state_source: str = "udp"  # udp | buffer
    gripper_initial_position: List[float] = field(default_factory=lambda: [0.7, 0.7])
    robot_r_host: str = "192.168.1.17"
    robot_l_host: str = "192.168.1.18"
    port_r: int = 8080
    port_l: int = 8080
    enable_joint: List[bool] = field(default_factory=lambda: [False, False, False])
    left_gripper_kwargs: Optional[Dict[str, Any]] = field(default_factory=lambda: {
        "dxl_name": "XM430",
        "Device_name": "/dev/ttyUSB0",
        "dxl_id": 1,
        "baudrate": 4000000,
        "min_position": 1395,
        "max_position": 2043,
    })
    right_gripper_kwargs: Optional[Dict[str, Any]] = None
    state_poll_hz: float = 100.0
    # 批量推理每组 test 开始前回到标准初始位（非每个 Ctrl+. 任务）
    reset_to_init_on_group_start: bool = False
    standard_init_joint_degrees: Optional[List[float]] = None
    reset_move_speed: int = 10
    reset_settle_sec: float = 1.5
    # standard | dataset_median | dataset_mean — ACT/DP 可从训练集 episode 首帧统计初始位
    init_pose_source: str = "standard"
    # 每个任务段（Ctrl+. 开始）在参考末端 YZ 圆盘内随机左臂初始位姿
    random_task_init_pose: bool = False
    random_task_init_radius_m: float = 0.10
    random_task_init_max_attempts: int = 32
    # 每个 Ctrl+. 开始任务段前回到 standard_init（与组间复位独立）
    reset_to_init_on_task_start: bool = False
    # 双臂：复位前用左臂关节重写右臂（见 dual_arm_init_mirror）；默认关
    mirror_right_init_from_left: bool = False
    joint_stream: JointStreamConfig = field(default_factory=JointStreamConfig)


@dataclass
class ExoSafetyConfig:
    """外骨骼推理安全保障；enabled=false 时不影响现有推理流程。"""

    enabled: bool = False
    takeover_key: str = "ctrl+1"
    safety_infer_key: str = "ctrl+2"
    exo_state_port: int = 18140
    exo_command_port: int = 30026
    collect_config_path: Optional[Path] = None
    record_start_delay_sec: float = 2.0
    collect_cam_mid_window_name: str = "exo_collect_cam_mid"
    # 接管前对齐门控；false=旧行为（Ctrl+1 立即跟人）
    align_enabled: bool = True
    # auto=驱动外骨骼到机械臂；manual=人摆外骨骼
    align_mode: str = "auto"
    align_max_joint_err_deg: float = 12.0
    align_rms_joint_err_deg: float = 8.0
    align_log_every_sec: float = 1.0
    align_target_port: int = 30027
    align_drive_max_step_deg: float = 2.0
    align_target_send_hz: float = 50.0


@dataclass
class PicoRecoveryConfig:
    enabled: bool = False
    endpoint: str = "tcp://127.0.0.1:5555"
    timeout_sec: float = 0.3
    gripper_open: Optional[List[float]] = None
    gripper_closed: Optional[List[float]] = None
    # 交回推理后首帧动作与实测关节（含 J7）的最大允许偏差，超过则原地 hold 不下发。
    first_action_max_joint_delta_deg: float = 8.0

    def validate_grippers(self, gripper_mode: str) -> None:
        grip = (self.gripper_open, self.gripper_closed)
        if grip[0] is not None and gripper_mode == "none":
            raise ValueError("PICO gripper controls require a configured gripper")
        if (grip[0] is None) != (grip[1] is None) or any(
            values is not None and (len(values) != 2 or not all(0 <= v <= 1 for v in values))
            for values in grip
        ):
            raise ValueError("Configure both PICO gripper targets as left/right ratios in [0, 1]")


@dataclass
class SyncConfig:
    max_time_diff: Optional[float]
    sync_delay: float
    buffer_size: int
    prefill_sec: float
    min_anchor_buffer: int
    clock_offset: Dict[str, float]

    def resolved_max_time_diff(self, fps: int) -> float:
        if self.max_time_diff is not None:
            return float(self.max_time_diff)
        return max(0.1, 2.0 / fps)


@dataclass
class RecordingConfig:
    enabled: bool = True
    save_videos: bool = True
    save_trajectory: bool = True
    save_third_cam: bool = True
    save_fourth_cam: bool = True
    save_depth_videos: bool = True  # JET 伪彩预览 mp4（不可反算 mm）
    save_depth_raw: bool = True  # uint16 mm PNG 序列（可分析/进训）
    inference_dir: Optional[Path] = None  # null → ckpt 同级 inference/
    generate_debug_plot: bool = True
    debug_plot_max_dims: int = 8
    debug_plot_dpi: int = 160
    golden_capture: bool = False
    golden_capture_dir: Optional[Path] = None
    golden_capture_mode: str = "policy"


@dataclass
class SessionConfig:
    fps: int = 15
    max_tests: Optional[int] = None
    max_duration_sec: Optional[float] = 30.0
    prompt_continue: bool = True
    auto_finalize: bool = True
    task_mode: str = "test"  # test | continuous_tasks
    visualize_cam_left: bool = False
    cam_left_window_name: str = "cam_left"
    visualize_cam_mid: bool = False
    cam_mid_window_name: str = "cam_mid"
    # 仅中间相机预览窗口尝试置顶；左/右不置顶。后端不支持时静默忽略。
    cam_mid_topmost: bool = True
    visualize_cam_right: bool = False
    cam_right_window_name: str = "cam_right"
    # 脚踏板控制任务段开始/结束；按键映射见顶层 pedal: 段
    pedal_task_control: bool = False
    tasks_per_group: Optional[int] = None
    # 每个任务段结束后询问是否修改 task_prompt（OOD language 测试）
    interactive_task_prompt: bool = False
    # 按 Ctrl+.（task_toggle）打点后立即追问文字描述，答案写
    # inference_eval_marks.jsonl，汇总写 inference_eval_summary.json。
    # timeout <= 0：一直等到输入，不超时（默认）。
    eval_prompt_on_marker: bool = True
    eval_prompt_timeout_sec: float = 0.0
    # 两次模型调用之间执行的控制步数。空则用动作块长度的五分之一。
    replan_steps: Optional[int] = None


@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_file: Optional[Path] = None
    machine_flow_enabled: bool = True
    alerts_enabled: bool = True
    alerts_path: Optional[Path] = None  # null → test_dir/alerts.jsonl
    # 硬件/网络/相机/夹爪状态采样与结束汇总
    device_health_enabled: bool = True
    device_health_sample_interval_sec: float = 1.0
    device_health_stale_state_sec: float = 0.5
    device_health_stale_camera_sec: float = 1.0
    device_health_ping_low_host: bool = True
    # 重大异常：实时界面 + 即刻暂停推理
    device_health_pause_on_anomaly: bool = True
    device_health_pause_require_confirm: bool = True
    device_health_pause_hold_sec: float = 1.5
    device_health_resume_hold_sec: float = 1.0
    device_health_pause_on_gripper_udp_stale: bool = True
    device_health_pause_on_gripper_tracking_error: bool = True
    device_health_gripper_tracking_err: float = 0.25
    device_health_gripper_tracking_consecutive: int = 12
    device_health_pause_on_low_host_unreachable: bool = True
    device_health_pause_on_state_stale: bool = True
    device_health_pause_on_camera_stale: bool = True
    device_health_pause_warmup_sec: float = 5.0
    device_health_live_status_ui: bool = True
    device_health_live_status_window_name: str = "device_health"
    # run.log 控制台/文件级别；DEBUG 可打开逐步向量
    human_log_level: Optional[str] = None
    file_log_level: Optional[str] = None
    log_sync_every_n_frames: int = 30
    log_action_steps_detail: bool = False


@dataclass
class EvaluationConfig:
    """推理结束后自动评估与报告。"""

    enabled: bool = True
    auto_after_each_test: bool = True
    save_report: bool = True
    report_filename: str = "evaluation_report.json"
    metrics: List[str] = field(default_factory=lambda: ["automatic"])
    # 人工标签文件（可选，用于 TSR / grasp_success 等）
    labels_file: Optional[Path] = None
    sync_warn_rate_threshold: float = 0.30
    eval_log_enabled: bool = True
    eval_log_filename: str = "eval_log.jsonl"
    eval_log_save_images: bool = False
    eval_log_image_dir: str = "eval_images"
    eval_log_image_format: str = "jpg"
    eval_log_save_videos: bool = False
    eval_log_video_dir: str = "eval_videos"
    eval_log_generate_plots: bool = True
    eval_log_plot_dir: str = "eval_plots"
    eval_log_plot_dpi: int = 150


@dataclass
class BatchRemoteSyncConfig:
    """批量推理结束后，将实验目录同步到远程服务器（SSH 私钥）。"""

    enabled: bool = False
    host: str = ""
    port: int = 22
    username: str = ""
    remote_dir: str = ""
    private_key_path: Path = field(default_factory=lambda: Path("~/.ssh/id_rsa"))


@dataclass
class VlaAutoEvalConfig:
    """批量推理 action-step 组结束后的 VLA 自动评估（vla_eval）。"""

    enabled: bool = True
    table_template: Optional[Path] = None
    config: Optional[Path] = None
    output_dirname: str = "eval_table"
    result_table_name: str = "eval_result.et"
    summary_name: str = "summary.md"


@dataclass
class TaskPromptPreset:
    """批量推理可选 task_prompt 条目。"""

    id: str
    label: str
    task_prompt: str


@dataclass
class BatchInferConfig:
    """批量真机消融（run_batch_infer.py）专用。"""

    action_step_parts: int = 5
    # 显式列表优先于 action_step_parts。denoise_steps 非空则走 2D 网格。
    # π0.5 ASYNC 下即 session.replan_steps 候选（结果目录 replan-steps_N/）。
    action_steps: Optional[List[int]] = None
    denoise_steps: Optional[List[int]] = None
    phase: str = "all"  # all | action-step | ood
    output_dir: Optional[Path] = None  # null → pretrained_model 同级目录
    best_action_step: Optional[int] = None
    skip_existing: bool = False
    ood_presets: Optional[Path] = None  # null → 同目录 ood_language_presets.yaml
    # 启动时从中选择 task_prompt；空则只用 policy.task_prompt
    task_prompt_presets: List[TaskPromptPreset] = field(default_factory=list)
    # action-step 每组独立任务数；null 时使用 policy.tasks_per_group
    tasks_per_group: Optional[int] = None
    # 已废弃：OOD 段数由 ood_language_presets 数量决定
    ood_tasks_per_group: int = 1
    # true：每组开始前自动生成 WPS 评分表 .et 并打开；false：跳过（无 WPS 或不需要人工评分时）
    feedback_score_sheet: bool = True
    vla_auto_eval: VlaAutoEvalConfig = field(default_factory=VlaAutoEvalConfig)
    remote_sync: BatchRemoteSyncConfig = field(default_factory=BatchRemoteSyncConfig)


@dataclass
class AblationRunSpec:
    name: str
    overrides: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    enabled: bool = True
    group: str = ""
    is_baseline: bool = False


@dataclass
class AblationConfig:
    enabled: bool = False
    experiment_id: Optional[str] = None
    runs: List[AblationRunSpec] = field(default_factory=list)
    compare_report_dir: Optional[Path] = None


def _parse_pedal_keys(raw: Any) -> PedalKeyBinding:
    if not isinstance(raw, dict):
        raw = {}
    binding = PedalKeyBinding(
        task_toggle=str(raw.get("task_toggle", "ctrl+.")).strip(),
        confirm=str(raw.get("confirm", "y")).strip(),
        cancel=str(raw.get("cancel", "n")).strip(),
        discard=str(raw.get("discard", "delete")).strip(),
        exo_takeover=str(raw.get("exo_takeover", "ctrl+1")).strip(),
        exo_safety_infer=str(raw.get("exo_safety_infer", "ctrl+2")).strip(),
    )
    if not binding.confirm or not binding.cancel:
        raise ValueError("pedal.confirm 与 pedal.cancel 不能为空")
    if binding.confirm.lower() == binding.cancel.lower():
        raise ValueError("pedal.confirm 与 pedal.cancel 不能相同")
    user_hotkey_to_pynput(binding.task_toggle)
    user_hotkey_to_pynput(binding.exo_takeover)
    user_hotkey_to_pynput(binding.exo_safety_infer)
    return binding


@dataclass
class InferConfig:
    experiment_name: str
    engine: EngineMode
    policy: PolicyConfig
    network: NetworkConfig
    camera_sender: CameraSenderConfig
    direct_robot: DirectRobotConfig
    exo_safety: ExoSafetyConfig
    sync: SyncConfig
    recording: RecordingConfig
    session: SessionConfig
    logging: LoggingConfig
    evaluation: EvaluationConfig
    ablation: AblationConfig
    batch_infer: BatchInferConfig
    pedal: PedalKeyBinding
    config_path: Path
    pico_recovery: PicoRecoveryConfig = field(default_factory=PicoRecoveryConfig)

    def resolve_inference_dir(self) -> Path:
        if self.recording.inference_dir is not None:
            return self.recording.inference_dir
        return self.policy.ckpt_dir.parent / "inference"

    def resolve_ablation_experiment_id(self) -> str:
        if self.ablation.experiment_id:
            return self.ablation.experiment_id
        ts = time.strftime("%Y%m%d_%H%M%S")
        return f"{self.experiment_name}_{ts}"


def _parse_int_list(raw: Any) -> Optional[List[int]]:
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"期望整数列表，得到 {type(raw).__name__}")
    values = [int(x) for x in raw]
    if any(v < 1 for v in values):
        raise ValueError(f"整数列表必须全 >= 1，得到 {values}")
    return values


def _parse_task_prompt_presets(raw: Any) -> List[TaskPromptPreset]:
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)):
        raise ValueError(
            f"batch_infer.task_prompt_presets 须为列表，得到 {type(raw).__name__}"
        )
    out: List[TaskPromptPreset] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(
                f"batch_infer.task_prompt_presets[{i}] 须为 dict，得到 {type(item).__name__}"
            )
        prompt = str(item.get("task_prompt") or "").strip()
        if not prompt:
            raise ValueError(
                f"batch_infer.task_prompt_presets[{i}].task_prompt 不能为空"
            )
        preset_id = str(item.get("id") or f"prompt_{i + 1}").strip()
        label = str(item.get("label") or item.get("name") or preset_id).strip()
        out.append(TaskPromptPreset(id=preset_id, label=label, task_prompt=prompt))
    return out


def _parse_batch_remote_sync(raw: Any) -> BatchRemoteSyncConfig:
    if not isinstance(raw, dict):
        raw = {}
    key_path = raw.get("private_key_path", "~/.ssh/id_rsa")
    return BatchRemoteSyncConfig(
        enabled=bool(raw.get("enabled", False)),
        host=str(raw.get("host", "")).strip(),
        port=int(raw.get("port", 22)),
        username=str(raw.get("username", "")).strip(),
        remote_dir=str(raw.get("remote_dir", "")).strip(),
        private_key_path=Path(key_path).expanduser(),
    )


def _parse_vla_auto_eval(config_path: Path, raw: Any) -> VlaAutoEvalConfig:
    if not isinstance(raw, dict):
        raw = {}
    table_template = (
        _resolve_sidecar_path(config_path, raw["table_template"])
        if raw.get("table_template")
        else None
    )
    eval_config = (
        _resolve_sidecar_path(config_path, raw["config"])
        if raw.get("config")
        else None
    )
    return VlaAutoEvalConfig(
        enabled=bool(raw.get("enabled", True)),
        table_template=table_template,
        config=eval_config,
        output_dirname=str(raw.get("output_dirname", "eval_table")),
        result_table_name=str(raw.get("result_table_name", "eval_result.et")),
        summary_name=str(raw.get("summary_name", "summary.md")),
    )


def _parse_policy_type(raw: str) -> PolicyType:
    t = str(raw).strip().lower()
    if t in ("dp", "diffusion"):
        return PolicyType.DIFFUSION
    if t in ("aloha_dp", "aloha-dp", "aloha_diffusion"):
        return PolicyType.ALOHA_DP
    if t in ("aloha_rdt", "aloha-rdt", "rdt", "aloha_rt"):
        return PolicyType.ALOHA_RDT
    if t in ("gr00t", "groot", "gr00t_n17", "groot_n17"):
        return PolicyType.GR00T
    if t in ("dreamzero", "dreamzero_5b"):
        return PolicyType.DREAMZERO
    if t in ("cosmos3", "cosmos", "cosmos3_nano"):
        return PolicyType.COSMOS3
    if t in ("internvla", "internvla_a15", "internvla_a1_5", "internvla-a1.5"):
        return PolicyType.INTERNVLA
    if t in ("memoryvla", "memvla", "memory_vla"):
        return PolicyType.MEMORYVLA
    if t in ("walloss", "wall-oss", "wall_oss", "wall-x", "wallx"):
        return PolicyType.WALLOSS
    if t in ("gigaworld", "giga_world", "giga-world", "gwp", "gwp05", "gwp0.5",
             "gigaworld_policy", "giga_world_policy"):
        return PolicyType.GIGAWORLD
    if t in ("tau0wm", "tau0", "tau-0-wm", "tau_0_wm", "tau-0", "tauwm"):
        return PolicyType.TAU0WM
    if t in ("dm05", "dm0.5", "dm0_5", "opendm"):
        return PolicyType.DM05
    return PolicyType(t)


def _parse_policy(raw: Dict[str, Any]) -> PolicyConfig:
    ptype = _parse_policy_type(raw["type"])
    return PolicyConfig(
        type=ptype,
        ckpt_dir=Path(raw["ckpt_dir"]).expanduser(),
        task_prompt=str(raw.get("task_prompt", "")),
        dataset_repo_id=raw.get("dataset_repo_id"),
        dataset_root=Path(raw["dataset_root"]).expanduser() if raw.get("dataset_root") else None,
        n_action_steps_override=raw.get("n_action_steps_override"),
        num_inference_steps_override=raw.get("num_inference_steps_override"),
        pin_exec_horizon=(
            bool(raw["pin_exec_horizon"]) if raw.get("pin_exec_horizon") is not None else None
        ),
        use_action_chunk=bool(raw.get("use_action_chunk", False)),
        async_inference=bool(raw.get("async_inference", True)),
        keep_dims=list(raw.get("keep_dims", [0, 1, 2, 3, 4, 5, 6, 14])),
        rtc_enabled=bool(raw.get("rtc_enabled", False)),
        rtc_config=normalize_rtc_config(
            raw.get("rtc_config"),
            enabled=bool(raw.get("rtc_enabled", False)),
        ),
        ik=_parse_walloss_ik(raw.get("ik"), ptype),
        async_blending=parse_async_blending(raw),
        blend_horizon=parse_blend_horizon(raw),
        blend_schedule=parse_blend_schedule(raw),
        action_scale=float(raw.get("action_scale", 1.0)),
        action_clip_abs=raw.get("action_clip_abs"),
        tasks_per_group=int(raw["tasks_per_group"])
        if raw.get("tasks_per_group") is not None
        else None,
        camera_keys=_parse_policy_camera_keys(raw.get("camera_keys")),
        va_obs_fps=(
            float(raw["va_obs_fps"]) if raw.get("va_obs_fps") is not None else None
        ),
        lingbot_backend=str(raw.get("lingbot_backend", "websocket")).strip().lower()
        or "websocket",
        lingbot_host=str(raw.get("lingbot_host", "127.0.0.1")).strip() or "127.0.0.1",
        lingbot_port=int(
            raw.get(
                "lingbot_port",
                {
                    PolicyType.WALLOSS: 1239,
                    PolicyType.MEMORYVLA: 1238,
                    PolicyType.INTERNVLA: 1237,
                    PolicyType.COSMOS3: 1236,
                    PolicyType.DREAMZERO: 1235,
                    PolicyType.LINGBOT_VLA: 8006,
                    PolicyType.PI05: 1242,
                    PolicyType.GIGAWORLD: 1243,
                    PolicyType.TAU0WM: 1244,
                }.get(ptype, 1234),
            )
        ),
        lingbot_connect_timeout_sec=float(
            raw.get("lingbot_connect_timeout_sec", 300.0)
        ),
        lingbot_expected_model=(
            str(raw["lingbot_expected_model"]).strip()
            if raw.get("lingbot_expected_model")
            else {
                PolicyType.WALLOSS: "walloss",
                PolicyType.MEMORYVLA: "memoryvla",
                PolicyType.INTERNVLA: "internvla_a15",
                PolicyType.COSMOS3: "cosmos3_nano",
                PolicyType.DREAMZERO: "dreamzero_5b",
                PolicyType.PI05: "pi05",
                PolicyType.GIGAWORLD: "gwp05",
                PolicyType.TAU0WM: "tau0wm",
            }.get(ptype)
        ),
        lingbot_robo_name=str(raw.get("lingbot_robo_name", "tongbot_16d")).strip()
        or "tongbot_16d",
        lingbot_action_key=str(raw.get("lingbot_action_key", "action")).strip()
        or "action",
        lingbot_image_key_map=_parse_str_str_map(raw.get("lingbot_image_key_map")),
        lingbot_robot_config=(
            Path(raw["lingbot_robot_config"]).expanduser()
            if raw.get("lingbot_robot_config")
            else None
        ),
        action_space=(
            None
            if str(raw.get("action_space") or "").strip().lower()
            in ("", "auto", "detect", "none")
            else str(raw["action_space"]).strip().lower()
        ),
        ee_ik_sdk_path=(
            Path(raw["ee_ik_sdk_path"]).expanduser()
            if raw.get("ee_ik_sdk_path")
            else None
        ),
        va_server_config=(
            Path(raw["va_server_config"]).expanduser()
            if raw.get("va_server_config")
            else None
        ),
        va_train_config=(
            Path(raw["va_train_config"]).expanduser()
            if raw.get("va_train_config")
            else None
        ),
        act_backend=str(raw.get("act_backend", "lerobot")).strip().lower(),
        aloha_checkpoint_name=str(raw.get("aloha_checkpoint_name", "policy_best.ckpt")),
        aloha_stats_path=Path(raw["aloha_stats_path"]).expanduser()
        if raw.get("aloha_stats_path") else None,
        aloha_model_args=dict(raw.get("aloha_model_args") or {}),
        pi05_backend=str(raw.get("pi05_backend", "lerobot")).strip().lower(),
        openpi_repo_path=Path(raw["openpi_repo_path"]).expanduser()
        if raw.get("openpi_repo_path") else None,
        openpi_config_name=str(raw.get("openpi_config_name", "pi05_aloha")).strip(),
        openpi_embodiment=(
            str(raw["openpi_embodiment"]).strip().lower()
            if raw.get("openpi_embodiment")
            else None
        ),
        openpi_camera_names=_parse_openpi_camera_names(raw.get("openpi_camera_names")),
        openpi_action_dim=int(raw.get("openpi_action_dim", 14)),
        openpi_state_dim=(
            int(raw["openpi_state_dim"])
            if raw.get("openpi_state_dim") is not None
            else None
        ),
        openpi_action_horizon=int(raw.get("openpi_action_horizon", 50)),
        openpi_image_size=int(raw.get("openpi_image_size", 224)),
        openpi_adapt_to_pi=(
            None if raw.get("openpi_adapt_to_pi") is None else bool(raw.get("openpi_adapt_to_pi"))
        ),
        openpi_delta_action_dims=_parse_openpi_delta_action_dims(
            raw.get("openpi_delta_action_dims")
        ),
        openpi_interleave_grippers=bool(raw.get("openpi_interleave_grippers", False)),
        openpi_vector_norm=_parse_openpi_vector_norm(raw.get("openpi_vector_norm")),
        gr00t_action_space=str(raw.get("gr00t_action_space", "joint16")),
        gr00t_host=str(raw.get("gr00t_host", "127.0.0.1")).strip() or "127.0.0.1",
        gr00t_port=int(raw.get("gr00t_port", 5555)),
        gr00t_timeout_ms=int(raw.get("gr00t_timeout_ms", 300000)),
        gr00t_action_key=str(raw.get("gr00t_action_key", "observation.state")).strip()
        or "observation.state",
        gr00t_state_key=str(raw.get("gr00t_state_key", "observation.state")).strip()
        or "observation.state",
        gr00t_language_key=str(
            raw.get("gr00t_language_key", "annotation.human.task_description")
        ).strip()
        or "annotation.human.task_description",
        gr00t_video_keys=_parse_policy_camera_keys(raw.get("gr00t_video_keys")),
        gr00t_image_key_map=_parse_str_str_map(raw.get("gr00t_image_key_map")),
        dm05_host=str(raw.get("dm05_host", "127.0.0.1")).strip() or "127.0.0.1",
        dm05_port=int(raw.get("dm05_port", 7891)),
        dm05_timeout_sec=float(raw.get("dm05_timeout_sec", 120.0)),
        dm05_history_slots=int(raw.get("dm05_history_slots", 32)),
        dm05_history_hz=float(raw.get("dm05_history_hz", 1.0)),
        dm05_history_tolerance_sec=float(raw.get("dm05_history_tolerance_sec", 0.2)),
        dm05_robot_type=str(raw.get("dm05_robot_type", "tong_dual_arm")).strip()
        or "tong_dual_arm",
        dm05_speed=str(raw.get("dm05_speed", "0.5")),
        dm05_send_history=bool(raw.get("dm05_send_history", True)),
        prompt_from_subtask=bool(raw.get("prompt_from_subtask", False)),
        vlm_planner=_parse_vlm_planner(raw.get("vlm_planner")),
    )


def _parse_walloss_ik(raw: Any, ptype: PolicyType) -> Optional[Dict[str, Any]]:
    if raw is None:
        return None
    if ptype not in (PolicyType.WALLOSS, PolicyType.TAU0WM, PolicyType.GR00T):
        return None
    ik_dir = Path(__file__).resolve().parents[2] / "recipes" / "walloss"
    if str(ik_dir) not in sys.path:
        sys.path.insert(0, str(ik_dir))
    from ik_config import parse_ik_config

    return parse_ik_config(raw).to_dict()


def _parse_vlm_planner(raw: Any) -> Optional[VlmPlannerConfig]:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("policy.vlm_planner 必须是字典或 null")
    enabled_raw = raw.get("enabled")
    enabled: Optional[bool]
    if enabled_raw is None:
        enabled = None
    else:
        enabled = bool(enabled_raw)
    weight = raw.get("weight_dir") or "/home/xk/VLA/openpi/models/e2_subtask_ce_best"
    tok = raw.get("tokenizer_path") or "/home/xk/VLA/openpi/models/pi05_fixed/tokenizer.model"
    tmpl = raw.get("prompt_template")
    return VlmPlannerConfig(
        enabled=enabled,
        weight_dir=Path(str(weight)).expanduser(),
        tokenizer_path=Path(str(tok)).expanduser(),
        device=str(raw.get("device") or "cuda:0").strip() or "cuda:0",
        camera=str(raw.get("camera") or "front").strip() or "front",
        interval_sec=float(raw.get("interval_sec", 1.0)),
        max_new_tokens=int(raw.get("max_new_tokens", 32)),
        prompt_template=(
            str(tmpl)
            if tmpl
            else "Task: {task}\nWhat is the current subtask?\nSubtask:"
        ),
    )


def _parse_str_str_map(raw: Any) -> Optional[Dict[str, str]]:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("policy.gr00t_image_key_map 必须是字符串到字符串的映射，或 null")
    out: Dict[str, str] = {}
    for key, value in raw.items():
        k = str(key).strip()
        v = str(value).strip()
        if not k or not v:
            raise ValueError("policy.gr00t_image_key_map 的 key/value 不能为空")
        out[k] = v
    return out or None


def _parse_openpi_camera_names(raw: Any) -> List[str]:
    if raw is None:
        return ["left", "front", "right"]
    if not isinstance(raw, list) or not raw:
        raise ValueError("policy.openpi_camera_names 必须是非空字符串列表")
    names: List[str] = []
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("policy.openpi_camera_names 每项须为非空字符串")
        names.append(item.strip().lower())
    return names


def _parse_openpi_vector_norm(raw: Any) -> Optional[str]:
    from infer.policies.openpi_pi05_runtime import canonicalize_vector_norm

    return canonicalize_vector_norm(raw)


def _parse_openpi_delta_action_dims(raw: Any) -> Optional[List[int]]:
    if raw is None:
        return None
    if not isinstance(raw, list) or not raw:
        raise ValueError("policy.openpi_delta_action_dims 必须是非空整数列表，或 null")
    dims: List[int] = []
    for item in raw:
        try:
            dims.append(int(item))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"policy.openpi_delta_action_dims 每项须为整数，得到 {item!r}"
            ) from exc
    return dims


def _parse_policy_camera_keys(raw: Any) -> Optional[List[str]]:
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValueError("policy.camera_keys 必须是字符串列表或 null")
    keys: List[str] = []
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("policy.camera_keys 每项须为非空字符串")
        keys.append(item.strip())
    return keys


def _parse_camera_devices(raw_devices: Any) -> List[CameraDeviceConfig]:
    if raw_devices is None:
        return []
    if not isinstance(raw_devices, list):
        raise ValueError("camera_sender.devices 必须是列表")

    devices: List[CameraDeviceConfig] = []
    for idx, item in enumerate(raw_devices):
        if item is None:
            raise ValueError(
                f"camera_sender.devices[{idx}] 为 null；"
                "单相机请直接删除第二路，不要用 `- null` 占位"
            )
        if not isinstance(item, dict):
            raise ValueError(
                f"camera_sender.devices[{idx}] 必须是映射，当前为 {type(item).__name__}"
            )
        serial = item.get("serial")
        if serial is None or not str(serial).strip():
            raise ValueError(f"camera_sender.devices[{idx}] 缺少 serial")
        devices.append(
            CameraDeviceConfig(
                serial=str(serial).strip(),
                key=str(item["key"]).strip() if item.get("key") else None,
                width=int(item["width"]) if item.get("width") is not None else None,
                height=int(item["height"]) if item.get("height") is not None else None,
                model_width=int(item["model_width"]) if item.get("model_width") is not None else None,
                model_height=int(item["model_height"]) if item.get("model_height") is not None else None,
                fps=int(item.get("fps", 30)),
            )
        )
    return devices


def _parse_joint_stream(raw: Any) -> JointStreamConfig:
    if raw is None:
        return JointStreamConfig()
    if not isinstance(raw, dict):
        raise ValueError("direct_robot.joint_stream 必须是映射")
    return JointStreamConfig(
        enabled=bool(raw.get("enabled", False)),
        hz=float(raw.get("hz", 100.0)),
        high_follow=bool(raw.get("high_follow", True)),
        trajectory_mode=int(raw.get("trajectory_mode", 0)),
        radio=int(raw.get("radio", 0)),
        slew_deg_per_sec=float(raw.get("slew_deg_per_sec", 90.0)),
    )


def _merge_sync_clock_offset(sync: Dict[str, Any]) -> Dict[str, float]:
    from common.sync_calibration import load_sync_calibration, merge_clock_offsets

    base = {str(k): float(v) for k, v in sync.get("clock_offset", {}).items()}
    cal_file = sync.get("calibration_file")
    shared = load_sync_calibration(cal_file) if cal_file else load_sync_calibration()
    return merge_clock_offsets(base, shared)


def _parse_ablation_run(
    raw: Dict[str, Any],
    *,
    default_name: str,
    default_group: str,
    default_enabled: bool = True,
    is_baseline: bool = False,
) -> AblationRunSpec:
    return AblationRunSpec(
        name=str(raw.get("name", default_name)),
        overrides=dict(raw.get("overrides", {})),
        description=str(raw.get("description", "")),
        enabled=bool(raw.get("enabled", default_enabled)),
        group=str(raw.get("group", default_group)),
        is_baseline=bool(raw.get("is_baseline", is_baseline)),
    )


def _parse_ablation_group_runs(ablation: Dict[str, Any]) -> List[AblationRunSpec]:
    """Parse baseline + grouped ablation items into a flat, execution-ordered run list."""
    runs: List[AblationRunSpec] = []
    groups = ablation.get("groups") or {}

    baseline_raw = ablation.get("baseline")
    if isinstance(baseline_raw, dict):
        baseline = _parse_ablation_run(
            baseline_raw,
            default_name="baseline",
            default_group="baseline",
            is_baseline=True,
        )
        if baseline.enabled:
            runs.append(baseline)
    elif groups:
        runs.append(
            AblationRunSpec(
                name="baseline",
                description="Base config without ablation overrides",
                group="baseline",
                is_baseline=True,
            )
        )

    if isinstance(groups, dict):
        group_iter = groups.items()
    elif isinstance(groups, list):
        group_iter = ((str(g.get("name", f"group_{i + 1}")), g) for i, g in enumerate(groups) if isinstance(g, dict))
    else:
        group_iter = []

    for group_name, group_raw in group_iter:
        if not isinstance(group_raw, dict) or not bool(group_raw.get("enabled", True)):
            continue
        for idx, item in enumerate(group_raw.get("items", [])):
            if not isinstance(item, dict) or not bool(item.get("enabled", True)):
                continue
            runs.append(
                _parse_ablation_run(
                    item,
                    default_name=f"{group_name}_{idx + 1}",
                    default_group=str(group_name),
                )
            )

    return runs


def explain_config_sources(config_path: str | Path) -> Dict[str, Any]:
    """
    说明主配置与 local_env 合并后，关键字段实际来自哪里。

    local_env 通过 deep_merge **覆盖** 主配置同名字段；只改 pi05_left_arm_direct.yaml
    而 local_env.yaml 里仍有 ckpt_dir 时，生效的是 local_env 的值。
    """
    config_path = _resolve_config_path(config_path)
    main_raw = _load_yaml(config_path)
    local_env_ref = main_raw.get("local_env")
    local_path: Optional[Path] = None
    local_raw: Dict[str, Any] = {}
    if local_env_ref:
        local_path = _resolve_sidecar_path(config_path, str(local_env_ref))
        if local_path.is_file():
            local_raw = _load_yaml(local_path)

    merged = _apply_local_env_config(config_path, main_raw)
    main_policy = main_raw.get("policy") or {}
    local_policy = local_raw.get("policy") or {}
    merged_policy = merged.get("policy") or {}
    main_bi = main_raw.get("batch_infer") or {}
    local_bi = local_raw.get("batch_infer") or {}
    merged_bi = merged.get("batch_infer") or {}

    def _field_source(
        field: str,
        *,
        section: str,
        main_sec: Dict[str, Any],
        local_sec: Dict[str, Any],
        merged_sec: Dict[str, Any],
    ) -> str:
        if local_path is not None and field in local_sec and local_sec.get(field) is not None:
            if main_sec.get(field) != local_sec.get(field):
                return f"local_env ({local_path.name})"
            return f"local_env ({local_path.name})，与主配置相同"
        if field in main_sec and main_sec.get(field) is not None:
            return f"主配置 ({config_path.name})"
        return "未配置"

    ckpt_src = _field_source(
        "ckpt_dir",
        section="policy",
        main_sec=main_policy,
        local_sec=local_policy,
        merged_sec=merged_policy,
    )
    tpg_src = "未配置"
    if merged_bi.get("tasks_per_group") is not None:
        tpg_src = _field_source(
            "tasks_per_group",
            section="batch_infer",
            main_sec=main_bi,
            local_sec=local_bi,
            merged_sec=merged_bi,
        )
    elif merged_policy.get("tasks_per_group") is not None:
        tpg_src = _field_source(
            "tasks_per_group",
            section="policy",
            main_sec=main_policy,
            local_sec=local_policy,
            merged_sec=merged_policy,
        )

    warnings: List[str] = []
    if (
        local_path is not None
        and local_policy.get("ckpt_dir") is not None
        and main_policy.get("ckpt_dir") is not None
        and str(local_policy.get("ckpt_dir")) != str(main_policy.get("ckpt_dir"))
    ):
        warnings.append(
            f"policy.ckpt_dir 已被 {local_path.name} 覆盖；"
            f"主配置={main_policy.get('ckpt_dir')} → 实际={merged_policy.get('ckpt_dir')}"
        )

    effective_tpg = merged_bi.get("tasks_per_group")
    if effective_tpg is None:
        effective_tpg = merged_policy.get("tasks_per_group")

    return {
        "config_path": str(config_path),
        "local_env_path": str(local_path) if local_path else None,
        "main_ckpt_dir": main_policy.get("ckpt_dir"),
        "local_env_ckpt_dir": local_policy.get("ckpt_dir"),
        "effective_ckpt_dir": merged_policy.get("ckpt_dir"),
        "ckpt_dir_source": ckpt_src,
        "effective_tasks_per_group": effective_tpg,
        "tasks_per_group_source_hint": tpg_src,
        "warnings": warnings,
    }


def load_infer_config(path: str | Path) -> InferConfig:
    config_path = _resolve_config_path(path)
    raw = _apply_controller_config(_apply_local_env_config(config_path, _load_yaml(config_path)))
    from infer.policies.pi05_embodiment import apply_pi05_embodiment_yaml

    raw = apply_pi05_embodiment_yaml(raw)

    net = raw.get("network", {})
    cam_sender = raw.get("camera_sender", {})
    direct = raw.get("direct_robot", {})
    sync = raw.get("sync", {})
    merged_clock_offset = _merge_sync_clock_offset(sync)
    rec = raw.get("recording", {})
    sess = raw.get("session", {})
    log = raw.get("logging", {})
    ev = raw.get("evaluation", {})
    ab = raw.get("ablation", {})
    bi = raw.get("batch_infer", {})
    pedal_raw = raw.get("pedal", {})
    exo_raw = raw.get("exo_safety", {})

    ablation_runs = _parse_ablation_group_runs(ab)
    for item in ab.get("runs", []):
        if not isinstance(item, dict):
            continue
        spec = _parse_ablation_run(
            item,
            default_name=str(item.get("name", f"run_{len(ablation_runs) + 1}")),
            default_group=str(item.get("group", "manual")),
        )
        if spec.enabled:
            ablation_runs.append(spec)

    engine_str = str(raw.get("engine", "legacy")).lower()
    if engine_str not in (EngineMode.LEGACY.value, EngineMode.NATIVE.value):
        raise ValueError(f"未知 engine: {engine_str}，可选: legacy/native")
    engine = EngineMode(engine_str)

    policy_cfg = _parse_policy(raw.get("policy", {}))
    session_tasks_per_group = (
        policy_cfg.tasks_per_group
        if policy_cfg.tasks_per_group is not None
        else sess.get("tasks_per_group")
    )
    if session_tasks_per_group is not None:
        session_tasks_per_group = int(session_tasks_per_group)

    task_mode = str(sess.get("task_mode", "test"))
    if sess.get("pedal_task_control") is not None:
        pedal_task_control = bool(sess.get("pedal_task_control"))
    elif session_tasks_per_group is not None and task_mode == "continuous_tasks":
        # policy.tasks_per_group + continuous_tasks 时默认启用脚踏板，否则 tasks_per_group 无法生效
        pedal_task_control = True
    else:
        pedal_task_control = False

    pedal_binding = _parse_pedal_keys(pedal_raw)
    exo_safety_cfg = ExoSafetyConfig(
        enabled=bool(exo_raw.get("enabled", False)),
        takeover_key=str(
            exo_raw.get("takeover_key", pedal_binding.exo_takeover)
        ).strip(),
        safety_infer_key=str(
            exo_raw.get("safety_infer_key", pedal_binding.exo_safety_infer)
        ).strip(),
        exo_state_port=int(exo_raw.get("exo_state_port", 18140)),
        exo_command_port=int(exo_raw.get("exo_command_port", 30026)),
        collect_config_path=_resolve_optional_sidecar_path(
            config_path,
            exo_raw.get("collect_config"),
        ),
        record_start_delay_sec=float(exo_raw.get("record_start_delay_sec", 2.0)),
        collect_cam_mid_window_name=str(
            exo_raw.get("collect_cam_mid_window_name", "exo_collect_cam_mid")
        ).strip()
        or "exo_collect_cam_mid",
        align_enabled=bool(exo_raw.get("align_enabled", True)),
        align_mode=str(exo_raw.get("align_mode", "auto")).strip().lower() or "auto",
        align_max_joint_err_deg=float(exo_raw.get("align_max_joint_err_deg", 12.0)),
        align_rms_joint_err_deg=float(exo_raw.get("align_rms_joint_err_deg", 8.0)),
        align_log_every_sec=float(exo_raw.get("align_log_every_sec", 1.0)),
        align_target_port=int(exo_raw.get("align_target_port", 30027)),
        align_drive_max_step_deg=float(exo_raw.get("align_drive_max_step_deg", 2.0)),
        align_target_send_hz=float(exo_raw.get("align_target_send_hz", 50.0)),
    )
    if exo_safety_cfg.enabled:
        pedal_binding = PedalKeyBinding(
            task_toggle=pedal_binding.task_toggle,
            confirm=pedal_binding.confirm,
            cancel=pedal_binding.cancel,
            discard=pedal_binding.discard,
            exo_takeover=exo_safety_cfg.takeover_key,
            exo_safety_infer=exo_safety_cfg.safety_infer_key,
        )
        user_hotkey_to_pynput(pedal_binding.exo_takeover)
        user_hotkey_to_pynput(pedal_binding.exo_safety_infer)

    return InferConfig(
        experiment_name=str(raw.get("experiment_name", "infer_experiment")),
        engine=engine,
        policy=policy_cfg,
        network=NetworkConfig(
            high_host=str(net.get("high_host", "192.168.1.12")),
            low_host=str(net.get("low_host", "192.168.1.13")),
            state_port=int(net.get("state_port", 18132)),
            control_port=int(net.get("control_port", 30005)),
            gripper_port=int(net.get("gripper_port", 30015)),
            camera_port=int(net.get("camera_port", 18888)),
            gripper_state_port=int(net.get("gripper_state_port", 30016)),
            gripper_state_stale_sec=float(net.get("gripper_state_stale_sec", 1.0)),
        ),
        camera_sender=CameraSenderConfig(
            enabled=bool(cam_sender.get("enabled", False)),
            transport=str(cam_sender.get("transport", "local")),
            target_host=cam_sender.get("target_host"),
            target_port=int(cam_sender["target_port"]) if cam_sender.get("target_port") else None,
            jpeg_quality=int(cam_sender.get("jpeg_quality", 95)),
            reconnect_delay_sec=float(cam_sender.get("reconnect_delay_sec", 0.5)),
            get_timeout_sec=float(cam_sender.get("get_timeout_sec", 3.0)),
            use_host_timestamp=bool(cam_sender.get("use_host_timestamp", True)),
            enable_depth=bool(cam_sender.get("enable_depth", True)),
            devices=_parse_camera_devices(cam_sender.get("devices")),
        ),
        direct_robot=DirectRobotConfig(
            enabled=bool(direct.get("enabled", False)),
            sdk_path=Path(direct["sdk_path"]).expanduser() if direct.get("sdk_path") else None,
            gripper_mode=str(direct.get("gripper_mode", "udp")),
            gripper_state_source=str(direct.get("gripper_state_source", "udp")),
            gripper_initial_position=[
                float(x) for x in direct.get("gripper_initial_position", [0.7, 0.7])
            ],
            robot_r_host=str(direct.get("robot_r_host", "192.168.1.17")),
            robot_l_host=str(direct.get("robot_l_host", "192.168.1.18")),
            port_r=int(direct.get("port_r", 8080)),
            port_l=int(direct.get("port_l", 8080)),
            enable_joint=list(direct.get("enable_joint", [False, False, False])),
            left_gripper_kwargs=direct.get(
                "left_gripper_kwargs",
                {
                    "dxl_name": "XM430",
                    "Device_name": "/dev/ttyUSB0",
                    "dxl_id": 1,
                    "baudrate": 4000000,
                    "min_position": 1395,
                    "max_position": 2043,
                },
            ),
            right_gripper_kwargs=direct.get("right_gripper_kwargs"),
            state_poll_hz=float(direct.get("state_poll_hz", 100.0)),
            reset_to_init_on_group_start=bool(
                direct.get("reset_to_init_on_group_start", False)
            ),
            standard_init_joint_degrees=(
                [float(x) for x in direct["standard_init_joint_degrees"]]
                if direct.get("standard_init_joint_degrees") is not None
                else None
            ),
            reset_move_speed=int(direct.get("reset_move_speed", 10)),
            reset_settle_sec=float(direct.get("reset_settle_sec", 1.5)),
            init_pose_source=str(direct.get("init_pose_source", "standard")),
            random_task_init_pose=bool(direct.get("random_task_init_pose", False)),
            random_task_init_radius_m=float(direct.get("random_task_init_radius_m", 0.10)),
            random_task_init_max_attempts=int(direct.get("random_task_init_max_attempts", 32)),
            reset_to_init_on_task_start=bool(
                direct.get("reset_to_init_on_task_start", False)
            ),
            mirror_right_init_from_left=bool(
                direct.get("mirror_right_init_from_left", False)
            ),
            joint_stream=_parse_joint_stream(direct.get("joint_stream")),
        ),
        exo_safety=exo_safety_cfg,
        pico_recovery=PicoRecoveryConfig(**raw.get("pico_recovery", {})),
        sync=SyncConfig(
            max_time_diff=sync.get("max_time_diff"),
            sync_delay=float(sync.get("sync_delay", 0.05)),
            buffer_size=int(sync.get("buffer_size", 1000)),
            prefill_sec=float(sync.get("prefill_sec", 2.0)),
            min_anchor_buffer=int(sync.get("min_anchor_buffer", 5)),
            clock_offset=merged_clock_offset,
        ),
        recording=RecordingConfig(
            enabled=bool(rec.get("enabled", True)),
            save_videos=bool(rec.get("save_videos", True)),
            save_trajectory=bool(rec.get("save_trajectory", True)),
            save_third_cam=bool(rec.get("save_third_cam", True)),
            save_fourth_cam=bool(rec.get("save_fourth_cam", True)),
            save_depth_videos=bool(rec.get("save_depth_videos", True)),
            save_depth_raw=bool(rec.get("save_depth_raw", True)),
            inference_dir=Path(rec["inference_dir"]).expanduser() if rec.get("inference_dir") else None,
            generate_debug_plot=bool(rec.get("generate_debug_plot", True)),
            debug_plot_max_dims=int(rec.get("debug_plot_max_dims", 8)),
            debug_plot_dpi=int(rec.get("debug_plot_dpi", 160)),
            golden_capture=bool(rec.get("golden_capture", False)),
            golden_capture_dir=Path(rec["golden_capture_dir"]).expanduser()
            if rec.get("golden_capture_dir")
            else None,
            golden_capture_mode=str(rec.get("golden_capture_mode", "policy")),
        ),
        session=SessionConfig(
            fps=int(sess.get("fps", 15)),
            max_tests=sess.get("max_tests"),
            max_duration_sec=sess.get("max_duration_sec", 30.0),
            prompt_continue=bool(sess.get("prompt_continue", True)),
            auto_finalize=bool(sess.get("auto_finalize", True)),
            task_mode=task_mode,
            visualize_cam_left=bool(sess.get("visualize_cam_left", False)),
            cam_left_window_name=str(sess.get("cam_left_window_name", "cam_left")),
            visualize_cam_mid=bool(sess.get("visualize_cam_mid", False)),
            cam_mid_window_name=str(sess.get("cam_mid_window_name", "cam_mid")),
            cam_mid_topmost=bool(sess.get("cam_mid_topmost", True)),
            visualize_cam_right=bool(sess.get("visualize_cam_right", False)),
            cam_right_window_name=str(sess.get("cam_right_window_name", "cam_right")),
            pedal_task_control=pedal_task_control,
            tasks_per_group=session_tasks_per_group,
            interactive_task_prompt=bool(sess.get("interactive_task_prompt", False)),
            eval_prompt_on_marker=bool(sess.get("eval_prompt_on_marker", True)),
            eval_prompt_timeout_sec=float(sess.get("eval_prompt_timeout_sec", 0.0)),
            replan_steps=(
                int(sess["replan_steps"]) if sess.get("replan_steps") is not None else None
            ),
        ),
        logging=LoggingConfig(
            level=str(log.get("level", "INFO")),
            log_file=Path(log["log_file"]).expanduser() if log.get("log_file") else None,
            machine_flow_enabled=bool(log.get("machine_flow_enabled", True)),
            alerts_enabled=bool(log.get("alerts_enabled", True)),
            alerts_path=Path(log["alerts_path"]).expanduser() if log.get("alerts_path") else None,
            human_log_level=log.get("human_log_level"),
            file_log_level=log.get("file_log_level"),
            log_sync_every_n_frames=int(log.get("log_sync_every_n_frames", 30)),
            log_action_steps_detail=bool(log.get("log_action_steps_detail", False)),
            device_health_enabled=bool(log.get("device_health_enabled", True)),
            device_health_sample_interval_sec=float(
                log.get("device_health_sample_interval_sec", 1.0)
            ),
            device_health_stale_state_sec=float(log.get("device_health_stale_state_sec", 0.5)),
            device_health_stale_camera_sec=float(
                log.get("device_health_stale_camera_sec", 1.0)
            ),
            device_health_ping_low_host=bool(log.get("device_health_ping_low_host", True)),
            device_health_pause_on_anomaly=bool(log.get("device_health_pause_on_anomaly", True)),
            device_health_pause_require_confirm=bool(
                log.get("device_health_pause_require_confirm", True)
            ),
            device_health_pause_hold_sec=float(log.get("device_health_pause_hold_sec", 1.5)),
            device_health_resume_hold_sec=float(log.get("device_health_resume_hold_sec", 1.0)),
            device_health_pause_on_gripper_udp_stale=bool(
                log.get("device_health_pause_on_gripper_udp_stale", True)
            ),
            device_health_pause_on_gripper_tracking_error=bool(
                log.get("device_health_pause_on_gripper_tracking_error", True)
            ),
            device_health_gripper_tracking_err=float(
                log.get("device_health_gripper_tracking_err", 0.25)
            ),
            device_health_gripper_tracking_consecutive=int(
                log.get("device_health_gripper_tracking_consecutive", 12)
            ),
            device_health_pause_on_low_host_unreachable=bool(
                log.get("device_health_pause_on_low_host_unreachable", True)
            ),
            device_health_pause_on_state_stale=bool(
                log.get("device_health_pause_on_state_stale", True)
            ),
            device_health_pause_on_camera_stale=bool(
                log.get("device_health_pause_on_camera_stale", True)
            ),
            device_health_pause_warmup_sec=float(log.get("device_health_pause_warmup_sec", 5.0)),
            device_health_live_status_ui=bool(log.get("device_health_live_status_ui", True)),
            device_health_live_status_window_name=str(
                log.get("device_health_live_status_window_name", "device_health")
            ),
        ),
        evaluation=EvaluationConfig(
            enabled=bool(ev.get("enabled", True)),
            auto_after_each_test=bool(ev.get("auto_after_each_test", True)),
            save_report=bool(ev.get("save_report", True)),
            report_filename=str(ev.get("report_filename", "evaluation_report.json")),
            metrics=list(ev.get("metrics", ["automatic"])),
            labels_file=Path(ev["labels_file"]).expanduser() if ev.get("labels_file") else None,
            sync_warn_rate_threshold=float(ev.get("sync_warn_rate_threshold", 0.30)),
            eval_log_enabled=bool(ev.get("eval_log_enabled", True)),
            eval_log_filename=str(ev.get("eval_log_filename", "eval_log.jsonl")),
            eval_log_save_images=bool(ev.get("eval_log_save_images", False)),
            eval_log_image_dir=str(ev.get("eval_log_image_dir", "eval_images")),
            eval_log_image_format=str(ev.get("eval_log_image_format", "jpg")),
            eval_log_save_videos=bool(ev.get("eval_log_save_videos", False)),
            eval_log_video_dir=str(ev.get("eval_log_video_dir", "eval_videos")),
            eval_log_generate_plots=bool(ev.get("eval_log_generate_plots", True)),
            eval_log_plot_dir=str(ev.get("eval_log_plot_dir", "eval_plots")),
            eval_log_plot_dpi=int(ev.get("eval_log_plot_dpi", 150)),
        ),
        ablation=AblationConfig(
            enabled=bool(ab.get("enabled", False)),
            experiment_id=ab.get("experiment_id"),
            runs=ablation_runs,
            compare_report_dir=Path(ab["compare_report_dir"]).expanduser()
            if ab.get("compare_report_dir")
            else None,
        ),
        batch_infer=BatchInferConfig(
            action_step_parts=int(bi.get("action_step_parts", 5)),
            action_steps=_parse_int_list(bi.get("action_steps")),
            denoise_steps=_parse_int_list(bi.get("denoise_steps")),
            phase=str(bi.get("phase", "all")),
            output_dir=Path(bi["output_dir"]).expanduser()
            if bi.get("output_dir")
            else None,
            best_action_step=int(bi["best_action_step"])
            if bi.get("best_action_step") is not None
            else None,
            skip_existing=bool(bi.get("skip_existing", False)),
            ood_presets=_resolve_sidecar_path(config_path, bi["ood_presets"])
            if bi.get("ood_presets")
            else None,
            task_prompt_presets=_parse_task_prompt_presets(bi.get("task_prompt_presets")),
            tasks_per_group=int(bi["tasks_per_group"])
            if bi.get("tasks_per_group") is not None
            else None,
            ood_tasks_per_group=int(bi.get("ood_tasks_per_group", 1)),
            feedback_score_sheet=bool(bi.get("feedback_score_sheet", True)),
            vla_auto_eval=_parse_vla_auto_eval(config_path, bi.get("vla_auto_eval", {})),
            remote_sync=_parse_batch_remote_sync(bi.get("remote_sync", {})),
        ),
        pedal=pedal_binding,
        config_path=config_path,
    )


def validate_rtc_option(policy: PolicyConfig) -> None:
    """``rtc_enabled`` 是开关：false 不走 RTC；true 仅 DiT/flow 允许。"""
    if not bool(getattr(policy, "rtc_enabled", False)):
        return
    expected = str(getattr(policy, "lingbot_expected_model", "") or "").strip().lower()
    if expected in {"pi0_fast", "pi0fast"}:
        raise ValueError("π0-FAST 不能开 rtc_enabled（token 生成，不是 flow DiT）")
    ptype = policy.type
    if not policy_supports_rtc_option(ptype):
        allowed = ", ".join(sorted(t.value for t in RTC_OPTIONAL_POLICY_TYPES))
        raise ValueError(
            f"rtc_enabled 仅可作为 {allowed} 的可选项打开，当前是 {ptype.value}"
        )


def validate_infer_config(cfg: InferConfig) -> None:
    remote_weight_types = (
        PolicyType.GR00T,
        PolicyType.LINGBOT_VLA,
        PolicyType.LINGBOT_VA,
        PolicyType.DREAMZERO,
        PolicyType.COSMOS3,
        PolicyType.INTERNVLA,
        PolicyType.MEMORYVLA,
        PolicyType.WALLOSS,
        PolicyType.GIGAWORLD,
        PolicyType.TAU0WM,
        PolicyType.DM05,
    )
    if cfg.policy.type not in remote_weight_types and not cfg.policy.ckpt_dir.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {cfg.policy.ckpt_dir}")
    if cfg.session.task_mode not in ("test", "continuous_tasks"):
        raise ValueError("session.task_mode 只能是 test 或 continuous_tasks")
    if cfg.camera_sender.transport not in ("local", "tcp"):
        raise ValueError("camera_sender.transport 只能是 local 或 tcp")
    if cfg.camera_sender.enabled and not cfg.camera_sender.devices:
        raise ValueError("camera_sender.enabled=true 时必须配置 camera_sender.devices")
    if cfg.policy.camera_keys is not None:
        if len({k for k in cfg.policy.camera_keys}) != len(cfg.policy.camera_keys):
            raise ValueError("policy.camera_keys 不能含重复项")
        for key in cfg.policy.camera_keys:
            if not key.startswith("observation."):
                raise ValueError(
                    f"policy.camera_keys 项须为 observation feature 名: {key!r}"
                )
    if cfg.direct_robot.gripper_mode not in ("udp", "local", "none"):
        raise ValueError("direct_robot.gripper_mode 只能是 udp、local 或 none")
    if cfg.direct_robot.gripper_state_source not in ("udp", "buffer"):
        raise ValueError("direct_robot.gripper_state_source 只能是 udp 或 buffer")
    stream = cfg.direct_robot.joint_stream
    if stream.enabled:
        if stream.hz <= 0 or stream.hz > 500:
            raise ValueError("direct_robot.joint_stream.hz 必须在 (0, 500] 内")
        if stream.high_follow and stream.hz < 100:
            raise ValueError(
                "direct_robot.joint_stream.high_follow 要求 hz >= 100（周期不超过 10ms）"
            )
        if stream.trajectory_mode not in (0, 1, 2):
            raise ValueError("direct_robot.joint_stream.trajectory_mode 只能是 0、1 或 2")
        if stream.trajectory_mode == 1 and not 0 <= stream.radio <= 100:
            raise ValueError("曲线拟合时 direct_robot.joint_stream.radio 必须在 0 到 100")
        if stream.trajectory_mode == 2 and not 0 <= stream.radio <= 999:
            raise ValueError("滤波时 direct_robot.joint_stream.radio 必须在 0 到 999")
        if stream.slew_deg_per_sec <= 0:
            raise ValueError("direct_robot.joint_stream.slew_deg_per_sec 必须大于 0")
    if cfg.policy.type == PolicyType.ACT and cfg.policy.act_backend not in ("lerobot", "aloha"):
        raise ValueError("policy.act_backend 只能是 lerobot 或 aloha")
    if cfg.policy.type == PolicyType.PI05 and cfg.policy.pi05_backend not in (
        "lerobot",
        "openpi",
        "websocket",
    ):
        raise ValueError("policy.pi05_backend 只能是 lerobot、openpi 或 websocket")
    if cfg.policy.rtc_enabled:
        validate_rtc_option(cfg.policy)
        cfg.policy.rtc_config = normalize_rtc_config(
            getattr(cfg.policy, "rtc_config", None), enabled=True
        )
        if cfg.policy.rtc_config is not None:
            cfg.policy.rtc_config["async_blending"] = bool(
                getattr(cfg.policy, "async_blending", False)
            )
            cfg.policy.rtc_config["blend_horizon"] = getattr(
                cfg.policy, "blend_horizon", None
            )
            cfg.policy.rtc_config["blend_schedule"] = str(
                getattr(cfg.policy, "blend_schedule", "linear")
            )
    elif getattr(cfg.policy, "rtc_config", None):
        cfg.policy.rtc_config = None
    if cfg.policy.type == PolicyType.PI05 and cfg.policy.pi05_backend in (
        "openpi",
        "websocket",
    ):
        if int(cfg.policy.openpi_action_dim) < 1:
            raise ValueError("policy.openpi_action_dim 必须 >= 1")
        if getattr(cfg.policy, "openpi_state_dim", None) is not None and int(cfg.policy.openpi_state_dim) < 1:
            raise ValueError("policy.openpi_state_dim 必须 >= 1")
        emb_raw = getattr(cfg.policy, "openpi_embodiment", None)
        if emb_raw:
            from infer.policies.pi05_embodiment import normalize_embodiment_name

            cfg.policy.openpi_embodiment = normalize_embodiment_name(emb_raw)
        from infer.policies.pi05_embodiment import infer_embodiment_name

        emb_name = infer_embodiment_name(
            getattr(cfg.policy, "openpi_embodiment", None),
            getattr(cfg.policy, "openpi_config_name", None),
        )
        from infer.policies.pi05_embodiment import is_sim_embodiment

        if is_sim_embodiment(emb_name) and cfg.direct_robot.enabled:
            raise ValueError(
                f"openpi_embodiment={emb_name} 禁止 direct_robot.enabled=true："
                "仿真动作不能下发 TongBot 真机。"
            )
        if emb_name in {"libero", "robocasa365"} and getattr(cfg.policy, "openpi_adapt_to_pi", None) is True:
            raise ValueError(f"{emb_name} 不能 openpi_adapt_to_pi=true")
        if int(cfg.policy.openpi_action_horizon) < 1:
            raise ValueError("policy.openpi_action_horizon 必须 >= 1")
        if int(cfg.policy.openpi_image_size) < 1:
            raise ValueError("policy.openpi_image_size 必须 >= 1")
        if len(cfg.policy.keep_dims) != int(cfg.policy.openpi_action_dim):
            raise ValueError(
                f"pi05_backend={cfg.policy.pi05_backend} 时 keep_dims 长度 "
                f"({len(cfg.policy.keep_dims)}) 必须等于 openpi_action_dim "
                f"({cfg.policy.openpi_action_dim})"
            )
        if cfg.policy.n_action_steps_override is not None:
            nas = int(cfg.policy.n_action_steps_override)
            horizon = int(cfg.policy.openpi_action_horizon)
            if not 1 <= nas <= horizon:
                raise ValueError(
                    f"n_action_steps_override ({nas}) 须在 1~openpi_action_horizon ({horizon})"
                )
            if cfg.policy.rtc_enabled and nas >= horizon:
                raise ValueError(
                    "rtc_enabled 需要 leftover 做 prefix 约束："
                    f"n_action_steps_override ({nas}) 必须 < openpi_action_horizon ({horizon})"
                )
        if cfg.policy.pi05_backend == "openpi":
            repo = cfg.policy.openpi_repo_path
            if repo is not None:
                repo = repo.expanduser()
                if not repo.is_dir():
                    raise FileNotFoundError(f"openpi_repo_path 不存在: {repo}")
                # Accept either .../openpi-agilex/src or .../openpi-agilex (with src/ inside).
                pkg = repo / "openpi" if (repo / "openpi").is_dir() else repo / "src" / "openpi"
                if not pkg.is_dir():
                    raise FileNotFoundError(
                        f"openpi_repo_path 下未找到 openpi 包: {repo} "
                        "(请指向含 openpi/ 的目录，或 openpi-agilex 仓库根目录；"
                        "生产环境可省略，默认使用 infer/policies/openpi)"
                    )
            else:
                from infer.policies.openpi_pi05_runtime import resolve_openpi_src_path

                resolve_openpi_src_path(None)
            ckpt = cfg.policy.ckpt_dir.expanduser()
            has_params = (ckpt / "params").exists() or any(ckpt.glob("params*"))
            has_pt = (ckpt / "model.safetensors").is_file()
            if ckpt.is_dir() and not (has_params or has_pt):
                raise FileNotFoundError(
                    f"openpi checkpoint 目录缺少 params/ 或 model.safetensors: {ckpt}"
                )
        if cfg.policy.camera_keys is not None and len(cfg.policy.camera_keys) != len(
            cfg.policy.openpi_camera_names
        ):
            raise ValueError(
                f"policy.camera_keys 有 {len(cfg.policy.camera_keys)} 项，"
                f"但 openpi_camera_names 有 {len(cfg.policy.openpi_camera_names)} 项"
            )
        # AlohaInputs(adapt_to_pi=True) 固定写 state[[6,13]]，非 14 维会 IndexError
        if getattr(cfg.policy, "openpi_interleave_grippers", False):
            dims = [int(x) for x in (cfg.policy.openpi_delta_action_dims or [])]
            if dims != [7, -1, 7, -1]:
                raise ValueError(
                    "openpi_interleave_grippers=true 时 openpi_delta_action_dims 必须是 "
                    f"[7, -1, 7, -1]，得到 {dims}"
                )
            if int(cfg.policy.openpi_action_dim) != 16:
                raise ValueError(
                    "openpi_interleave_grippers 只用于 TongBot 16 维，"
                    f"当前 openpi_action_dim={cfg.policy.openpi_action_dim}"
                )
            if getattr(cfg.policy, "openpi_adapt_to_pi", None) is True:
                raise ValueError(
                    "openpi_interleave_grippers=true 时 openpi_adapt_to_pi 必须为 false"
                )
        if (
            getattr(cfg.policy, "openpi_adapt_to_pi", None) is True
            and int(cfg.policy.openpi_action_dim) != 14
        ):
            raise ValueError(
                "openpi_adapt_to_pi=true 要求 openpi_action_dim=14（Aloha 双臂布局）；"
                f"当前为 {cfg.policy.openpi_action_dim}。TongBot 请设 openpi_adapt_to_pi: false，"
                "且 openpi_config_name 须与训练该 checkpoint 时一致。"
            )
    if cfg.policy.type in (PolicyType.ACT, PolicyType.DIFFUSION) and not (
        cfg.policy.type == PolicyType.ACT and cfg.policy.act_backend == "aloha"
    ):
        if not cfg.policy.dataset_repo_id or not cfg.policy.dataset_root:
            raise ValueError("ACT/Diffusion 需配置 policy.dataset_repo_id 与 dataset_root")
        root = cfg.policy.dataset_root.expanduser()
        if not root.is_dir():
            raise FileNotFoundError(f"dataset_root 不存在: {root}")
    if cfg.policy.type in (PolicyType.ALOHA_DP, PolicyType.ALOHA_RDT):
        # Native ALOHA DP/RDT use dataset_stats.pkl beside ckpt; LeRobot dataset optional.
        pass
    if cfg.policy.type == PolicyType.DM05:
        if float(cfg.policy.dm05_history_hz) <= 0:
            raise ValueError("policy.dm05_history_hz 须 > 0")
        if int(cfg.policy.dm05_history_slots) < 1:
            raise ValueError("policy.dm05_history_slots 须 >= 1")
        if cfg.policy.camera_keys is not None and len(cfg.policy.camera_keys) != 3:
            raise ValueError("DM0.5 需要三路相机：头、左腕、右腕")
    if not cfg.policy.task_prompt.strip():
        raise ValueError("policy.task_prompt 不能为空")
    if vlm_planner_enabled(cfg.policy):
        planner = cfg.policy.vlm_planner or VlmPlannerConfig()
        cfg.policy.vlm_planner = planner
        if not (planner.weight_dir / "model.safetensors").is_file():
            raise FileNotFoundError(
                "prompt_from_subtask / vlm_planner 已开启，但上层 VLM 权重不存在: "
                f"{planner.weight_dir / 'model.safetensors'}"
            )
        if not planner.tokenizer_path.is_file():
            raise FileNotFoundError(
                f"上层 VLM tokenizer 不存在: {planner.tokenizer_path}"
            )
    dr = cfg.direct_robot
    if dr.reset_to_init_on_group_start:
        if not dr.enabled:
            raise ValueError(
                "direct_robot.reset_to_init_on_group_start 需要 direct_robot.enabled=true"
            )
        pose = dr.standard_init_joint_degrees or STANDARD_INIT_JOINT_17D_DEG
        if len(pose) != 17:
            raise ValueError(
                "direct_robot.standard_init_joint_degrees 需要 17 个关节角（度）"
            )
        src = str(dr.init_pose_source or "standard").lower()
        if src not in ("standard", "dataset_median", "dataset_mean"):
            raise ValueError(
                "direct_robot.init_pose_source 只能是 standard / dataset_median / dataset_mean"
            )
    if dr.reset_to_init_on_task_start:
        if not dr.enabled:
            raise ValueError(
                "direct_robot.reset_to_init_on_task_start 需要 direct_robot.enabled=true"
            )
        pose = dr.standard_init_joint_degrees or STANDARD_INIT_JOINT_17D_DEG
        if len(pose) != 17:
            raise ValueError(
                "direct_robot.standard_init_joint_degrees 需要 17 个关节角（度）"
            )
        if dr.random_task_init_pose:
            raise ValueError(
                "reset_to_init_on_task_start 与 random_task_init_pose 不能同时开启"
            )
    if dr.random_task_init_pose:
        if not dr.enabled:
            raise ValueError(
                "direct_robot.random_task_init_pose 需要 direct_robot.enabled=true"
            )
        if float(dr.random_task_init_radius_m) < 0:
            raise ValueError("direct_robot.random_task_init_radius_m 不能为负")
        if int(dr.random_task_init_max_attempts) < 1:
            raise ValueError("direct_robot.random_task_init_max_attempts 必须 >= 1")
    pico = cfg.pico_recovery
    if pico.enabled:
        if not dr.enabled or cfg.engine is not EngineMode.NATIVE or not cfg.policy.async_inference:
            raise ValueError("PICO recovery requires native async inference with direct_robot enabled")
        from math import isfinite

        if not isfinite(pico.timeout_sec) or pico.timeout_sec <= 0:
            raise ValueError("pico_recovery.timeout_sec must be positive")
        if (
            not isfinite(pico.first_action_max_joint_delta_deg)
            or pico.first_action_max_joint_delta_deg <= 0
        ):
            raise ValueError("pico_recovery.first_action_max_joint_delta_deg must be positive")
        pico.validate_grippers(dr.gripper_mode)
    if cfg.exo_safety.enabled and not dr.enabled:
        raise ValueError(
            "exo_safety.enabled=true 需要 direct_robot.enabled=true（臂控与接管均走上位机 API）"
        )
    if cfg.exo_safety.enabled:
        collect_path = cfg.exo_safety.collect_config_path
        if collect_path is None:
            raise ValueError(
                "exo_safety.enabled=true 需要配置 collect_config（与 data_collect 相同的 YAML）"
            )
        if not collect_path.is_file():
            raise ValueError(f"exo_safety.collect_config 文件不存在: {collect_path}")
