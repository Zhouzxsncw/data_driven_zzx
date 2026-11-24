"""RoboTG_g 内嵌 Dynamixel 夹爪电机直读（不经过 get_robot_state / 机械臂 SDK）。

兼容 TongGenesis 真实夹爪对象：多数版本不把 kwargs 存成 min_position/max_position 属性，
但提供 read_position_ratio / present_position_read 等接口。读取顺序必须与此对齐，
不能在缺行程属性时直接失败。
"""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple, Optional, Tuple

from gripper_udp_packet import clamp_gripper_ratio

ADDR_PRESENT_CURRENT = 126
ADDR_PRESENT_POSITION = 132
CURRENT_UNIT_MA = {
    "XM430": 2.69,
    "XL330": 1.0,
    "XC330": 1.0,
}

_RATIO_API_NAMES = (
    "read_position_ratio",
    "get_position_ratio",
    "read_ratio",
)

_RAW_API_NAMES = (
    "present_position_read",
    "read_present_position",
    "get_present_position",
    "read_position",
)


class GripperMotorSample(NamedTuple):
    ratio: Optional[float]
    raw_position: Optional[int] = None
    ok: bool = False
    source: str = "none"
    error: str = ""


def _ratio_from_raw(raw: float, min_pos: int, max_pos: int) -> float:
    denom = int(max_pos) - int(min_pos)
    if denom == 0:
        raise ValueError("min_position == max_position，无法换算开口比")
    return float((float(raw) - int(min_pos)) / denom)


def _motor_min_max(
    gripper_obj: Any,
    *,
    calib: Optional[Mapping[str, Any]] = None,
) -> Tuple[Optional[int], Optional[int]]:
    """从对象属性或外部标定取行程；取不到返回 (None, None)，不抛错。"""
    if calib is not None:
        c_min = calib.get("min_position", calib.get("min_pos"))
        c_max = calib.get("max_position", calib.get("max_pos"))
        if c_min is not None and c_max is not None:
            return int(c_min), int(c_max)

    for min_name, max_name in (
        ("min_position", "max_position"),
        ("min_pos", "max_pos"),
        ("position_min", "position_max"),
        ("lower_limit", "upper_limit"),
    ):
        min_pos = getattr(gripper_obj, min_name, None)
        max_pos = getattr(gripper_obj, max_name, None)
        if min_pos is not None and max_pos is not None:
            return int(min_pos), int(max_pos)

    # TongGenesis 部分 servo 用 position_range=[close, open]
    pos_range = getattr(gripper_obj, "position_range", None)
    if isinstance(pos_range, (list, tuple)) and len(pos_range) >= 2:
        return int(pos_range[0]), int(pos_range[1])

    return None, None


def apply_gripper_calibration(gripper_obj: Any, calib: Optional[Mapping[str, Any]]) -> None:
    """把 only_gripper kwargs 行程写到对象上，便于后续直读 raw→ratio。"""
    if gripper_obj is None or not calib:
        return
    min_pos = calib.get("min_position", calib.get("min_pos"))
    max_pos = calib.get("max_position", calib.get("max_pos"))
    if min_pos is None or max_pos is None:
        return
    if getattr(gripper_obj, "min_position", None) is None:
        try:
            setattr(gripper_obj, "min_position", int(min_pos))
        except Exception:
            pass
    if getattr(gripper_obj, "max_position", None) is None:
        try:
            setattr(gripper_obj, "max_position", int(max_pos))
        except Exception:
            pass


def _dxl_name(gripper_obj: Any) -> str:
    for attr in ("dxl_name", "model", "motor_name"):
        val = getattr(gripper_obj, attr, None)
        if val is None:
            continue
        name = str(val).upper()
        for key in CURRENT_UNIT_MA:
            if key in name:
                return key
    return "XM430"


def _signed_int16(raw: int) -> int:
    v = int(raw) & 0xFFFF
    return v - 65536 if v > 32767 else v


def _raw_current_to_mA(raw: int, dxl_name: str) -> float:
    unit = float(CURRENT_UNIT_MA.get(dxl_name, 2.69))
    return float(_signed_int16(raw)) * unit


def _as_scalar(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("present_position 返回空序列")
        return value[0]
    return value


def _looks_like_ratio(value: Any) -> bool:
    """TongGenesis 部分 API 名像 raw，实际已返回 [0,1] 开口比。"""
    try:
        v = float(_as_scalar(value))
    except (TypeError, ValueError):
        return False
    # 真机 DXL raw 一般在几百~几千；浮点且落在 [0,1] 几乎只能是开口比
    return isinstance(_as_scalar(value), float) and 0.0 <= v <= 1.0


def _coerce_raw_position(value: Any) -> int:
    if value is None:
        raise ValueError("present_position 返回 None")
    value = _as_scalar(value)
    if isinstance(value, float) and 0.0 <= float(value) <= 1.0:
        raise ValueError(
            f"present_position 返回疑似开口比 {value}，拒绝当作 raw 使用"
        )
    return int(value)


def _looks_like_failed_raw(raw: int, min_pos: int, max_pos: int) -> bool:
    if int(raw) != 0:
        return False
    lo = min(int(min_pos), int(max_pos))
    return lo >= 100


def read_sample_from_gripper_motor(
    gripper_obj: Any,
    *,
    calib: Optional[Mapping[str, Any]] = None,
) -> GripperMotorSample:
    """读开口比：优先对象自带 ratio API；再 raw+行程；绝不回填指令。"""
    if gripper_obj is None:
        return GripperMotorSample(
            ratio=None, ok=False, source="none", error="gripper_obj is None"
        )

    errors: list[str] = []

    # 1) TongGenesis 常见：对象自己换算好的开口比（不依赖属性名）
    for name in _RATIO_API_NAMES:
        fn = getattr(gripper_obj, name, None)
        if not callable(fn):
            continue
        try:
            ratio = clamp_gripper_ratio(float(fn()))
            return GripperMotorSample(
                ratio=ratio, raw_position=None, ok=True, source=name, error=""
            )
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    min_pos, max_pos = _motor_min_max(gripper_obj, calib=calib)

    # 2) present_position_read 等：有的返回 raw int，有的直接返回开口比 float
    for name in _RAW_API_NAMES:
        fn = getattr(gripper_obj, name, None)
        if not callable(fn):
            continue
        try:
            value = fn()
            if _looks_like_ratio(value):
                ratio = clamp_gripper_ratio(float(_as_scalar(value)))
                return GripperMotorSample(
                    ratio=ratio,
                    raw_position=None,
                    ok=True,
                    source=f"{name}_as_ratio",
                    error="",
                )
            raw = _coerce_raw_position(value)
            if min_pos is None or max_pos is None:
                errors.append(f"{name}: raw={raw} 但无行程标定")
                continue
            if _looks_like_failed_raw(raw, min_pos, max_pos):
                errors.append(f"{name}: raw=0 像通信失败")
                continue
            ratio = clamp_gripper_ratio(_ratio_from_raw(raw, min_pos, max_pos))
            return GripperMotorSample(
                ratio=ratio, raw_position=raw, ok=True, source=name, error=""
            )
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    # 3) 直读 DXL 表 Present_Position@132
    packet_handler = getattr(gripper_obj, "packet_handler", None)
    port_handler = getattr(
        gripper_obj, "port_handler", getattr(gripper_obj, "portHandler", None)
    )
    dxl_id = getattr(gripper_obj, "dxl_id", getattr(gripper_obj, "id", None))
    if packet_handler is None or port_handler is None or dxl_id is None:
        errors.append("无 packet_handler/port_handler/dxl_id，无法直读 DXL 表")
    elif min_pos is None or max_pos is None:
        errors.append("有 DXL 句柄但无行程标定，无法换算开口比")
    else:
        read_fn = getattr(packet_handler, "read4ByteTxRx", None)
        if not callable(read_fn):
            errors.append("packet_handler 无 read4ByteTxRx")
        else:
            try:
                value, comm_result, dxl_error = read_fn(
                    port_handler, int(dxl_id), ADDR_PRESENT_POSITION
                )
                comm_ok = getattr(gripper_obj, "_comm_success", 0)
                if dxl_error != 0:
                    errors.append(f"dxl_table: dxl_error={dxl_error}")
                elif comm_result != comm_ok:
                    errors.append(f"dxl_table: comm_result={comm_result}")
                else:
                    raw = int(value)
                    if _looks_like_failed_raw(raw, min_pos, max_pos):
                        errors.append("dxl_table: raw=0 像通信失败")
                    else:
                        ratio = clamp_gripper_ratio(
                            _ratio_from_raw(raw, min_pos, max_pos)
                        )
                        return GripperMotorSample(
                            ratio=ratio,
                            raw_position=raw,
                            ok=True,
                            source="dxl_table",
                            error="",
                        )
            except Exception as exc:
                errors.append(f"dxl_table: {exc}")

    if min_pos is None or max_pos is None:
        errors.append(
            "无行程标定（对象无 min_position/max_position，也未传入 calib）"
        )

    return GripperMotorSample(
        ratio=None,
        raw_position=None,
        ok=False,
        source="none",
        error="; ".join(errors) if errors else "Present_Position 不可用",
    )


def read_ratio_from_gripper_motor(
    gripper_obj: Any,
    *,
    calib: Optional[Mapping[str, Any]] = None,
) -> Optional[float]:
    sample = read_sample_from_gripper_motor(gripper_obj, calib=calib)
    return sample.ratio if sample.ok else None


def read_current_mA_from_gripper_motor(gripper_obj: Any) -> Optional[float]:
    """读 Present_Current，单位 mA；失败返回 None（电流可选，不替代位置）。"""
    if gripper_obj is None:
        return None

    for name in (
        "read_current_mA",
        "read_present_current_mA",
        "get_present_current_mA",
    ):
        fn = getattr(gripper_obj, name, None)
        if callable(fn):
            try:
                return float(fn())
            except Exception:
                continue

    for name in (
        "read_present_current",
        "present_current_read",
        "get_present_current",
        "read_current",
    ):
        fn = getattr(gripper_obj, name, None)
        if not callable(fn):
            continue
        try:
            raw = fn()
            if isinstance(raw, float) and abs(raw) > 50:
                return float(raw)
            return _raw_current_to_mA(int(raw), _dxl_name(gripper_obj))
        except Exception:
            continue

    packet_handler = getattr(gripper_obj, "packet_handler", None)
    port_handler = getattr(
        gripper_obj, "port_handler", getattr(gripper_obj, "portHandler", None)
    )
    dxl_id = getattr(gripper_obj, "dxl_id", getattr(gripper_obj, "id", None))
    if packet_handler is None or port_handler is None or dxl_id is None:
        return None
    read_fn = getattr(packet_handler, "read2ByteTxRx", None)
    if not callable(read_fn):
        return None
    try:
        value, comm_result, dxl_error = read_fn(
            port_handler, int(dxl_id), ADDR_PRESENT_CURRENT
        )
        if dxl_error != 0:
            return None
        comm_ok = getattr(gripper_obj, "_comm_success", 0)
        if comm_result != comm_ok:
            return None
        return _raw_current_to_mA(int(value), _dxl_name(gripper_obj))
    except Exception:
        return None


def read_gripper_ratios_from_robot(
    robot: Any,
    *,
    left_calib: Optional[Mapping[str, Any]] = None,
    right_calib: Optional[Mapping[str, Any]] = None,
) -> Tuple[float, float]:
    """只返回左右实测开口比；任一爪读失败则抛错，禁止 default/指令兜底。"""
    left_s = read_sample_from_gripper_motor(
        getattr(robot, "left_gripper", None), calib=left_calib
    )
    right_s = read_sample_from_gripper_motor(
        getattr(robot, "right_gripper", None), calib=right_calib
    )
    if not left_s.ok or left_s.ratio is None:
        raise RuntimeError(f"左夹爪 Present_Position 读取失败: {left_s.error}")
    if not right_s.ok or right_s.ratio is None:
        raise RuntimeError(f"右夹爪 Present_Position 读取失败: {right_s.error}")
    return float(left_s.ratio), float(right_s.ratio)


def read_gripper_state_from_robot(
    robot: Any,
    *,
    read_current: bool = True,
    left_calib: Optional[Mapping[str, Any]] = None,
    right_calib: Optional[Mapping[str, Any]] = None,
) -> Tuple[float, float, Optional[float], Optional[float]]:
    left, right = read_gripper_ratios_from_robot(
        robot, left_calib=left_calib, right_calib=right_calib
    )
    if not read_current:
        return left, right, None, None
    left_i = read_current_mA_from_gripper_motor(getattr(robot, "left_gripper", None))
    right_i = read_current_mA_from_gripper_motor(getattr(robot, "right_gripper", None))
    return left, right, left_i, right_i


def read_gripper_state_samples_from_robot(
    robot: Any,
    *,
    read_current: bool = True,
    left_calib: Optional[Mapping[str, Any]] = None,
    right_calib: Optional[Mapping[str, Any]] = None,
) -> Tuple[GripperMotorSample, GripperMotorSample, Optional[float], Optional[float]]:
    left = read_sample_from_gripper_motor(
        getattr(robot, "left_gripper", None), calib=left_calib
    )
    right = read_sample_from_gripper_motor(
        getattr(robot, "right_gripper", None), calib=right_calib
    )
    if not read_current:
        return left, right, None, None
    left_i = read_current_mA_from_gripper_motor(getattr(robot, "left_gripper", None))
    right_i = read_current_mA_from_gripper_motor(getattr(robot, "right_gripper", None))
    return left, right, left_i, right_i
