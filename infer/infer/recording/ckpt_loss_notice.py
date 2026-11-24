"""推理启动时，把这一档 checkpoint 的训练 loss 打给操作员。

数字和闸门与 docs/kb/pull_loss_gate.yaml 对齐（2026-09-29）。
只显示，不拦截开跑。纯文本进 run.log；ANSI 只给终端。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# 实验根再往上就是整盘 checkpoint/，不要扫到别的实验。
_STOP_DIR_NAMES = {
    "checkpoint",
    "checkpoints",
    "ckpt",
    "ckpts",
    "media",
    "mnt",
    "home",
    "data",
    "xk",
}
_STRUCTURED = (
    "trainer_state.json",
    "train_loss_history.json",
    "loss_probe.jsonl",
)
_LOGS = ("train.log", "host_launch.log")
_STEP_TOL = 150
_TAIL_BYTES = 512_000
_JSON_MAX = 8_000_000

_NUM = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_LOSS_THEN_STEP = re.compile(
    rf"""['"]loss['"]\s*:\s*({_NUM}).{{0,800}}?['"]step['"]\s*:\s*(\d+)""",
    re.S,
)
_STEP_THEN_LOSS = re.compile(
    rf"""['"]step['"]\s*:\s*(\d+).{{0,800}}?['"]loss['"]\s*:\s*({_NUM})""",
    re.S,
)
_ACTION_KEYS = (
    "flow_matching_loss_action",
    "training/action_loss",
    "action_loss",
    "loss_action",
)
_VIDEO_KEYS = (
    "flow_matching_loss_vision",
    "flow_matching_loss_video",
    "training/video_loss",
    "video_loss",
    "loss_video",
)

_WAM = {
    "cosmos3": "cosmos3",
    "cosmos": "cosmos3",
    "dreamzero": "dreamzero",
    "fastwam": "fastwam",
    "fast-wam": "fastwam",
    "motus": "motus",
    "tau0wm": "tau0wm",
    "tau0": "tau0wm",
    "tau": "tau0wm",
    "lingbot_va": "lingbot_va",
    "gigaworld": "gigaworld",
    "gwp": "gigaworld",
}
_VLA = {
    "pi05": "pi05",
    "pi0.5": "pi05",
    "gr00t": "gr00t",
    "groot": "gr00t",
    "walloss": "walloss",
    "wall-oss": "walloss",
    "lingbot_vla": "lingbot_vla",
    "internvla": "internvla",
    "xvla": "xvla",
    "memoryvla": "memoryvla",
    "acot_vla": "acot",
    "acot": "acot",
    "pi0fast": "pi0_fast",
    "pi0_fast": "pi0_fast",
    "pi0-fast": "pi0_fast",
}

_RED = "\033[31;1m"
_YEL = "\033[33;1m"
_GRN = "\033[32;1m"
_RST = "\033[0m"


@dataclass
class Reading:
    scalar: Optional[float]
    action: Optional[float]
    video: Optional[float]
    step: Optional[int]
    source: str

    @property
    def split(self) -> bool:
        return self.action is not None or self.video is not None


@dataclass
class LossNotice:
    ckpt_label: str
    family: str
    kind: str
    metric_label: str
    value: Optional[float]
    step: Optional[int]
    ckpt_step: Optional[int]
    source: str
    verdict: str
    color: str
    gate: str

    @property
    def plain(self) -> str:
        ckpt = self.ckpt_label or "?"
        if self.verdict == "未找到" or self.value is None:
            return f"LOSS ckpt={ckpt}  未找到这一档的训练 loss  不放行"
        if self.step is None:
            step = "step=?"
        elif self.ckpt_step is not None and self.ckpt_step != self.step:
            step = f"step={self.step}（ckpt {self.ckpt_step}）"
        else:
            step = f"step={self.step}"
        src = f"来源={self.source}" if self.source else ""
        return (
            f"LOSS ckpt={ckpt}  {self.metric_label} {_fmt_num(self.value)}  "
            f"{step}  {self.family}  {self.verdict}  {self.gate}  {src}"
        ).rstrip()

    @property
    def ansi(self) -> str:
        color = {"red": _RED, "yellow": _YEL, "green": _GRN}.get(self.color, "")
        if not color:
            return self.plain
        return f"{color}{self.plain}{_RST}"


def notice_for_ckpt(ckpt_dir: object, policy_type: str) -> LossNotice:
    kind, family = classify_policy(policy_type)
    label = _ckpt_label(ckpt_dir)
    ckpt_step = step_from_path(Path(str(ckpt_dir))) if ckpt_dir else None
    reading = lookup_reading(ckpt_dir)
    metric, value, verdict, color, gate = judge(kind, family, reading)
    return LossNotice(
        ckpt_label=label,
        family=family,
        kind=kind,
        metric_label=metric,
        value=value,
        step=None if reading is None else reading.step,
        ckpt_step=ckpt_step,
        source="" if reading is None else reading.source,
        verdict=verdict,
        color=color,
        gate=gate,
    )


def classify_policy(policy_type: str) -> tuple[str, str]:
    key = (policy_type or "").strip().lower().replace(" ", "")
    if key in _WAM:
        return "wam", _WAM[key]
    if key in _VLA:
        return "vla", _VLA[key]
    return "other", key or "?"


def lookup_reading(ckpt_dir: object) -> Optional[Reading]:
    if not ckpt_dir:
        return None
    ckpt = Path(str(ckpt_dir)).expanduser()
    want = step_from_path(ckpt)
    structured: list[Reading] = []
    logs: list[Path] = []
    for root in _search_roots(ckpt):
        if not root.is_dir():
            continue
        for name in _STRUCTURED:
            path = root / name
            if path.is_file():
                reading = _parse_structured(path, want)
                if reading is not None:
                    structured.append(reading)
        for name in _LOGS:
            path = root / name
            if path.is_file():
                logs.append(path)
    best = _pick(structured, want)
    if best is not None and _step_ok(best, want):
        return best
    from_logs = []
    for path in logs:
        reading = _parse_log_tail(path, want)
        if reading is not None:
            from_logs.append(reading)
    return _pick(structured + from_logs, want)


def judge(
    kind: str,
    family: str,
    reading: Optional[Reading],
) -> tuple[str, Optional[float], str, str, str]:
    """返回 metric 名、数值、结论、颜色、闸门短句。"""
    if reading is None:
        return "", None, "未找到", "yellow", ""
    if kind == "wam":
        return _judge_wam(family, reading)
    if kind == "vla":
        return _judge_vla(family, reading)
    if reading.scalar is None:
        return "", None, "未找到", "yellow", ""
    return "标量", reading.scalar, "无闸门", "plain", "表内没有这一族的门槛"


def step_from_path(path: Path) -> Optional[int]:
    for part in reversed(path.parts):
        match = re.search(r"(?:checkpoint-|step[_-]?)(\d{3,})", part, re.I)
        if match:
            return int(match.group(1))
        if re.fullmatch(r"\d{4,}", part):
            return int(part)
    return None


def parse_client_yaml(path: object) -> tuple[str, str]:
    text = Path(str(path)).read_text(encoding="utf-8")
    ckpt_match = re.search(r"(?m)^[ \t]*ckpt_dir:[ \t]*([^\n#]+)", text)
    ckpt = ckpt_match.group(1).strip().strip("\"'") if ckpt_match else ""
    policy = ""
    if ckpt_match:
        types = list(
            re.finditer(r"(?m)^[ \t]*type:[ \t]*([^\n#]+)", text[: ckpt_match.start()])
        )
        if types:
            policy = types[-1].group(1).strip().strip("\"'")
    if not policy:
        match = re.search(r"(?m)^[ \t]*type:[ \t]*([^\n#]+)", text)
        if match:
            policy = match.group(1).strip().strip("\"'")
    return ckpt, policy


def color_for_message(message: str) -> str:
    if "推荐不拉" in message:
        return "red"
    if "可以拉" in message:
        return "green"
    if any(word in message for word in ("偏高", "未拆", "未找到", "不放行")):
        return "yellow"
    return "plain"


def _judge_vla(
    family: str, reading: Reading
) -> tuple[str, Optional[float], str, str, str]:
    if reading.scalar is None:
        return "", None, "未找到", "yellow", ""
    value = reading.scalar
    if family == "pi05":
        if value >= 0.030:
            return "标量", value, "推荐不拉", "red", "闸门≥0.030"
        if value > 0.010:
            return "标量", value, "偏高", "yellow", "可拉≤0.010"
        return "标量", value, "可以拉", "green", "可拉≤0.010"
    if family == "walloss":
        if value > 0.0064:
            return "标量", value, "推荐不拉", "red", "闸门≤0.0064"
        return "标量", value, "可以拉", "green", "可拉≤0.0064"
    if family == "acot":
        if value >= 0.129:
            return "标量", value, "推荐不拉", "red", "闸门≥0.129"
        return "标量", value, "无闸门", "plain", "低于0.129，表内没有可拉线"
    if family == "gr00t":
        if value >= 0.5:
            return "标量", value, "推荐不拉", "red", "闸门≥0.5（0.9一带）"
        return "标量", value, "无闸门", "plain", "低于0.5，表内没有可拉线"
    if family == "pi0_fast":
        if value <= 0.6:
            return "标量", value, "可以拉", "green", "表内约0.6，不要用0.030"
        return "标量", value, "偏高", "yellow", "高于表内0.6，没有不拉线"
    return "标量", value, "无闸门", "plain", "表内没有这一族的 loss 门槛"


def _judge_wam(
    family: str, reading: Reading
) -> tuple[str, Optional[float], str, str, str]:
    if reading.action is not None:
        value = reading.action
        if family == "cosmos3":
            if value >= 0.02:
                return "action", value, "推荐不拉", "red", "闸门action≥0.02"
            if value <= 0.008:
                return "action", value, "可以拉", "green", "可拉action≤0.008"
            return "action", value, "偏高", "yellow", "可拉action≤0.008"
        if family == "tau0wm":
            if value >= 0.12:
                return "action", value, "推荐不拉", "red", "闸门action≥0.12"
            return "action", value, "无闸门", "plain", "低于0.12，表内没有 action 可拉线"
        if family == "fastwam":
            if value >= 0.073:
                return "action", value, "推荐不拉", "red", "闸门action≥0.073"
            return "action", value, "无闸门", "plain", "低于0.073，表内没有可拉线"
        if family == "dreamzero":
            if value >= 0.02:
                return "action", value, "推荐不拉", "red", "闸门action≥0.02"
            return "action", value, "无闸门", "plain", "低于0.02，表内没有可拉线"
        if family == "motus":
            if value > 0.0879:
                return "action", value, "推荐不拉", "red", "闸门action>0.0879"
            return "action", value, "可以拉", "green", "可拉≤0.0879"
        return "action", value, "无闸门", "plain", "表内没有这一族的 action 门槛"
    if reading.scalar is None:
        return "", None, "未找到", "yellow", ""
    value = reading.scalar
    if reading.split:
        return "总loss", value, "未拆action", "yellow", "有 video/别的项，action 没读到"
    if family == "tau0wm":
        if value >= 0.32:
            return "总loss", value, "推荐不拉", "red", "只有总loss时≥0.32"
        if value <= 0.14:
            return "总loss", value, "可以拉", "green", "只有总loss时≤0.14"
        return "总loss", value, "偏高", "yellow", "只有总loss，0.14–0.32"
    if family == "motus":
        if value > 0.0879:
            return "总loss", value, "推荐不拉", "red", "只有一个数时>0.0879"
        return "总loss", value, "可以拉", "green", "只有一个数且≤0.0879"
    return "总loss", value, "未拆action", "yellow", "总loss不放行，要 action"


def _search_roots(ckpt: Path) -> list[Path]:
    roots = [ckpt]
    cur = ckpt
    for _ in range(4):
        parent = cur.parent
        if parent == cur or parent.name.lower() in _STOP_DIR_NAMES:
            break
        roots.append(parent)
        cur = parent
    return roots


def _ckpt_label(ckpt_dir: object) -> str:
    if not ckpt_dir:
        return "?"
    parts = Path(str(ckpt_dir)).parts
    if not parts:
        return "?"
    if parts[-1] in {"pretrained_model", "pretrained"} and len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1]}"
    return parts[-1]


def _step_ok(reading: Reading, want: Optional[int]) -> bool:
    if want is None or reading.step is None:
        return True
    return abs(reading.step - want) <= _STEP_TOL


def _pick(readings: list[Reading], want: Optional[int]) -> Optional[Reading]:
    ok = [item for item in readings if _step_ok(item, want)]
    if not ok:
        return None

    def rank(item: Reading) -> tuple[int, int, int]:
        dist = 0 if want is None or item.step is None else abs(item.step - want)
        source_rank = {
            "trainer_state.json": 0,
            "loss_probe.jsonl": 1,
            "train_loss_history.json": 2,
            "train.log": 3,
            "host_launch.log": 4,
        }.get(item.source, 9)
        return (dist, source_rank, 0 if item.action is not None else 1)

    return min(ok, key=rank)


def _parse_structured(path: Path, want: Optional[int]) -> Optional[Reading]:
    try:
        if path.name == "trainer_state.json":
            return _parse_trainer_state(path)
        if path.name == "train_loss_history.json":
            return _parse_history(path, want)
        if path.name == "loss_probe.jsonl":
            return _parse_probe(path, want)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    return None


def _parse_trainer_state(path: Path) -> Optional[Reading]:
    size = path.stat().st_size
    if size <= _JSON_MAX:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            reading = _reading_from_rows(
                payload.get("log_history") or [], "trainer_state.json"
            )
            if reading is not None:
                return reading
    return _reading_from_blob(_tail_text(path, _TAIL_BYTES), "trainer_state.json")


def _parse_history(path: Path, want: Optional[int]) -> Optional[Reading]:
    if path.stat().st_size > _JSON_MAX:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "losses" in payload:
        steps = payload.get("steps") or []
        losses = payload.get("losses") or []
        actions = payload.get("action_losses") or payload.get("actions")
        picked = _nearest_series(steps, losses, want)
        if picked is None:
            return None
        step, scalar = picked
        action = None
        if isinstance(actions, list) and steps:
            action_pick = _nearest_series(steps, actions, want)
            if action_pick is not None and action_pick[0] == step:
                action = action_pick[1]
        return Reading(scalar, action, None, step, "train_loss_history.json")
    if isinstance(payload, list):
        return _reading_from_rows(payload, "train_loss_history.json", want)
    return None


def _parse_probe(path: Path, want: Optional[int]) -> Optional[Reading]:
    blob = _tail_text(path, 65_536)
    last = ""
    for raw in blob.splitlines():
        raw = raw.strip()
        if raw.startswith("{"):
            last = raw
    if not last:
        return None
    row = json.loads(last)
    if not isinstance(row, dict):
        return None
    reading = _reading_from_rows([row], "loss_probe.jsonl")
    if reading is None or not _step_ok(reading, want):
        return None
    return reading


def _parse_log_tail(path: Path, want: Optional[int]) -> Optional[Reading]:
    try:
        reading = _reading_from_blob(_tail_text(path, _TAIL_BYTES), path.name)
    except OSError:
        return None
    if reading is None or not _step_ok(reading, want):
        return None
    return reading


def _reading_from_rows(
    rows: object,
    source: str,
    want: Optional[int] = None,
) -> Optional[Reading]:
    if not isinstance(rows, list):
        return None
    loss_row = None
    for row in rows:
        if isinstance(row, dict) and "loss" in row and _as_float(row.get("loss")) is not None:
            loss_row = row
    if loss_row is None:
        return None
    step = _as_int(loss_row.get("step"))
    if want is not None and step is not None and abs(step - want) > _STEP_TOL:
        # 列表里可能有更近的点；上面取的是最后一条。再按步就近。
        nearest = None
        nearest_dist = None
        for row in rows:
            if not isinstance(row, dict):
                continue
            row_step = _as_int(row.get("step"))
            scalar = _as_float(row.get("loss"))
            if row_step is None or scalar is None:
                continue
            dist = abs(row_step - want)
            if nearest_dist is None or dist < nearest_dist:
                nearest = row
                nearest_dist = dist
        if nearest is None or nearest_dist is None or nearest_dist > _STEP_TOL:
            return None
        loss_row = nearest
        step = _as_int(loss_row.get("step"))
    action = _metric_from_row(loss_row, _ACTION_KEYS)
    video = _metric_from_row(loss_row, _VIDEO_KEYS)
    if step is not None and (action is None or video is None):
        for row in reversed(rows):
            if not isinstance(row, dict) or _as_int(row.get("step")) != step:
                continue
            if action is None:
                action = _metric_from_row(row, _ACTION_KEYS)
            if video is None:
                video = _metric_from_row(row, _VIDEO_KEYS)
    return Reading(_as_float(loss_row.get("loss")), action, video, step, source)


def _reading_from_blob(blob: str, source: str) -> Optional[Reading]:
    loss_match = _last_regex(_LOSS_THEN_STEP, blob)
    step_match = _last_regex(_STEP_THEN_LOSS, blob)
    chosen = None
    scalar = None
    step = None
    if loss_match is not None and (
        step_match is None or loss_match.end() >= step_match.end()
    ):
        chosen = loss_match
        scalar = float(loss_match.group(1))
        step = int(loss_match.group(2))
    elif step_match is not None:
        chosen = step_match
        step = int(step_match.group(1))
        scalar = float(step_match.group(2))
    if scalar is None or chosen is None:
        return None
    # action/video 只认和这条 loss 同一段，避免尾巴里更早的键被安到这一步上。
    window = blob[max(0, chosen.start() - 500) : chosen.end() + 500]
    action = _last_keyed_float(window, _ACTION_KEYS)
    video = _last_keyed_float(window, _VIDEO_KEYS)
    return Reading(scalar, action, video, step, source)


def _nearest_series(
    steps: object, values: object, want: Optional[int]
) -> Optional[tuple[int, float]]:
    if not isinstance(steps, list) or not isinstance(values, list):
        return None
    count = min(len(steps), len(values))
    if count == 0:
        return None
    best_i = None
    best_dist = None
    for index in range(count):
        step = _as_int(steps[index])
        value = _as_float(values[index])
        if step is None or value is None:
            continue
        dist = 0 if want is None else abs(step - want)
        if want is None:
            best_i = index
            best_dist = 0
            continue
        if best_dist is None or dist < best_dist:
            best_i = index
            best_dist = dist
    if best_i is None or best_dist is None:
        return None
    if want is not None and best_dist > _STEP_TOL:
        return None
    step = _as_int(steps[best_i])
    value = _as_float(values[best_i])
    if step is None or value is None:
        return None
    return step, value


def _metric_from_row(row: dict, keys: tuple[str, ...]) -> Optional[float]:
    for key in keys:
        if key in row:
            value = _as_float(row.get(key))
            if value is not None:
                return value
    return None


def _last_keyed_float(blob: str, keys: tuple[str, ...]) -> Optional[float]:
    last = None
    for key in keys:
        pattern = re.compile(rf"""['"]{re.escape(key)}['"]\s*:\s*({_NUM})""")
        match = _last_regex(pattern, blob)
        if match is not None and (last is None or match.end() > last.end()):
            last = match
    if last is None:
        return None
    return float(last.group(1))


def _last_regex(pattern: re.Pattern[str], blob: str) -> Optional[re.Match[str]]:
    found = None
    for match in pattern.finditer(blob):
        found = match
    return found


def _tail_text(path: Path, nbytes: int) -> str:
    size = path.stat().st_size
    with path.open("rb") as handle:
        if size > nbytes:
            handle.seek(size - nbytes)
        blob = handle.read().decode("utf-8", "replace")
    # 半截的第一行丢掉，避免从对象中间开始。
    cut = blob.find("\n")
    if cut != -1 and size > nbytes:
        blob = blob[cut + 1 :]
    return blob


def _as_float(value: object) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _as_int(value: object) -> Optional[int]:
    number = _as_float(value)
    if number is None:
        return None
    return int(number)


def _fmt_num(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".") or "0"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="打印这一档 checkpoint 的训练 loss")
    parser.add_argument("--config", default="")
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--type", default="")
    parser.add_argument("--color", action="store_true")
    args = parser.parse_args(argv)
    ckpt = args.ckpt
    policy = args.type
    if args.config:
        ckpt, policy = parse_client_yaml(args.config)
    notice = notice_for_ckpt(ckpt, policy)
    sys.stdout.write((notice.ansi if args.color else notice.plain) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
