#!/usr/bin/env python3
"""按 sim_eval_protocol.yaml 跑前 10 个任务、每个 10 次，并写下成功率记录。

不启动服务。LIBERO / RoboTwin / RoboCasa 的服务要已经在听。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import yaml

_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parent.parent
_PROTOCOL = _THIS / "sim_eval_protocol.yaml"


def _load() -> dict:
    return yaml.safe_load(_PROTOCOL.read_text(encoding="utf-8"))


def _py(sim: str) -> str:
    env = {
        "libero": "PI05_LIBERO_PYTHON",
        "robotwin": "ROBOTWIN_PYTHON",
        "robocasa": "ROBOCASA_PYTHON",
    }[sim]
    override = os.environ.get(env)
    if override:
        return override
    defaults = {
        "libero": "/home/xk/CQTongBot/.venv310/bin/python",
        "robotwin": "/home/xk/miniconda3/envs/RoboTwin/bin/python",
        "robocasa": "/home/xk/miniconda3/envs/robocasa/bin/python",
    }
    path = defaults[sim]
    if Path(path).is_file():
        return path
    return "python3"


def _task_prompt(task: str) -> str:
    roots = [
        os.environ.get("ROBOTWIN_ROOT") or "",
        "/home/xk/RoboTwin",
        "/home/cigai22/RoboTwin",
        "/home/tk/RoboTwin",
    ]
    path = None
    for root in roots:
        if not root:
            continue
        candidate = Path(root) / "description" / "task_instruction" / f"{task}.json"
        if candidate.is_file():
            path = candidate
            break
    if path is None:
        return task.replace("_", " ")
    data = json.loads(path.read_text(encoding="utf-8"))
    text = str(data.get("full_description") or task)
    return text.replace("<", "").replace(">", "")


def _run(cmd: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def _read_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# 和 eval_libero_client.RTC_SOTA 里这几档一致。协议脚本不 import 客户端，避免带上仿真依赖。
RTC_PRESETS = {
    "exp": ("exp", 5.0),
    "exp_strong": ("exp", 10.0),
    "hard_window": ("linear", 5.0),
    "delay_lock": ("zeros", 5.0),
    "full_ones": ("ones", 5.0),
}


def run_sim(
    sim: str,
    protocol: dict,
    out_root: Path,
    host: str,
    port: int,
    rtc_algo: str,
) -> dict:
    trials = int(protocol["trials_per_task"])
    rtc = bool(rtc_algo) or bool(protocol.get("rtc"))
    sim_dir = out_root / sim
    sim_dir.mkdir(parents=True, exist_ok=True)
    base_env = os.environ.copy()
    base_env["PYTHONPATH"] = str(_REPO / "infer") + (
        os.pathsep + base_env["PYTHONPATH"] if base_env.get("PYTHONPATH") else ""
    )
    if sim == "robotwin":
        base_env["ROBOTWIN_ROOT"] = os.environ.get("ROBOTWIN_ROOT") or "/home/xk/RoboTwin"
    rows = []
    if sim == "libero":
        spec = protocol["libero"]
        for task_id in spec["task_ids"]:
            save = sim_dir / f"task_{int(task_id):02d}"
            cmd = [
                _py(sim),
                "-u",
                str(_THIS / "eval_libero_client.py"),
                "--config",
                str(_REPO / "infer/infer/config/lerobot/pi05_libero_ws_client.yaml"),
                "--host",
                host,
                "--port",
                str(port),
                "--suite",
                str(spec["suite"]),
                "--task-id",
                str(int(task_id)),
                "--trials",
                str(trials),
                "--save-dir",
                str(save),
            ]
            if rtc_algo:
                cmd.extend(["--rtc-algo", rtc_algo])
            elif rtc:
                cmd.append("--rtc")
            _run(cmd, base_env)
            rows.append(_read_summary(save / "eval_summary.json"))
    elif sim == "robotwin":
        spec = protocol["robotwin"]
        for task in spec["tasks"]:
            save = sim_dir / task
            cmd = [
                _py(sim),
                "-u",
                str(_THIS / "eval_robotwin_client.py"),
                "--host",
                host,
                "--port",
                str(port),
                "--task",
                task,
                "--task-config",
                str(spec["task_config"]),
                "--prompt",
                _task_prompt(task),
                "--trials",
                str(trials),
                "--save-dir",
                str(save),
            ]
            if rtc_algo:
                cmd.extend(["--rtc-algo", rtc_algo])
            elif rtc:
                cmd.append("--rtc")
            _run(cmd, base_env)
            rows.append(_read_summary(save / "eval_summary.json"))
    elif sim == "robocasa":
        spec = protocol["robocasa"]
        for task in spec["tasks"]:
            save = sim_dir / task
            cmd = [
                _py(sim),
                "-u",
                str(_THIS / "eval_robocasa_client.py"),
                "--host",
                host,
                "--port",
                str(port),
                "--env-name",
                task,
                "--robot",
                str(spec["robot"]),
                "--split",
                str(spec["split"]),
                "--trials",
                str(trials),
                "--save-dir",
                str(save),
            ]
            if rtc_algo:
                cmd.extend(["--rtc-algo", rtc_algo])
            elif rtc:
                cmd.append("--rtc")
            _run(cmd, base_env)
            rows.append(_read_summary(save / "eval_summary.json"))
    else:
        raise SystemExit(f"未知仿真 {sim}")
    successes = sum(
        sum(1 for item in row.get("trial_log") or [] if item.get("success"))
        for row in rows
    )
    total = sum(len(row.get("trial_log") or []) or int(row["trials"]) for row in rows)
    return {
        "sim": sim,
        "trials_per_task": trials,
        "n_tasks": len(rows),
        "successes": successes,
        "trials": total,
        "success_rate": (successes / total) if total else 0.0,
        "tasks": rows,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sim", required=True, choices=["libero", "robotwin", "robocasa", "all"])
    p.add_argument("--host", default=os.environ.get("PI05_HOST") or "127.0.0.1")
    p.add_argument("--port", type=int, default=int(os.environ.get("PI05_PORT") or 0))
    p.add_argument("--out", default="")
    p.add_argument(
        "--rtc-algo",
        default="",
        choices=["", *RTC_PRESETS.keys()],
        help="覆盖协议 yaml 的 rtc_algo。可选 " + " ".join(RTC_PRESETS),
    )
    args = p.parse_args()
    protocol = _load()
    rtc_algo = args.rtc_algo or (str(protocol.get("rtc_algo") or "") if protocol.get("rtc") else "")
    if rtc_algo and rtc_algo not in RTC_PRESETS:
        raise SystemExit(f"未知 rtc_algo {rtc_algo}")
    schedule, weight = RTC_PRESETS[rtc_algo] if rtc_algo else (None, None)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out) if args.out else _REPO / "outputs" / "sim_eval" / stamp
    out_root.mkdir(parents=True, exist_ok=True)
    sims = ["libero", "robotwin", "robocasa"] if args.sim == "all" else [args.sim]
    default_port = {"libero": 1242, "robotwin": 1242, "robocasa": 1242}
    blocks = []
    for sim in sims:
        port = args.port or default_port[sim]
        blocks.append(run_sim(sim, protocol, out_root, args.host, port, rtc_algo))
    record = {
        "created": stamp,
        "protocol": str(_PROTOCOL),
        "inference": {
            "framework": "pi05 websocket",
            "rtc": bool(rtc_algo),
            "rtc_algo": rtc_algo or None,
            "rtc_schedule": schedule,
            "rtc_max_guidance_weight": weight,
        },
        "sims": blocks,
    }
    path = out_root / "record.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[record] {path}")
    for block in blocks:
        print(
            f"{block['sim']}: {block['successes']}/{block['trials']} "
            f"success_rate={block['success_rate']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
