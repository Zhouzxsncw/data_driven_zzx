#!/usr/bin/env python3
"""RoboTwin 2.0 客户端：连已启动的 π₀.₅ websocket，按官方 task yml 开一局。

默认任务 ``beat_block_hammer``，配置 ``demo_clean``（aloha-agilex）。
状态和动作都是 14 维绝对 qpos：左 6 + 左爪 + 右 6 + 右爪。
图像按 head、left、right 送进模型的 front、left、right。``take_action`` 走 qpos。

  source "$SIM_ROOT/env.sh" robotwin
  PI05_EMBODIMENT=robotwin PI05_ROBOTWIN_CKPT=/path/to/pretrained_model \\
    bash recipes/pi05/launch_pi05_server.sh
  bash recipes/pi05/launch_robotwin_client.sh
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
_INFER_ROOT = Path(os.environ.get("INFER_ROOT", _REPO_ROOT / "infer")).expanduser()
if str(_INFER_ROOT) not in sys.path:
    sys.path.insert(0, str(_INFER_ROOT))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from eval_libero_client import RTC_SOTA, _connect_ws, _infer_chunk  # noqa: E402
from infer.policies.openpi_pi05_runtime import OpenpiPi05Runtime  # noqa: E402
from infer.policies.pi05_embodiment import get_pi05_embodiment  # noqa: E402
from infer.policies.pi05_sim_obs import robotwin_images, robotwin_state  # noqa: E402


def _robotwin_root() -> Path:
    root = Path(os.environ.get("ROBOTWIN_ROOT") or "").expanduser()
    if not root.is_dir():
        sim = os.environ.get("SIM_ROOT")
        if sim:
            root = Path(sim) / "src" / "RoboTwin"
    if not (root / "envs").is_dir():
        raise FileNotFoundError(f"找不到 RoboTwin 代码：{root}。先跑 bootstrap_sim_infer.sh")
    return root


def _load_task_args(root: Path, task_name: str, task_config: str) -> dict[str, Any]:
    cfg_dir = root / "env_cfg" / "task_config"
    cfg_path = cfg_dir / f"{task_config}.yml"
    if not cfg_path.is_file():
        raise FileNotFoundError(cfg_path)
    args = yaml.load(cfg_path.read_text(encoding="utf-8"), Loader=yaml.FullLoader)
    args["task_name"] = task_name
    args["task_config"] = task_config
    args["eval_mode"] = True
    args["save_data"] = False
    args["render_freq"] = 0
    data_type = dict(args.get("data_type") or {})
    data_type["rgb"] = True
    data_type["qpos"] = True
    args["data_type"] = data_type
    emb_map = yaml.load(
        (cfg_dir / "_embodiment_config.yml").read_text(encoding="utf-8"),
        Loader=yaml.FullLoader,
    )
    embodiment = args.get("embodiment")
    if not isinstance(embodiment, list) or len(embodiment) not in (1, 3):
        raise ValueError(f"embodiment 配置无法用：{embodiment!r}")

    def emb_file(name: str) -> str:
        robot_file = emb_map[name]["file_path"]
        path = Path(robot_file)
        if not path.is_absolute():
            path = (root / robot_file).resolve()
        return str(path)

    if len(embodiment) == 1:
        args["left_robot_file"] = emb_file(embodiment[0])
        args["right_robot_file"] = emb_file(embodiment[0])
        args["dual_arm_embodied"] = True
        embodiment_name = str(embodiment[0])
    else:
        args["left_robot_file"] = emb_file(embodiment[0])
        args["right_robot_file"] = emb_file(embodiment[1])
        args["embodiment_dis"] = embodiment[2]
        args["dual_arm_embodied"] = False
        embodiment_name = f"{embodiment[0]}+{embodiment[1]}"
    for side in ("left", "right"):
        robot_file = args[f"{side}_robot_file"]
        text = Path(robot_file, "config.yml").read_text(encoding="utf-8")
        args[f"{side}_embodiment_config"] = yaml.load(text, Loader=yaml.FullLoader)
    args["embodiment_name"] = embodiment_name
    return args


def _runtime_from_meta(meta: dict[str, Any]) -> OpenpiPi05Runtime:
    profile = get_pi05_embodiment(
        str(meta.get("embodiment") or "robotwin"),
        str(meta.get("config_name") or "") or None,
    )
    return OpenpiPi05Runtime(
        ckpt_dir=Path("/tmp/pi05_robotwin_unused"),
        config_name=str(meta.get("config_name") or profile.config_name),
        repo_path=None,
        camera_names=list(profile.cameras),
        action_dim=int(meta.get("action_dim") or profile.action_dim),
        state_dim=int(meta.get("state_dim") or profile.state_dim),
        action_horizon=int(meta.get("action_horizon") or profile.action_horizon),
        embodiment=profile.name,
    )


def run_eval(args: argparse.Namespace) -> int:
    root = _robotwin_root()
    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    def class_decorator(task_name: str):
        module = importlib.import_module(f"envs.{task_name}")
        try:
            return getattr(module, task_name)()
        except Exception as exc:
            raise SystemExit(f"没有任务 {task_name}: {exc}") from exc

    client, meta = _connect_ws(args.host, args.port, args.timeout)
    print(f"[handshake] {meta}")
    if str(meta.get("model") or "").lower() != "pi05":
        print(f"FAIL: model={meta.get('model')!r}")
        return 1
    runtime = _runtime_from_meta(meta)
    task_args = _load_task_args(root, args.task, args.task_config)
    env = class_decorator(args.task)
    successes = 0
    save_dir = Path(args.save_dir) if args.save_dir else None
    rtc_cfg = getattr(args, "rtc_cfg", None)
    if rtc_cfg is None and bool(getattr(args, "rtc", False)):
        rtc_cfg = RTC_SOTA["exp"]
    from envs.utils.create_actor import UnStableError

    trial_log: list[dict] = []
    seed_cursor = int(args.seed)
    for trial in range(int(args.trials)):
        skipped_unstable: list[int] = []
        while True:
            trial_seed = seed_cursor
            seed_cursor += 1
            try:
                env.setup_demo(now_ep_num=trial, seed=trial_seed, **task_args)
                break
            except UnStableError as exc:
                skipped_unstable.append(trial_seed)
                print(f"[trial {trial}] skip unstable seed={trial_seed} ({exc})")
                try:
                    env.close_env()
                except Exception:
                    pass
                if len(skipped_unstable) >= 40:
                    raise RuntimeError(
                        f"task={args.task} trial={trial} 连续 40 个种子场景不稳"
                    ) from exc
        client.infer({"reset": True})
        obs = env.get_obs()
        state = robotwin_state(obs)
        if state.shape[0] != runtime.state_dim:
            env.close_env()
            print(f"FAIL: qpos 维 {state.shape[0]}，服务端 state {runtime.state_dim}")
            return 2
        step = 0
        success = False
        prev_chunk = None
        executed = 0
        while step < int(args.max_steps):
            images = robotwin_images(obs)
            packed = runtime.build_observation(images, robotwin_state(obs), args.prompt)
            chunk, info = _infer_chunk(
                client,
                runtime,
                packed,
                prev_chunk=prev_chunk,
                executed_steps=executed,
                rtc=rtc_cfg,
            )
            prev_chunk = chunk
            executed = 0
            n_exec = min(int(args.replan_steps), int(chunk.shape[0]), int(args.max_steps) - step)
            for i in range(n_exec):
                act = np.asarray(chunk[i], dtype=np.float32).reshape(-1)
                if act.shape[0] != runtime.action_dim:
                    env.close_env()
                    print(f"FAIL: 动作维 {act.shape[0]}，服务端 {runtime.action_dim}")
                    return 2
                env.take_action(act.tolist(), action_type="qpos")
                step += 1
                executed += 1
                success = bool(getattr(env, "eval_success", False)) or bool(env.check_success())
                if success:
                    break
                obs = env.get_obs()
            if success:
                break
        successes += int(success)
        trial_log.append({
            "trial": trial,
            "seed": trial_seed,
            "skipped_unstable": skipped_unstable,
            "success": bool(success),
            "steps": step,
            "rtc_used": bool(info.get("rtc_used")) if rtc_cfg else False,
        })
        print(
            f"[trial {trial}] success={success} steps={step} task={args.task} "
            f"seed={trial_seed} rtc_used={trial_log[-1]['rtc_used']}"
        )
        env.close_env()
    n = max(1, int(args.trials))
    summary = {
        "sim": "robotwin",
        "task": args.task,
        "task_config": args.task_config,
        "prompt": args.prompt,
        "trials": int(args.trials),
        "rtc": rtc_cfg,
        "success_rate": successes / n,
        "perturbation": "demo_randomized_seed" if args.task_config == "demo_randomized" else args.task_config,
        "trial_log": trial_log,
        "server": {
            "ckpt": meta.get("ckpt"),
            "embodiment": meta.get("embodiment"),
            "action_horizon": meta.get("action_horizon"),
            "action_dim": meta.get("action_dim"),
            "state_dim": meta.get("state_dim"),
        },
    }
    print(f"success_rate {summary['success_rate']:.3f} ({successes}/{args.trials})")
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "eval_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default=os.environ.get("PI05_HOST") or "127.0.0.1")
    p.add_argument("--port", type=int, default=int(os.environ.get("PI05_PORT") or 1242))
    p.add_argument("--timeout", type=float, default=float(os.environ.get("PI05_TIMEOUT") or 600))
    p.add_argument("--prompt", default=os.environ.get("PI05_PROMPT") or "beat the block with the hammer")
    p.add_argument("--task", default=os.environ.get("ROBOTWIN_TASK") or "beat_block_hammer")
    p.add_argument("--task-config", default=os.environ.get("ROBOTWIN_TASK_CONFIG") or "demo_clean")
    p.add_argument("--rtc", action="store_true", help="从第二段 chunk 起带 RTC，默认算法 exp，引导权重 5")
    p.add_argument(
        "--rtc-algo",
        default="",
        choices=["", *RTC_SOTA.keys()],
        help="单档 RTC： " + " ".join(RTC_SOTA),
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--trials", type=int, default=int(os.environ.get("ROBOTWIN_TRIALS") or 1))
    p.add_argument("--max-steps", type=int, default=int(os.environ.get("ROBOTWIN_MAX_STEPS") or 400))
    p.add_argument("--replan-steps", type=int, default=int(os.environ.get("ROBOTWIN_REPLAN") or 16))
    p.add_argument("--save-dir", default=os.environ.get("ROBOTWIN_SAVE_DIR") or "")
    args = p.parse_args()
    if not args.save_dir:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        root = Path(os.environ.get("SIM_ROOT") or str(_REPO_ROOT / "outputs"))
        args.save_dir = str(root / "robotwin_eval" / f"{args.task}_{stamp}")
    if args.rtc_algo:
        args.rtc_cfg = RTC_SOTA[args.rtc_algo]
        args.rtc = args.rtc_cfg is not None
    elif args.rtc:
        args.rtc_cfg = RTC_SOTA["exp"]
    else:
        args.rtc_cfg = None
    return run_eval(args)


if __name__ == "__main__":
    raise SystemExit(main())
