#!/usr/bin/env python3
"""RoboCasa365 客户端：连已启动的 π₀.₅ websocket，在无头厨房里 rollout。

不打开真机相机或臂。状态 16 维、动作 12 维，必须和 ``env.action_dim`` 一致，对不上就停。

  source "$SIM_ROOT/env.sh" robocasa
  PI05_EMBODIMENT=robocasa365 PI05_ROBOCASA_CKPT=/path/to/pretrained_model \\
    bash recipes/pi05/launch_pi05_server.sh
  bash recipes/pi05/launch_robocasa_client.sh
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
_INFER_ROOT = Path(os.environ.get("INFER_ROOT", _REPO_ROOT / "infer")).expanduser()
if str(_INFER_ROOT) not in sys.path:
    sys.path.insert(0, str(_INFER_ROOT))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from eval_libero_client import (  # noqa: E402
    RTC_SOTA,
    _connect_ws,
    _infer_chunk,
    _pin_mujoco_gpu,
)
from infer.policies.openpi_pi05_runtime import OpenpiPi05Runtime  # noqa: E402
from infer.policies.pi05_embodiment import get_pi05_embodiment  # noqa: E402
from infer.policies.pi05_sim_obs import (  # noqa: E402
    ROBOCASA365_ACTION_DIM,
    robocasa365_action_to_env,
    robocasa365_images,
    robocasa365_state,
)


def _runtime_from_meta(meta: dict[str, Any]) -> OpenpiPi05Runtime:
    profile = get_pi05_embodiment(
        str(meta.get("embodiment") or "robocasa365"),
        str(meta.get("config_name") or "") or None,
    )
    return OpenpiPi05Runtime(
        ckpt_dir=Path("/tmp/pi05_robocasa_unused"),
        config_name=str(meta.get("config_name") or profile.config_name),
        repo_path=None,
        camera_names=list(profile.cameras),
        action_dim=int(meta.get("action_dim") or profile.action_dim),
        state_dim=int(meta.get("state_dim") or profile.state_dim),
        action_horizon=int(meta.get("action_horizon") or profile.action_horizon),
        embodiment=profile.name,
    )


def run_eval(args: argparse.Namespace) -> int:
    _pin_mujoco_gpu()
    from robocasa.utils.env_utils import create_env

    client, meta = _connect_ws(args.host, args.port, args.timeout)
    print(f"[handshake] {meta}")
    if str(meta.get("model") or "").lower() != "pi05":
        print(f"FAIL: model={meta.get('model')!r}")
        return 1
    runtime = _runtime_from_meta(meta)
    registries = tuple(
        part.strip() for part in str(args.obj_registries).split(",") if part.strip()
    )
    env = create_env(
        env_name=args.env_name,
        robots=args.robot,
        split=args.split,
        seed=int(args.seed),
        render_onscreen=False,
        camera_widths=int(args.resolution),
        camera_heights=int(args.resolution),
        obj_registries=registries,
    )
    successes = 0
    action_dim = None
    save_dir = Path(args.save_dir) if args.save_dir else None
    rtc_cfg = getattr(args, "rtc_cfg", None)
    if rtc_cfg is None and bool(getattr(args, "rtc", False)):
        rtc_cfg = RTC_SOTA["exp"]
    trial_log: list[dict] = []
    try:
        for trial in range(int(args.trials)):
            raw = None
            skipped_empty: list[int] = []
            trial_seed = int(args.seed) + trial * 1009 + 1
            # target split 里有些类别池为空，rng.choice 会 NaN。多换种子；仍失败就记 skipped，不整条停。
            for attempt in range(40):
                if hasattr(env, "unset_ep_meta"):
                    env.unset_ep_meta()
                trial_seed = int(args.seed) + trial * 1009 + 1 + attempt * 100003
                env.rng = np.random.default_rng(trial_seed)
                try:
                    raw = env.reset()
                    break
                except ValueError as exc:
                    if "Probabilities" not in str(exc):
                        raise
                    skipped_empty.append(trial_seed)
                    print(f"[trial {trial}] skip empty object sample attempt={attempt} seed={trial_seed}")
            if raw is None:
                trial_log.append({
                    "trial": trial,
                    "seed": trial_seed,
                    "layout_id": None,
                    "style_id": None,
                    "success": False,
                    "steps": 0,
                    "rtc_used": False,
                    "skipped_empty_sample": skipped_empty,
                })
                print(f"[trial {trial}] skip empty object sample after 40 attempts env={args.env_name}")
                continue
            if isinstance(raw, tuple):
                raw = raw[0]
            if action_dim is None:
                # robosuite 在第一次 reset 里才把 _action_dim 从 None 写成机器人维数。
                action_dim = int(env.action_dim)
                if action_dim != runtime.action_dim or action_dim != ROBOCASA365_ACTION_DIM:
                    print(
                        f"FAIL: 环境动作维 {action_dim}，服务端 {runtime.action_dim}，"
                        f"RoboCasa365 控制器为 {ROBOCASA365_ACTION_DIM}。"
                        "服务端要用 PI05_EMBODIMENT=robocasa365。"
                    )
                    return 2
            client.infer({"reset": True})
            prompt = args.prompt
            if not prompt and hasattr(env, "get_ep_meta"):
                prompt = str((env.get_ep_meta() or {}).get("lang") or "")
            if not prompt:
                prompt = "pick the object and place it in the cabinet"
            ep_meta = env.get_ep_meta() if hasattr(env, "get_ep_meta") else {}
            layout_id = ep_meta.get("layout_id") if isinstance(ep_meta, dict) else None
            style_id = ep_meta.get("style_id") if isinstance(ep_meta, dict) else None
            print(f"[prompt] {prompt} layout={layout_id} style={style_id}")
            success = False
            step = 0
            prev_chunk = None
            executed = 0
            infer_info: dict[str, Any] = {}
            while step < int(args.max_steps):
                # robosuite 的相机会上下颠倒。官方 gym wrapper 只做垂直翻转
                # （img[::-1, :, :]），数据集就是这个朝向。不要再左右镜像。
                images = [
                    np.ascontiguousarray(np.asarray(img)[::-1, :, :])
                    for img in robocasa365_images(raw)
                ]
                state = robocasa365_state(raw)
                if state.shape[0] != runtime.state_dim:
                    raise RuntimeError(
                        f"RoboCasa state 维 {state.shape[0]}，服务端 {runtime.state_dim}"
                    )
                obs = runtime.build_observation(images, state, prompt)
                chunk, infer_info = _infer_chunk(
                    client,
                    runtime,
                    obs,
                    prev_chunk=prev_chunk,
                    executed_steps=executed,
                    rtc=rtc_cfg,
                )
                prev_chunk = chunk
                executed = 0
                n_exec = min(int(args.replan_steps), int(chunk.shape[0]), int(args.max_steps) - step)
                done = False
                info: dict[str, Any] = {}
                for i in range(n_exec):
                    act = np.asarray(chunk[i], dtype=np.float32).reshape(-1)
                    if act.shape[0] != action_dim:
                        raise RuntimeError(f"动作维 {act.shape[0]} != {action_dim}")
                    out = env.step(robocasa365_action_to_env(act))
                    executed += 1
                    raw = out[0]
                    reward = float(out[1]) if len(out) > 1 else 0.0
                    done = bool(out[2]) if len(out) > 2 else False
                    info = out[3] if len(out) > 3 and isinstance(out[3], dict) else {}
                    step += 1
                    # robosuite 的 step 把成功放在稀疏 reward 里，info 是空字典。
                    # 任务实现的是 _check_success，没有公开的 check_success。
                    success = bool(info.get("success")) or reward >= 1.0
                    if hasattr(env, "_check_success"):
                        success = success or bool(env._check_success())
                    if success or done:
                        break
                if success or done:
                    break
            successes += int(success)
            entry = {
                "trial": trial,
                "seed": trial_seed,
                "layout_id": layout_id,
                "style_id": style_id,
                "success": bool(success),
                "steps": step,
                "rtc_used": bool(infer_info.get("rtc_used")) if rtc_cfg else False,
            }
            if skipped_empty:
                entry["skipped_empty_sample"] = skipped_empty
            trial_log.append(entry)
            print(
                f"[trial {trial}] success={success} steps={step} env={args.env_name} "
                f"layout={layout_id} style={style_id}"
            )
    finally:
        env.close()
    n = max(1, int(args.trials))
    summary = {
        "sim": "robocasa365",
        "env_name": args.env_name,
        "robot": args.robot,
        "split": args.split,
        "trials": int(args.trials),
        "rtc": rtc_cfg,
        "success_rate": successes / n,
        "perturbation": "target_kitchen_resample",
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
    p.add_argument("--prompt", default=os.environ.get("PI05_PROMPT") or "")
    p.add_argument("--env-name", default=os.environ.get("ROBOCASA_ENV") or "PickPlaceCounterToCabinet")
    p.add_argument("--robot", default=os.environ.get("ROBOCASA_ROBOT") or "PandaOmron")
    p.add_argument("--split", default=os.environ.get("ROBOCASA_SPLIT") or "target")
    p.add_argument(
        "--obj-registries",
        default=os.environ.get("ROBOCASA_OBJ_REGISTRIES") or "objaverse,lightwheel",
        help="逗号分隔。采集默认是 objaverse 和 lightwheel，不含 aigen",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rtc", action="store_true", help="从第二段 chunk 起带 RTC，默认算法 exp，引导权重 5")
    p.add_argument(
        "--rtc-algo",
        default="",
        choices=["", *RTC_SOTA.keys()],
        help="单档 RTC： " + " ".join(RTC_SOTA),
    )
    p.add_argument("--trials", type=int, default=int(os.environ.get("ROBOCASA_TRIALS") or 1))
    p.add_argument("--max-steps", type=int, default=int(os.environ.get("ROBOCASA_MAX_STEPS") or 400))
    p.add_argument("--replan-steps", type=int, default=int(os.environ.get("ROBOCASA_REPLAN") or 5))
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--save-dir", default=os.environ.get("ROBOCASA_SAVE_DIR") or "")
    args = p.parse_args()
    if not args.save_dir:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        root = Path(os.environ.get("SIM_ROOT") or str(_REPO_ROOT / "outputs"))
        args.save_dir = str(root / "robocasa_eval" / f"{args.env_name}_{stamp}")
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
