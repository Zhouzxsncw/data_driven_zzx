#!/usr/bin/env python3
"""π₀.₅ LIBERO 客户端：连同一套 websocket 服务端，在 OffScreenRenderEnv 里 rollout。

真机请继续用 ``launch_pi05_client.sh``（embodiment=tongbot）。本脚本 **不会** 打开 RealSense / UDP 臂。

默认读同一份 YAML ``pi05_openpi_tongbot_16d_00092_ws_client.yaml``，强制 embodiment=libero：

  PI05_EMBODIMENT=libero bash recipes/pi05/launch_pi05_server.sh
  bash recipes/pi05/launch_pi05_libero_client.sh
  bash recipes/pi05/launch_pi05_client.sh   # YAML/环境是 libero 时也会转到本脚本

``--fake-obs`` 不依赖 libero/mujoco，只向服务端发 8D+双相机合成观测（握手/维数冒烟）。
"""
from __future__ import annotations

import argparse
import json
import math
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

from infer.policies.openpi_pi05_runtime import OpenpiPi05Runtime  # noqa: E402
from infer.policies.pi05_embodiment import DEFAULT_PI05_YAML, LIBERO, load_pi05_yaml  # noqa: E402

TASK_SUITE_MAX_STEPS = {
    "libero_spatial": 280,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(-1)
    if q.shape[0] != 4:
        raise ValueError(f"quaternion 需要 4 维，得到 {q.shape}")
    w = float(np.clip(q[3], -1.0, 1.0))
    den = math.sqrt(max(0.0, 1.0 - w * w))
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return ((q[:3] * 2.0 * math.acos(w)) / den).astype(np.float32)


def libero_state_from_obs(raw: dict[str, Any]) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(raw["robot0_eef_pos"], dtype=np.float32).reshape(-1),
            _quat2axisangle(raw["robot0_eef_quat"]),
            np.asarray(raw["robot0_gripper_qpos"], dtype=np.float32).reshape(-1),
        ]
    ).astype(np.float32)


def rotate180(image: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(image)[::-1, ::-1])


def postprocess_libero_action(action: np.ndarray, *, binarize_gripper: bool) -> np.ndarray:
    """把模型的 7 维动作原样交给 LIBERO。

    官方 ``examples/libero/main.py`` 是 ``env.step(action.tolist())``，不再改夹爪。
    反归一化后的夹爪已经在 ``[-1, 1]``（约 -1 为张开）。此前把 ``[0,1]`` 映射再取反，
    会把模型的 -1 执行成 +1，抓取和开抽屉整段失败。
    ``binarize_gripper`` 保留参数以兼容旧命令行，不再使用。
    """
    del binarize_gripper
    return np.asarray(action, dtype=np.float32).copy()


def _fill_eval_from_yaml(args: argparse.Namespace) -> None:
    path = getattr(args, "config", None) or os.environ.get("PI05_CONFIG")
    if not path and DEFAULT_PI05_YAML.is_file():
        path = str(DEFAULT_PI05_YAML)
    if not path:
        return
    raw = load_pi05_yaml(path, embodiment="libero")
    pol = raw.get("policy") if isinstance(raw.get("policy"), dict) else {}
    lib = raw.get("libero") if isinstance(raw.get("libero"), dict) else {}
    if not getattr(args, "host", None):
        args.host = str(pol.get("lingbot_host") or "127.0.0.1")
    if getattr(args, "port", None) is None:
        args.port = int(pol.get("lingbot_port") or 1242)
    if not getattr(args, "prompt", None):
        args.prompt = str(pol.get("task_prompt") or "")
    if getattr(args, "timeout", None) is None:
        args.timeout = float(pol.get("lingbot_connect_timeout_sec") or 600.0)
    if not getattr(args, "suite", None):
        args.suite = str(lib.get("suite") or "libero_spatial")
    if getattr(args, "task_id", None) is None:
        args.task_id = int(lib.get("task_id") or 0)
    if getattr(args, "trials", None) is None:
        args.trials = int(lib.get("trials") or 1)
    if getattr(args, "max_steps", None) is None:
        args.max_steps = int(lib.get("max_steps") or 0)
    if getattr(args, "replan_steps", None) is None:
        args.replan_steps = int(lib.get("replan_steps") or pol.get("n_action_steps_override") or 5)
    if getattr(args, "num_steps_wait", None) is None:
        args.num_steps_wait = int(lib.get("num_steps_wait") or 10)
    if getattr(args, "resolution", None) is None:
        args.resolution = int(lib.get("resolution") or 256)


def _connect_ws(host: str, port: int, timeout: float):
    from infer.policies._lingbot_ws_client import WebsocketClientPolicy

    client = WebsocketClientPolicy(host=host, port=port, wait_timeout_sec=timeout)
    meta = dict(client.get_server_metadata() or {})
    return client, meta


def _make_runtime() -> OpenpiPi05Runtime:
    return OpenpiPi05Runtime(
        ckpt_dir=Path("/tmp/pi05_libero_unused"),
        config_name=LIBERO.config_name,
        repo_path=None,
        camera_names=list(LIBERO.cameras),
        action_dim=LIBERO.action_dim,
        state_dim=LIBERO.state_dim,
        action_horizon=LIBERO.action_horizon,
        embodiment="libero",
    )


def _obs_from_libero_raw(runtime: OpenpiPi05Runtime, raw: dict[str, Any], prompt: str) -> dict:
    images = [
        rotate180(raw["agentview_image"]),
        rotate180(raw["robot0_eye_in_hand_image"]),
    ]
    return runtime.build_observation(images, libero_state_from_obs(raw), prompt)


# Physical Intelligence RTC（prefix attention）可测档。评测只比 success_rate / task_completion。
RTC_SOTA: dict[str, dict | None] = {
    "none": None,
    "delay_lock": {"prefix_attention_schedule": "zeros", "max_guidance_weight": 5.0},
    "hard_window": {"prefix_attention_schedule": "linear", "max_guidance_weight": 5.0},
    "ramp": {"prefix_attention_schedule": "ramp", "max_guidance_weight": 5.0},
    "exp": {"prefix_attention_schedule": "exp", "max_guidance_weight": 5.0},
    "exp_strong": {"prefix_attention_schedule": "exp", "max_guidance_weight": 10.0},
    "full_ones": {"prefix_attention_schedule": "ones", "max_guidance_weight": 5.0},
    "delay_lock_w10": {"prefix_attention_schedule": "zeros", "max_guidance_weight": 10.0},
    "hard_window_w10": {"prefix_attention_schedule": "linear", "max_guidance_weight": 10.0},
    "ramp_w10": {"prefix_attention_schedule": "ramp", "max_guidance_weight": 10.0},
}


def _infer_chunk(
    client,
    runtime: OpenpiPi05Runtime,
    obs: dict[str, Any],
    *,
    prev_chunk: np.ndarray | None = None,
    executed_steps: int = 0,
    rtc: dict | None = None,
    debug_attention: bool = False,
) -> tuple[np.ndarray, dict]:
    payload: dict[str, Any] = {"obs": obs}
    if rtc and prev_chunk is not None:
        payload["prev_action_chunk"] = np.asarray(prev_chunk, dtype=np.float32)
        payload["executed_steps"] = int(executed_steps)
        payload["inference_delay"] = int(executed_steps)
        payload["rtc"] = {
            "prefix_attention_schedule": str(rtc.get("prefix_attention_schedule", "exp")),
            "max_guidance_weight": float(rtc.get("max_guidance_weight", 5.0)),
        }
    if debug_attention:
        payload["debug_attention"] = True
    result = client.infer(payload)
    actions = result.get("actions")
    if actions is None:
        actions = result.get("action")
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[-1] < runtime.action_dim:
        raise RuntimeError(f"服务端 action 维 {arr.shape[-1]} < {runtime.action_dim}")
    info = {
        "rtc": result.get("rtc"),
        "rtc_used": result.get("rtc_used"),
        "attention": result.get("attention"),
        "forward": result.get("forward"),
    }
    return arr[:, : runtime.action_dim], info


def run_fake_obs(args: argparse.Namespace) -> int:
    client, meta = _connect_ws(args.host, args.port, args.timeout)
    print(f"[handshake] {meta}")
    if str(meta.get("model") or "").lower() != "pi05":
        print(f"FAIL: model={meta.get('model')!r}")
        return 1
    emb = str(meta.get("embodiment") or "")
    if emb and emb != "libero":
        print(f"FAIL: 服务端 embodiment={emb!r}，LIBERO 客户端不能连真机 serve")
        return 1
    dim = int(meta.get("action_dim") or LIBERO.action_dim)
    state_dim = int(meta.get("state_dim") or LIBERO.state_dim)
    horizon = int(meta.get("action_horizon") or LIBERO.action_horizon)
    runtime = _make_runtime()
    runtime._action_dim = dim
    runtime._state_dim = state_dim
    images = [np.zeros((256, 256, 3), dtype=np.uint8) for _ in range(2)]
    state = np.zeros(state_dim, dtype=np.float32)
    obs = runtime.build_observation(images, state, args.prompt)
    t0 = time.perf_counter()
    actions, _info = _infer_chunk(client, runtime, obs)
    dt = (time.perf_counter() - t0) * 1000
    print(f"[fake-obs] actions={actions.shape} server_ms={dt:.0f}")
    if actions.shape[0] != horizon or actions.shape[1] != dim:
        print(f"FAIL: 期望 ({horizon}, {dim})")
        return 1
    print("OK")
    return 0


def _pin_mujoco_gpu() -> None:
    """LIBERO 离屏画面用 NVIDIA EGL，不用 Mesa/OSMesa 的 CPU 光栅。

    必须在 ``import mujoco`` / ``import libero`` 之前调用。
    ``MUJOCO_GL=osmesa`` 时保留 CPU 路径，方便对照。
    """
    gl = os.environ.get("MUJOCO_GL", "").strip().lower()
    if gl == "osmesa":
        return
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
    icd = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
    if os.path.isfile(icd):
        os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", icd)


def _mujoco_render_info(env: Any) -> dict[str, Any]:
    """记下实际用的 GL 后端，写进 eval_summary。"""
    info: dict[str, Any] = {
        "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
        "PYOPENGL_PLATFORM": os.environ.get("PYOPENGL_PLATFORM"),
        "MUJOCO_EGL_DEVICE_ID": os.environ.get("MUJOCO_EGL_DEVICE_ID"),
        "nvidia_icd": os.environ.get("__EGL_VENDOR_LIBRARY_FILENAMES"),
    }
    try:
        import mujoco

        info["mujoco"] = str(getattr(mujoco, "__version__", ""))
    except Exception as exc:
        info["mujoco_error"] = f"{type(exc).__name__}: {exc}"
    sim = getattr(env, "sim", None)
    if sim is None:
        sim = getattr(getattr(env, "env", None), "sim", None)
    ctx = getattr(sim, "_render_context_offscreen", None) if sim is not None else None
    if ctx is not None:
        info["robosuite_context"] = type(ctx).__name__
        glc = getattr(ctx, "gl_ctx", None)
        if glc is not None:
            info["robosuite_gl"] = type(glc).__name__
        info["render_device_id"] = getattr(ctx, "device_id", None)
    gl_name = str(info.get("robosuite_gl") or "")
    info["gpu"] = gl_name in {"EGLGLContext", "GLContext"} and info.get("MUJOCO_GL") == "egl"
    return info


def _make_libero_env(suite_name: str, task_id: int, resolution: int):
    _pin_mujoco_gpu()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bench = benchmark.get_benchmark_dict()
    if suite_name not in bench:
        raise SystemExit(
            f"未知 LIBERO suite {suite_name!r}；可选: {', '.join(sorted(bench))}"
        )
    suite = bench[suite_name]()
    task = suite.get_task(int(task_id))
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=int(resolution),
        camera_widths=int(resolution),
    )
    return env, suite, task


def _f32_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    return [round(float(x), 6) for x in arr]


def _chunk_list(chunk: np.ndarray) -> list[list[float]]:
    arr = np.asarray(chunk, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return [[round(float(x), 6) for x in row] for row in arr]


def _write_mp4(path: Path, frames: list[np.ndarray], fps: int) -> None:
    if not frames:
        return
    try:
        import imageio.v2 as imageio

        imageio.mimsave(str(path), frames, fps=fps)
    except Exception as exc:
        print(f"[warn] 写视频失败 {path.name}: {exc}")


def _save_trial_outputs(
    save_dir: Path,
    *,
    trial: int,
    prompt: str,
    success: bool,
    step: int,
    frames: list[np.ndarray],
    wrist_frames: list[np.ndarray],
    actions: list[np.ndarray],
    states: list[np.ndarray],
    joints: list[np.ndarray],
    fps: int = 10,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    prefix = save_dir / f"trial_{trial:02d}"
    act = np.stack(actions, axis=0) if actions else np.zeros((0, 7), dtype=np.float32)
    state_arr = np.stack(states, axis=0) if states else np.zeros((0, 8), dtype=np.float32)
    joint_arr = np.stack(joints, axis=0) if joints else np.zeros((0, 7), dtype=np.float32)
    np.savez_compressed(
        str(prefix) + "_actions.npz",
        actions=act,
        states=state_arr,
        joint_pos=joint_arr,
    )
    summary = {
        "trial": trial,
        "prompt": prompt,
        "success": bool(success),
        "steps": int(step),
        "action_shape": list(act.shape),
        "state_shape": list(state_arr.shape),
        "joint_shape": list(joint_arr.shape),
        "action_abs_mean": float(np.abs(act).mean()) if act.size else 0.0,
        "n_frames": len(frames),
        "n_wrist_frames": len(wrist_frames),
        "flow": f"trial_{trial:02d}_flow.jsonl",
        "video": f"trial_{trial:02d}.mp4",
        "wrist_video": f"trial_{trial:02d}_wrist.mp4" if wrist_frames else None,
    }
    (Path(str(prefix) + "_summary.json")).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_mp4(Path(str(prefix) + ".mp4"), frames, fps)
    _write_mp4(Path(str(prefix) + "_wrist.mp4"), wrist_frames, fps)
    print(
        f"[saved] {prefix}_summary.json actions={act.shape} "
        f"joints={joint_arr.shape} frames={len(frames)} flow={prefix}_flow.jsonl"
    )


def run_eval(args: argparse.Namespace) -> int:
    client, meta = _connect_ws(args.host, args.port, args.timeout)
    print(f"[handshake] {meta}")
    emb = str(meta.get("embodiment") or "")
    if emb and emb != "libero":
        raise SystemExit(f"服务端 embodiment={emb!r}，拒绝在 LIBERO 客户端使用")
    dim = int(meta.get("action_dim") or LIBERO.action_dim)
    state_dim = int(meta.get("state_dim") or LIBERO.state_dim)
    runtime = _make_runtime()
    runtime._action_dim = dim
    runtime._state_dim = state_dim

    env, suite, task = _make_libero_env(args.suite, args.task_id, args.resolution)
    render_info = _mujoco_render_info(env)
    print(
        f"[render] gpu={render_info.get('gpu')} gl={render_info.get('robosuite_gl')} "
        f"MUJOCO_GL={render_info.get('MUJOCO_GL')} device={render_info.get('MUJOCO_EGL_DEVICE_ID')}"
    )
    # 套件任务自己的语言指令。YAML 里的真机 task_prompt 不能盖住它。
    prompt = getattr(task, "language", "") or args.prompt or "do the task"
    max_steps = int(args.max_steps or TASK_SUITE_MAX_STEPS.get(args.suite, 300))
    replan = max(1, int(args.replan_steps))
    successes = 0
    completions: list[float] = []
    attn_rows: list[dict] = []
    dummy = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
    save_dir = Path(args.save_dir).expanduser() if args.save_dir else None
    rtc_cfg = getattr(args, "rtc_cfg", None)
    if rtc_cfg is None and bool(getattr(args, "rtc", False)):
        rtc_cfg = RTC_SOTA["exp"]
    use_rtc = rtc_cfg is not None
    trial_log: list[dict] = []

    try:
        for trial in range(int(args.trials)):
            raw = env.reset()
            init_states = None
            try:
                init_states = suite.get_task_init_states(int(args.task_id))
            except Exception:
                init_states = None
            init_index = None
            if init_states is not None and len(init_states) > 0:
                init_index = trial % len(init_states)
                raw = env.set_init_state(init_states[init_index])
            for _ in range(int(args.num_steps_wait)):
                raw, _, _, _ = env.step(dummy.tolist())
            client.infer({"reset": True})
            success = False
            step = 0
            frames: list[np.ndarray] = []
            wrist_frames: list[np.ndarray] = []
            executed: list[np.ndarray] = []
            states: list[np.ndarray] = []
            joints: list[np.ndarray] = []
            prev_chunk = None
            executed_since_infer = 0
            want_attn = bool(getattr(args, "attention", False))
            flow_fp = None
            if save_dir is not None:
                save_dir.mkdir(parents=True, exist_ok=True)
                flow_fp = (save_dir / f"trial_{trial:02d}_flow.jsonl").open("w", encoding="utf-8")
            try:
              while step < max_steps:
                state_in = libero_state_from_obs(raw)
                joint_in = np.asarray(raw.get("robot0_joint_pos", []), dtype=np.float32).reshape(-1)
                obs = _obs_from_libero_raw(runtime, raw, prompt)
                model_out, info = _infer_chunk(
                    client,
                    runtime,
                    obs,
                    prev_chunk=prev_chunk,
                    executed_steps=executed_since_infer,
                    rtc=rtc_cfg,
                    debug_attention=want_attn,
                )
                want_attn = False
                if info.get("attention"):
                    attn_rows.append(info["attention"])
                chunk = postprocess_libero_action(model_out, binarize_gripper=not args.no_binarize)
                if flow_fp is not None:
                    flow_fp.write(json.dumps({
                        "trial": trial,
                        "env_step": step,
                        "kind": "infer",
                        "model_input": {
                            "prompt": prompt,
                            "state": _f32_list(state_in),
                            "joint_pos": _f32_list(joint_in),
                            "images": ["agentview", "eye_in_hand"],
                        },
                        "model_output": {
                            "actions": _chunk_list(model_out),
                            "rtc_used": info.get("rtc_used"),
                            "rtc": info.get("rtc"),
                            "forward": info.get("forward"),
                        },
                    }, ensure_ascii=False) + "\n")
                    flow_fp.flush()
                prev_chunk = chunk
                executed_since_infer = 0
                n_exec = min(replan, int(chunk.shape[0]), max_steps - step)
                done = False
                for i in range(n_exec):
                    act = np.asarray(chunk[i], dtype=np.float32).reshape(-1)
                    if act.shape[0] != 7:
                        act = np.resize(act, 7)
                    raw, _, done, info = env.step(act.tolist())
                    step += 1
                    executed.append(act.copy())
                    states.append(libero_state_from_obs(raw))
                    joints.append(np.asarray(raw.get("robot0_joint_pos", np.zeros(7)), dtype=np.float32).reshape(-1))
                    executed_since_infer += 1
                    if save_dir is not None and "agentview_image" in raw:
                        frames.append(np.ascontiguousarray(raw["agentview_image"]))
                    if save_dir is not None and "robot0_eye_in_hand_image" in raw:
                        wrist_frames.append(np.ascontiguousarray(raw["robot0_eye_in_hand_image"]))
                    if flow_fp is not None:
                        flow_fp.write(json.dumps({
                            "trial": trial,
                            "env_step": step,
                            "kind": "execute",
                            "action": _f32_list(act),
                            "joint_pos": _f32_list(raw.get("robot0_joint_pos")),
                            "state": _f32_list(libero_state_from_obs(raw)),
                        }, ensure_ascii=False) + "\n")
                        flow_fp.flush()
                    success = bool(env.check_success()) if hasattr(env, "check_success") else bool(
                        (info or {}).get("success") or done
                    )
                    if success or done:
                        break
                if success or done:
                    break
            finally:
                if flow_fp is not None:
                    flow_fp.close()
            successes += int(success)
            trial_log.append({
                "trial": trial,
                "init_state_index": init_index,
                "success": bool(success),
                "steps": step,
            })
            # 成功记 1；失败记走到的步数占上限的比例，当作任务完成度
            completions.append(1.0 if success else (float(step) / float(max_steps) if max_steps else 0.0))
            print(
                f"[trial {trial}] success={success} completion={completions[-1]:.3f} "
                f"steps={step} rtc={use_rtc} prompt={prompt!r}"
            )
            if save_dir is not None:
                _save_trial_outputs(
                    save_dir,
                    trial=trial,
                    prompt=prompt,
                    success=success,
                    step=step,
                    frames=frames,
                    wrist_frames=wrist_frames,
                    actions=executed,
                    states=states,
                    joints=joints,
                )
    finally:
        env.close()

    n = max(1, int(args.trials))
    summary = {
        "suite": args.suite,
        "task_id": int(args.task_id),
        "trials": int(args.trials),
        "rtc": rtc_cfg,
        "success_rate": successes / n,
        "task_completion": float(np.mean(completions)) if completions else 0.0,
        "perturbation": "libero_init_states",
        "trial_log": trial_log,
        "server": {
            "ckpt": meta.get("ckpt"),
            "embodiment": meta.get("embodiment"),
            "action_horizon": meta.get("action_horizon"),
            "action_dim": meta.get("action_dim"),
            "state_dim": meta.get("state_dim"),
        },
        "render": render_info,
        "logs": {
            "flow": "trial_*_flow.jsonl",
            "actions": "trial_*_actions.npz",
            "summary": "eval_summary.json",
        },
        "attention": attn_rows[:1],
    }
    print(
        f"success_rate {summary['success_rate']:.3f}  "
        f"task_completion {summary['task_completion']:.3f}  "
        f"({successes}/{args.trials})"
    )
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "eval_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        default=os.environ.get("PI05_CONFIG") or "",
        help="统一 π₀.₅ YAML；本客户端强制按 libero overlay 读 host/port/prompt/suite",
    )
    p.add_argument("--host", default=os.environ.get("PI05_HOST") or "")
    p.add_argument("--port", type=int, default=int(os.environ["PI05_PORT"]) if os.environ.get("PI05_PORT") else None)
    p.add_argument("--timeout", type=float, default=None)
    p.add_argument("--prompt", default=os.environ.get("PI05_PROMPT") or "")
    p.add_argument("--suite", default=os.environ.get("LIBERO_SUITE") or "")
    p.add_argument("--task-id", type=int, default=int(os.environ["LIBERO_TASK_ID"]) if os.environ.get("LIBERO_TASK_ID") else None)
    p.add_argument("--trials", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--replan-steps", type=int, default=None)
    p.add_argument("--num-steps-wait", type=int, default=None)
    p.add_argument("--resolution", type=int, default=None)
    p.add_argument(
        "--no-binarize",
        action="store_true",
        help="保留兼容。夹爪已按官方评测原样执行，此开关不再改动作",
    )
    p.add_argument("--fake-obs", action="store_true", help="不启动 libero，只打合成观测")
    p.add_argument("--rtc", action="store_true", help="从第二段 chunk 起带 RTC，默认算法 exp")
    p.add_argument(
        "--rtc-algo",
        default="",
        choices=["", *RTC_SOTA.keys()],
        help="单档 RTC： " + " ".join(RTC_SOTA),
    )
    p.add_argument(
        "--rtc-sweep",
        action="store_true",
        help="按 RTC_SOTA 每一档各跑一遍，汇总 success_rate 和 task_completion",
    )
    p.add_argument("--attention", action="store_true", help="每个 trial 第一次推理要模态注意力摘要")
    p.add_argument(
        "--save-dir",
        default=os.environ.get("LIBERO_SAVE_DIR") or "",
        help="写出 mp4 + actions.npz + summary.json",
    )
    args = p.parse_args()
    _fill_eval_from_yaml(args)
    args.host = args.host or "127.0.0.1"
    if args.port is None:
        args.port = 1242
    if args.timeout is None:
        args.timeout = 600.0
    args.suite = args.suite or "libero_spatial"
    if args.task_id is None:
        args.task_id = 0
    if args.trials is None:
        args.trials = 1
    if args.max_steps is None:
        args.max_steps = 0
    if args.replan_steps is None:
        args.replan_steps = 5
    if args.num_steps_wait is None:
        args.num_steps_wait = 10
    if args.resolution is None:
        args.resolution = 256
    if args.max_steps == 0:
        args.max_steps = None
    args.save_dir = args.save_dir or None
    # 仿真和真机一样，每次推理都落日志。真机由 engine 写 infer_probe / machine_flow；
    # 这里没给 --save-dir 时也写出 trial_*_flow.jsonl（模型输出 + 关节/状态）。
    if not args.fake_obs and not args.save_dir:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args.save_dir = str(
            _REPO_ROOT / "outputs" / "libero_eval" / f"{args.suite}_t{int(args.task_id)}_{stamp}"
        )
    if args.fake_obs:
        return run_fake_obs(args)
    if args.rtc_sweep:
        rows = []
        base_save = args.save_dir
        for name, cfg in RTC_SOTA.items():
            args.rtc_cfg = cfg
            args.rtc = cfg is not None
            if base_save:
                args.save_dir = str(Path(base_save) / name)
            print(f"[rtc-sweep] {name} {cfg}")
            run_eval(args)
            summary_path = Path(args.save_dir) / "eval_summary.json" if args.save_dir else None
            row = {"algo": name, "rtc": cfg}
            if summary_path and summary_path.is_file():
                row.update(json.loads(summary_path.read_text(encoding="utf-8")))
            rows.append(row)
        table = [
            {
                "algo": r["algo"],
                "success_rate": r.get("success_rate"),
                "task_completion": r.get("task_completion"),
            }
            for r in rows
        ]
        print(json.dumps(table, ensure_ascii=False, indent=2))
        if base_save:
            out = Path(base_save) / "rtc_sweep.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0
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
