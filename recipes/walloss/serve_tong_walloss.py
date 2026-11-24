#!/usr/bin/env python3
"""WALL-OSS TongBot websocket server, VA-client compatible.

Protocol (same as LingBot-VA / MemoryVLA):
  handshake metadata  {model: walloss, action_dim: 20, solve_ik: true, ...}
  reset:              {reset: True, prompt: [str]}
  compute_kv_cache:   ignored
  infer:              {obs: [cam_dict, ...], state: (16,), n_exec?: int}
  reply:              {action: (n_exec, 20)}  # 绝对 EE（模型输出）
  solve_ik:           {solve_ik: True, abs_ee: (n,20), state: (16,), n_exec?, ik?}
  reply:              {action: (n_planned, 16)}  # RealMan 连续 IK

Default ckpt: jiangsuan-16 00086 EE SFT
(relative EE 20D → absolute EE; client 读完再 ws solve_ik → 16D).
"""
from __future__ import annotations

import argparse
import asyncio
import functools
import http
import logging
import os
import sys
import time
import traceback
from pathlib import Path

import msgpack
import numpy as np
import websockets
import websockets.asyncio.server as _server
import websockets.frames

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE if (HERE / "ee_infer_pipeline.py").is_file() else Path(
    "/home/xk/VLA/wall_oss_ppu_setup/scripts"
)
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(Path(os.environ.get("WALLX_CODE", "/data/wall-oss/src/wall-x"))))

from walloss_policy import (  # noqa: E402
    DEFAULT_CKPT,
    DEFAULT_PROMPT,
    WallOssTongPolicy,
    to_hwc_uint8,
)
from ik_config import parse_ik_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("walloss_serve")

CAM_FRONT = "observation.images.front"
CAM_LEFT = "observation.images.left"
CAM_RIGHT = "observation.images.right"
TONGBOT_DIM = 16
# 服务端分段耗时（predict_logical16 的 info），随 action 一起回传给客户端
STAGE_KEYS = ("fk_ms", "flow_ms", "convert_ms", "ik_ms")


def _request_n_exec(msg: dict, default: int, horizon: int) -> int:
    """客户端 action-step → 绝对 EE 路点数；缺字段时用启动默认值。"""
    raw = msg.get("n_exec", msg.get("n_action_steps"))
    try:
        n = int(default if raw is None else raw)
    except (TypeError, ValueError):
        n = int(default)
    return max(1, min(n, max(1, int(horizon))))


def pack_array(obj):
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(
            buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"]
        )
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


Packer = functools.partial(msgpack.Packer, default=pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)


def _state16(state) -> np.ndarray:
    arr = np.asarray(state, dtype=np.float64).reshape(-1)
    if arr.size < TONGBOT_DIM:
        pad = np.zeros(TONGBOT_DIM, dtype=np.float64)
        pad[: arr.size] = arr
        return pad
    return arr[:TONGBOT_DIM]


class WallOssTongServePolicy:
    def __init__(self, backend: WallOssTongPolicy, instruction: str):
        self.backend = backend
        self.instruction = instruction

    def reset(self, prompt: str | None, ik: dict | None = None) -> None:
        if prompt:
            self.instruction = str(prompt).strip() or self.instruction
            self.backend.instruction = self.instruction
        LOGGER.info("reset prompt=%s", self.instruction[:160])

    def solve_ik(self, msg: dict) -> dict:
        from ee_infer_pipeline import plan_to_logical16

        if isinstance(msg.get("ik"), dict):
            self.backend.apply_ik_config(msg["ik"])
        joints = _state16(msg.get("state"))
        abs_ee = np.asarray(msg.get("abs_ee"), dtype=np.float64)
        if abs_ee.ndim == 1:
            abs_ee = abs_ee[np.newaxis, :]
        if abs_ee.shape[-1] != 20:
            raise ValueError(f"solve_ik 需要 abs_ee EE20，得到 {abs_ee.shape}")
        horizon = int(getattr(self.backend.infer_cfg, "action_horizon", 32) or 32)
        n_exec = _request_n_exec(msg, self.backend.n_exec, horizon)
        t0 = time.perf_counter()
        plans = self.backend.ik.ik_abs_chunk(joints, abs_ee, n_exec=n_exec)
        ik_ms = (time.perf_counter() - t0) * 1e3
        if not plans:
            actions = np.asarray(joints, dtype=np.float32).reshape(1, TONGBOT_DIM)
            info = {
                "aborted_ik": True,
                "n_planned": 0,
                "cartesian_limited": False,
                "held_ik": True,
                "max_step_deg": 0.0,
                "rejected_step_deg": 0.0,
                "cartesian_t": 0.0,
            }
        else:
            actions = np.stack(
                [plan_to_logical16(p).astype(np.float32) for p in plans], axis=0
            )
            info = {
                "aborted_ik": len(plans) < n_exec,
                "n_planned": len(plans),
                "cartesian_limited": any(bool(p.cartesian_limited) for p in plans),
                "held_ik": any(bool(p.held) for p in plans),
                "max_step_deg": max(
                    (max(p.left_step_max_deg, p.right_step_max_deg) for p in plans),
                    default=0.0,
                ),
                "rejected_step_deg": max(
                    (float(p.rejected_step_deg) for p in plans), default=0.0
                ),
                "cartesian_t": min((float(p.cartesian_t) for p in plans), default=1.0),
            }
        LOGGER.info(
            "solve_ik action=%s n_exec=%s n_planned=%s aborted=%s cart_lim=%s "
            "mode=%s ik=%.1f ms",
            tuple(actions.shape),
            n_exec,
            info["n_planned"],
            info["aborted_ik"],
            info["cartesian_limited"],
            self.backend.ik.ik_settings.mode,
            ik_ms,
        )
        return {
            "action": np.ascontiguousarray(actions, dtype=np.float32),
            "server_timing": {"ik_ms": ik_ms, **info, "ik_mode": self.backend.ik.ik_settings.mode},
        }

    def infer(self, msg: dict) -> dict:
        if msg.get("reset"):
            prompt = msg.get("prompt")
            if isinstance(prompt, (list, tuple)) and prompt:
                prompt = prompt[0]
            self.reset(prompt if isinstance(prompt, str) else None)
            return {}
        if msg.get("compute_kv_cache"):
            return {}
        if msg.get("solve_ik"):
            return self.solve_ik(msg)

        obs_list = msg.get("obs")
        if not obs_list:
            raise KeyError("infer 消息缺少 obs")
        last = obs_list[-1]
        images = {}
        for key in (CAM_FRONT, CAM_LEFT, CAM_RIGHT):
            if key not in last:
                raise KeyError(f"obs 缺相机 {key}；keys={list(last)}")
            images[key] = to_hwc_uint8(last[key])
        joints = _state16(msg.get("state"))
        horizon = int(getattr(self.backend.infer_cfg, "action_horizon", 32) or 32)
        n_exec = _request_n_exec(msg, self.backend.n_exec, horizon)
        t0 = time.perf_counter()
        actions, info = self.backend.predict_abs_ee20(
            images, joints, self.instruction, n_exec=n_exec
        )
        stage_ms = {
            k: float(info[k]) for k in STAGE_KEYS if isinstance(info.get(k), (int, float))
        }
        LOGGER.info(
            "infer abs_ee=%s n_exec=%s n_ee=%s sec=%.2f fk=%.1f flow=%.1f conv=%.1f ms",
            tuple(actions.shape),
            n_exec,
            info.get("n_ee"),
            time.perf_counter() - t0,
            *(stage_ms.get(k, 0.0) for k in STAGE_KEYS[:3]),
        )
        return {
            "action": np.ascontiguousarray(actions, dtype=np.float32),
            "server_timing": {**stage_ms, "solve_ik": True},
        }


class WebsocketPolicyServer:
    def __init__(self, policy, host: str, port: int, metadata: dict | None = None):
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
            ping_interval=None,
            ping_timeout=None,
        ) as server:
            LOGGER.info("listening ws://%s:%s", self._host, self._port)
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        LOGGER.info("client %s connected", websocket.remote_address)
        packer = Packer()
        await websocket.send(packer.pack(self._metadata))
        while True:
            try:
                obs = unpackb(await websocket.recv())
                t0 = time.monotonic()
                action = self._policy.infer(obs)
                if not isinstance(action, dict):
                    action = {"action": action}
                server_timing = dict(action.get("server_timing") or {})
                server_timing["infer_ms"] = (time.monotonic() - t0) * 1000
                action["server_timing"] = server_timing
                await websocket.send(packer.pack(action))
            except websockets.ConnectionClosed:
                LOGGER.info("client %s closed", websocket.remote_address)
                break
            except Exception:
                tb = traceback.format_exc()
                LOGGER.exception("infer failed")
                await websocket.send(tb)
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error",
                )
                break


def _health_check(connection, request):
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.environ.get("WALLOSS_CKPT", str(DEFAULT_CKPT)))
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=int(os.environ.get("WALLOSS_PORT", "1239")))
    p.add_argument("--instruction", default=os.environ.get("WALLOSS_PROMPT", DEFAULT_PROMPT))
    p.add_argument("--n-exec", type=int, default=int(os.environ.get("WALLOSS_N_EXEC", "8")))
    p.add_argument("--ik-mode", default=os.environ.get("WALLOSS_IK_MODE", "abort_rest"))
    p.add_argument("--ik-solver", default=os.environ.get("WALLOSS_IK_SOLVER", "realman"))
    p.add_argument("--ik-traj-opt", default=os.environ.get("WALLOSS_IK_TRAJ_OPT", "none"))
    p.add_argument(
        "--ik-max-step-deg",
        type=float,
        default=float(os.environ.get("WALLOSS_IK_MAX_STEP_DEG", "40")),
    )
    p.add_argument("--ik-config", default=os.environ.get("WALLOSS_IK_CONFIG", ""))
    p.add_argument("--load-only", action="store_true")
    args = p.parse_args()

    ik_raw: dict = {
        "mode": args.ik_mode,
        "solver": args.ik_solver,
        "traj_opt": args.ik_traj_opt,
        "max_joint_step_deg": args.ik_max_step_deg,
    }
    cfg_path = str(args.ik_config or "").strip()
    if cfg_path:
        import yaml

        loaded = yaml.safe_load(Path(cfg_path).expanduser().read_text()) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"--ik-config 须是字典 YAML: {cfg_path}")
        if isinstance(loaded.get("ik"), dict):
            loaded = loaded["ik"]
        elif isinstance(loaded.get("policy"), dict) and isinstance(loaded["policy"].get("ik"), dict):
            loaded = loaded["policy"]["ik"]
        ik_raw.update(loaded)
        ik_raw.setdefault("mode", args.ik_mode)
        ik_raw.setdefault("solver", args.ik_solver)
        ik_raw.setdefault("traj_opt", args.ik_traj_opt)

    ik_settings = parse_ik_config(ik_raw)
    LOGGER.info(
        "loading WALL-OSS ckpt=%s n_exec=%s (infer 出 EE；solve_ik 再解关节)",
        args.ckpt,
        args.n_exec,
    )
    backend = WallOssTongPolicy(
        args.ckpt,
        instruction=args.instruction,
        n_exec=args.n_exec,
        ik=ik_settings,
    )
    if args.load_only:
        LOGGER.info("load-only OK norm_key=%s", backend.wrapper.norm_key)
        return

    policy = WallOssTongServePolicy(backend, args.instruction)
    meta = {
        "model": "walloss",
        "ckpt": str(Path(args.ckpt).resolve()),
        "n_exec": args.n_exec,
        "solve_ik": True,
        "action_horizon": backend.infer_cfg.action_horizon,
        "action_dim": 20,
        "rtc_capable": False,
    }
    WebsocketPolicyServer(policy, args.host, args.port, metadata=meta).serve_forever()


if __name__ == "__main__":
    main()
