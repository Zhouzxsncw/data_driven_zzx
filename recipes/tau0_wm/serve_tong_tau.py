#!/usr/bin/env python3
"""τ0-WM TongBot websocket server, VA-client compatible.

Protocol (same as LingBot-VA / WALL-OSS / GWP):
  handshake metadata  {model: tau0wm, action_horizon: 33, action_dim: 20, solve_ik: true, ...}
  reset:              {reset: True, prompt: [str]}
  compute_kv_cache:   ignored
  infer:              {obs: [cam_dict, ...], state: (16,), n_exec?: int}
  reply:              {action: (n_exec, 20)}  绝对 EE（模型输出）
  solve_ik:           {solve_ik: True, abs_ee: (n,20), state: (16,), n_exec?, ik?}
  reply:              {action: (n_planned, 16)}  RealMan 连续 IK

GPU 的 play 只出 EE。客户端读完这份输出，立刻同一条 ws 送回来解关节。
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

_THIS_DIR = Path(__file__).resolve().parent
_RECIPES_DIR = _THIS_DIR.parent
if str(_RECIPES_DIR) not in sys.path:
    sys.path.insert(0, str(_RECIPES_DIR))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from _server_logger import ServerRequestLogger  # noqa: E402
from tau_tongbot import (  # noqa: E402
    ACTION_CHUNK,
    CAM_KEYS,
    DEFAULT_CKPT,
    DEFAULT_DENOISE,
    DEFAULT_N_EXEC,
    DEFAULT_PORT,
    DEFAULT_PRETRAINED,
    DEFAULT_PROMPT,
    DEFAULT_STAT,
    DEFAULT_WAN,
    HANDSHAKE_MODEL,
    TONGBOT_DIM,
    WALL_OSS_SCRIPTS,
    apply_column_rot6d_patches,
    ee20_to_play_state_and_grippers,
    resolve_tau_ckpt,
    state16,
    tau_weights_ready,
    views_from_obs,
    write_deploy_yaml,
)

import msgpack
import numpy as np
import websockets
import websockets.asyncio.server as _server
import websockets.frames

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("tau0wm_serve")

STAGE_KEYS = ("fk_ms", "flow_ms", "convert_ms", "ik_ms")
TAU_ROOT = Path(os.environ.get("TAU_ROOT", "/home/xk/VLA/tau-0-wm"))


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


def _request_n_exec(msg: dict, default: int, horizon: int) -> int:
    raw = msg.get("n_exec", msg.get("n_action_steps"))
    try:
        n = int(default if raw is None else raw)
    except (TypeError, ValueError):
        n = int(default)
    return max(1, min(n, max(1, int(horizon))))


def _setup_tau_path() -> None:
    root = str(TAU_ROOT.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    os.chdir(root)


def _load_tau_policy(args) -> object:
    _setup_tau_path()
    import torch

    from web_infer_utils.TauPolicy import TauPolicy

    device = args.device
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(device)
    model_path = resolve_tau_ckpt(args.ckpt)
    runtime_yaml = Path(args.runtime_yaml)
    write_deploy_yaml(
        runtime_yaml,
        model_path=model_path,
        wan_root=args.wan_root,
        statistics_file=args.statistics,
    )
    LOGGER.info("deploy yaml=%s model_path=%s", runtime_yaml, model_path)
    policy = TauPolicy(str(runtime_yaml), device=torch.device(device))
    return policy


def _ik_settings_from_args(args) -> dict:
    raw = {
        "mode": args.ik_mode,
        "solver": args.ik_solver,
        "traj_opt": args.ik_traj_opt,
        "max_joint_step_deg": args.ik_max_step_deg,
    }
    cfg_path = str(getattr(args, "ik_config", "") or "").strip()
    if not cfg_path:
        return raw
    import yaml

    loaded = yaml.safe_load(Path(cfg_path).expanduser().read_text()) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"--ik-config 须是字典 YAML: {cfg_path}")
    if isinstance(loaded.get("ik"), dict):
        loaded = loaded["ik"]
    elif isinstance(loaded.get("policy"), dict) and isinstance(loaded["policy"].get("ik"), dict):
        loaded = loaded["policy"]["ik"]
    raw.update(loaded)
    return raw


def _make_fk_pipeline(n_exec: int, ik=None):
    """FK for play + IK for the follow-up ``solve_ik`` RPC (column rot6d)."""
    apply_column_rot6d_patches(WALL_OSS_SCRIPTS)
    from ee_infer_pipeline import EEInferPipeline, plan_to_logical16
    from ik_config import parse_ik_config

    settings = parse_ik_config(ik)
    pipe = EEInferPipeline(n_exec=n_exec, ik=settings)
    LOGGER.info(
        "FK+IK pipeline ready mode=%s (infer 出 EE；solve_ik 再解关节)",
        settings.mode,
    )
    return pipe, plan_to_logical16


def _plans_to_logical16(plans, seed16, n_exec, plan_to_logical16):
    if not plans:
        hold = np.asarray(seed16, dtype=np.float32).reshape(1, TONGBOT_DIM)
        return hold, {
            "aborted_ik": True,
            "n_planned": 0,
            "n_requested": int(n_exec),
            "cartesian_limited": False,
            "held_ik": True,
            "max_step_deg": 0.0,
            "rejected_step_deg": 0.0,
            "cartesian_t": 0.0,
        }
    actions = np.stack(
        [np.asarray(plan_to_logical16(p), dtype=np.float32) for p in plans],
        axis=0,
    )
    return np.ascontiguousarray(actions, dtype=np.float32), {
        "aborted_ik": len(plans) < int(n_exec),
        "n_planned": len(plans),
        "n_requested": int(n_exec),
        "cartesian_limited": any(bool(p.cartesian_limited) for p in plans),
        "held_ik": any(bool(p.held) for p in plans),
        "max_step_deg": max(
            (max(p.left_step_max_deg, p.right_step_max_deg) for p in plans),
            default=0.0,
        ),
        "rejected_step_deg": max((float(p.rejected_step_deg) for p in plans), default=0.0),
        "cartesian_t": min((float(p.cartesian_t) for p in plans), default=1.0),
    }


class TauTongServePolicy:
    def __init__(self, backend, ik, plan_to_logical16, instruction: str, n_exec: int, denoise: int):
        self.backend = backend
        self.ik = ik
        self._plan_to_logical16 = plan_to_logical16
        self.instruction = instruction
        self.n_exec = int(n_exec)
        self.denoise = int(denoise)
        self._need_reset_context = True

    def reset(self, prompt: str | None, ik: dict | None = None) -> None:
        if prompt:
            self.instruction = str(prompt).strip() or self.instruction
        if ik is not None:
            self.ik.apply_ik_settings(ik)
        if hasattr(self.backend, "reset"):
            self.backend.reset()
        self._need_reset_context = True
        LOGGER.info(
            "reset prompt=%s ik_mode=%s",
            self.instruction[:160],
            self.ik.ik_settings.mode,
        )

    def solve_ik(self, msg: dict) -> dict:
        if isinstance(msg.get("ik"), dict):
            self.ik.apply_ik_settings(msg["ik"])
        joints = state16(msg.get("state")).astype(np.float64)
        abs_ee = np.asarray(msg.get("abs_ee"), dtype=np.float64)
        if abs_ee.ndim == 1:
            abs_ee = abs_ee[np.newaxis, :]
        if abs_ee.shape[-1] != 20:
            raise ValueError(f"solve_ik 需要 abs_ee EE20，得到 {abs_ee.shape}")
        n_exec = _request_n_exec(msg, self.n_exec, ACTION_CHUNK)
        t_ik = time.perf_counter()
        plans = self.ik.ik_abs_chunk(joints, abs_ee, n_exec=n_exec)
        ik_ms = (time.perf_counter() - t_ik) * 1000
        actions, info = _plans_to_logical16(
            plans, joints, n_exec, self._plan_to_logical16
        )
        LOGGER.info(
            "solve_ik action=%s n_exec=%s n_planned=%s aborted=%s cart_lim=%s held=%s "
            "mode=%s step_deg=%.2f rejected=%.1f cart_t=%.2f ik=%.1f ms",
            tuple(actions.shape),
            n_exec,
            info["n_planned"],
            info["aborted_ik"],
            info["cartesian_limited"],
            info["held_ik"],
            self.ik.ik_settings.mode,
            info["max_step_deg"],
            info["rejected_step_deg"],
            info["cartesian_t"],
            ik_ms,
        )
        return {
            "action": actions,
            "_timing": {"ik_ms": ik_ms},
            "server_timing": {
                "ik_ms": ik_ms,
                **{k: info[k] for k in (
                    "aborted_ik", "n_planned", "n_requested",
                    "cartesian_limited", "held_ik",
                    "max_step_deg", "rejected_step_deg", "cartesian_t",
                )},
                "ik_mode": self.ik.ik_settings.mode,
            },
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
        t_pre = time.perf_counter()
        views = views_from_obs(last)
        joints = state16(msg.get("state")).astype(np.float64)
        n_exec = _request_n_exec(msg, self.n_exec, ACTION_CHUNK)
        preprocess_ms = (time.perf_counter() - t_pre) * 1000

        t_fk = time.perf_counter()
        ee20 = self.ik.observe_ee20(joints)
        play_state, grippers = ee20_to_play_state_and_grippers(ee20)
        fk_ms = (time.perf_counter() - t_fk) * 1000

        t_flow = time.perf_counter()
        abs_ee = self.backend.play(
            views,
            prompt=self.instruction,
            state=play_state,
            gripper_states=grippers,
            num_inference_steps=self.denoise,
            execution_step=n_exec,
            reset_context=self._need_reset_context,
        )
        self._need_reset_context = False
        flow_ms = (time.perf_counter() - t_flow) * 1000

        abs_ee = np.asarray(abs_ee, dtype=np.float64)
        if abs_ee.ndim == 1:
            abs_ee = abs_ee[np.newaxis, :]
        n_out = max(1, min(int(abs_ee.shape[0]), n_exec))
        actions = np.ascontiguousarray(abs_ee[:n_out], dtype=np.float32)
        stage_ms = {
            "preprocess_ms": preprocess_ms,
            "fk_ms": fk_ms,
            "flow_ms": flow_ms,
            "convert_ms": 0.0,
            "ik_ms": 0.0,
        }
        LOGGER.info(
            "infer abs_ee=%s n_exec=%s pre=%.1f fk=%.1f flow=%.1f ms range=[%.3f,%.3f]",
            tuple(actions.shape),
            n_exec,
            preprocess_ms,
            fk_ms,
            flow_ms,
            float(actions.min()),
            float(actions.max()),
        )
        return {
            "action": actions,
            "_timing": stage_ms,
            "server_timing": {
                **{k: float(stage_ms[k]) for k in STAGE_KEYS if k in stage_ms},
                "n_ee": int(actions.shape[0]),
                "solve_ik": True,
            },
        }


class WebsocketPolicyServer:
    def __init__(self, policy, host: str, port: int, metadata: dict | None = None,
                 srv_logger: ServerRequestLogger | None = None):
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._srv_logger = srv_logger
        self._request_id = 0

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        if self._srv_logger is not None:
            self._srv_logger.start()
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
        client_addr = str(websocket.remote_address)
        LOGGER.info("client %s connected", client_addr)
        if self._srv_logger:
            self._srv_logger.log_connection(
                client_addr=client_addr, event="connect", metadata=self._metadata
            )
        packer = Packer()
        await websocket.send(packer.pack(self._metadata))
        while True:
            try:
                obs = unpackb(await websocket.recv())
                self._request_id += 1
                t0 = time.monotonic()
                result = self._policy.infer(obs)
                if not isinstance(result, dict):
                    result = {"action": result}
                infer_ms = (time.monotonic() - t0) * 1000
                timing = result.pop("_timing", None)
                server_timing = dict(result.get("server_timing") or {})
                server_timing["infer_ms"] = infer_ms
                result["server_timing"] = server_timing
                t_send = time.monotonic()
                await websocket.send(packer.pack(result))
                send_ms = (time.monotonic() - t_send) * 1000
                if self._srv_logger:
                    log_timing = dict(timing) if timing else {}
                    log_timing["total_ms"] = round(infer_ms, 2)
                    log_timing["network_send_ms"] = round(send_ms, 2)
                    self._srv_logger.log_request(
                        request_id=self._request_id,
                        client_addr=client_addr,
                        infer_ms=infer_ms,
                        request_data=obs if isinstance(obs, dict) else None,
                        response_data=result,
                        timing=log_timing,
                    )
            except websockets.ConnectionClosed:
                LOGGER.info("client %s closed", client_addr)
                if self._srv_logger:
                    self._srv_logger.log_connection(client_addr=client_addr, event="disconnect")
                break
            except Exception:
                tb = traceback.format_exc()
                LOGGER.exception("infer failed")
                if self._srv_logger:
                    self._srv_logger.log_request(
                        request_id=self._request_id,
                        client_addr=client_addr,
                        infer_ms=(time.monotonic() - t0) * 1000,
                        error=tb.splitlines()[-1] if tb else "unknown",
                    )
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
    p.add_argument("--ckpt", default=os.environ.get("TAU_CKPT", DEFAULT_CKPT))
    p.add_argument("--wan-root", default=os.environ.get("WAN_ROOT", DEFAULT_WAN))
    p.add_argument("--statistics", default=os.environ.get("TAU_STAT", DEFAULT_STAT))
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=int(os.environ.get("TAU_PORT", str(DEFAULT_PORT))))
    p.add_argument("--instruction", default=os.environ.get("TAU_PROMPT", DEFAULT_PROMPT))
    p.add_argument("--denoise", type=int, default=int(os.environ.get("TAU_DENOISE", str(DEFAULT_DENOISE))))
    p.add_argument("--n-exec", type=int, default=int(os.environ.get("TAU_N_EXEC", str(DEFAULT_N_EXEC))))
    p.add_argument("--ik-mode", default=os.environ.get("TAU_IK_MODE", "abort_rest"))
    p.add_argument("--ik-solver", default=os.environ.get("TAU_IK_SOLVER", "realman"))
    p.add_argument("--ik-traj-opt", default=os.environ.get("TAU_IK_TRAJ_OPT", "none"))
    p.add_argument(
        "--ik-max-step-deg",
        type=float,
        default=float(os.environ.get("TAU_IK_MAX_STEP_DEG", "40")),
    )
    p.add_argument("--ik-config", default=os.environ.get("TAU_IK_CONFIG", ""))
    p.add_argument("--device", default=os.environ.get("TAU_DEVICE", "cuda:0"))
    p.add_argument("--pretrained-only", action="store_true")
    p.add_argument("--load-only", action="store_true")
    p.add_argument("--log-dir", default=os.environ.get("TAU_LOG_DIR", ""))
    p.add_argument("--no-request-log", action="store_true")
    p.add_argument(
        "--runtime-yaml",
        default=os.environ.get(
            "TAU_RUNTIME_YAML",
            str(_THIS_DIR / "configs" / "deployment" / "_runtime_tongbot_00036_ee.yaml"),
        ),
    )
    args = p.parse_args()

    if args.pretrained_only:
        args.ckpt = os.environ.get("TAU_WEIGHTS", DEFAULT_PRETRAINED)
        LOGGER.warning("pretrained-only: 加载发布基座，不是 00036 EE SFT")

    if not tau_weights_ready(args.ckpt):
        LOGGER.error(
            "权重不存在: %s\n"
            "推理只要 step_N 里的 diffusion_pytorch_model.bin（约 11G），"
            "不要拉 latest_accelerator_state（optimizer）。\n"
            "  bash recipes/tau0_wm/pull_tau_ckpt.sh\n"
            "远端: %s/step_30000（以实际落盘为准）",
            args.ckpt,
            os.environ.get(
                "TAU_H100_OUTPUT",
                "/data1/tau0wm/outputs/tongbot_00036_ee/tau0wm_vam5b_ee_rela_192x256_bs1",
            ),
        )
        sys.exit(1)
    if not Path(args.statistics).is_file():
        LOGGER.error("统计文件不存在: %s", args.statistics)
        sys.exit(1)
    wan = Path(args.wan_root)
    if not (wan / "Wan2.2_VAE.pth").is_file():
        LOGGER.error("Wan VAE 不存在: %s", wan / "Wan2.2_VAE.pth")
        sys.exit(1)

    load_t0 = time.time()
    backend = _load_tau_policy(args)
    ik_raw = _ik_settings_from_args(args)
    ik, plan_to_logical16 = _make_fk_pipeline(args.n_exec, ik=ik_raw)
    load_sec = time.time() - load_t0
    LOGGER.info(
        "model ready action_horizon=%d denoise=%d n_exec=%d solve_ik=True (%.1fs)",
        ACTION_CHUNK,
        args.denoise,
        args.n_exec,
        load_sec,
    )

    srv_logger = None
    if not args.no_request_log:
        log_dir = Path(args.log_dir) if args.log_dir else _THIS_DIR
        log_path = log_dir / "tau0wm_requests.jsonl"
        srv_logger = ServerRequestLogger(log_path, gpu_device=0, model_name=HANDSHAKE_MODEL)
        LOGGER.info("请求级日志: %s", log_path)

    if args.load_only:
        LOGGER.info("load-only OK")
        if srv_logger:
            srv_logger.close()
        return

    if srv_logger:
        srv_logger.log_model_loaded(
            model_name=HANDSHAKE_MODEL,
            ckpt_path=str(Path(args.ckpt).resolve()),
            load_time_sec=load_sec,
            extra={
                "action_horizon": ACTION_CHUNK,
                "denoise": args.denoise,
                "n_exec": args.n_exec,
                "solve_ik": True,
                "rot6d": "columns",
            },
        )

    policy = TauTongServePolicy(
        backend, ik, plan_to_logical16, args.instruction, args.n_exec, args.denoise
    )
    meta = {
        "model": HANDSHAKE_MODEL,
        "ckpt": str(Path(args.ckpt).resolve()),
        "action_horizon": ACTION_CHUNK,
        "n_exec": args.n_exec,
        "denoise": args.denoise,
        "solve_ik": True,
        "action_dim": 20,
        "model_action_dim": 20,
        "cameras": list(CAM_KEYS),
        "img_size": [192, 256],
        "rot6d": "columns",
        "rtc_capable": False,
    }
    LOGGER.info("server listening on %s:%s", args.host, args.port)
    server = WebsocketPolicyServer(policy, args.host, args.port, metadata=meta, srv_logger=srv_logger)
    try:
        server.serve_forever()
    finally:
        if srv_logger:
            summary = srv_logger.close()
            LOGGER.info(
                "服务端统计: %d 请求 (%d 错误), 运行 %.1fs, 日志: %s",
                summary["total_requests"],
                summary.get("error_count", 0),
                summary["duration_sec"],
                summary["log_path"],
            )


if __name__ == "__main__":
    main()
