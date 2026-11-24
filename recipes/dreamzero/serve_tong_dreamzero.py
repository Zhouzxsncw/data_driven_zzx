#!/usr/bin/env python3
"""DreamZero TongBot websocket server, VA-client compatible.

Speaks the same msgpack protocol as LingBot-VA (`run_lingbotva_infer.py --mode server`):
  handshake metadata
  reset:            {reset: True, prompt: [str]}
  compute_kv_cache: ignored (causal KV is built inside WANPolicyHead)
  infer:            {obs: [cam_dict, ...], state: (16,)}
  reply:            {action: (T, 16)}  # T = action_horizon (16)

Causal video (same as eval_utils/serve_dreamzero_wan22.py):
  first infer in a session uses 1 frame and warms KV;
  later infers use the last FRAMES_PER_CHUNK=4 frames and append to KV.
  T=1 on every packet would reset current_start_frame and break temporal context.

Camera mapping (00073 / TongBot):
  observation.images.left  -> video.left  (main)
  observation.images.front -> video.front
  observation.images.right -> video.right
"""
from __future__ import annotations

import argparse
import asyncio
import functools
import http
import json
import logging
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from pathlib import Path

os.environ.setdefault("DREAMZERO_PATH", "/home/xk/VLA/dreamzero")
os.environ.setdefault("EMBODIED_PATH", "/home/xk/VLA/RLinf/examples/sft")
sys.path.insert(0, "/home/xk/VLA/dreamzero_staging")
sys.path.insert(0, os.environ["DREAMZERO_PATH"])
sys.path.insert(0, "/home/xk/VLA/RLinf")

import msgpack
import numpy as np
import torch
import websockets
import websockets.asyncio.server as _server
import websockets.frames

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
LOGGER = logging.getLogger("dreamzero_serve")

# 默认用 00073 SFT 收官权重（full_full20260911k，step 20000）；
# 回退早期 ckpt：--ckpt .../global_step_10000/actor/model_state_dict/infer_weights.pt
DEFAULT_CKPT = (
    "/home/xk/VLA/h100_ckpts/dreamzero_14b/global_step_20000/"
    "actor/model_state_dict/infer_weights.pt"
)
DEFAULT_CONFIG = "tong00073_infer_dreamzero_14b_local"


def model_family_id(config_name: str) -> str:
    """握手用的稳定模型标识（不随 hydra config 名变化）。

    客户端 infer/policies/dreamzero_adapter.py 用 ``lingbot_expected_model``
    精确比对 metadata["model"]，所以这里必须与其它 serve_tong_*.py 一样发
    家族名（dreamzero_5b / dreamzero_14b），不能发 config 名——否则
    ``--config-name`` 一改客户端就报"连错模型"。config 名另放 metadata["config"]。
    """
    return "dreamzero_5b" if "5b" in config_name else "dreamzero_14b"


DEFAULT_PROMPT = (
    "Using both hands, pick up the courier items from the green conveyor belt. "
    "Place the white courier bags into the left box and the brown courier boxes into the right box."
)
CAM_LEFT = "observation.images.left"
CAM_FRONT = "observation.images.front"
CAM_RIGHT = "observation.images.right"
# Official 5B causal serve: first call T=1, later calls T=4 (VAE 4x + num_frame_per_block=2).
FRAMES_PER_CHUNK = 4
_MAX_BUFFER_FRAMES = 16

TZ = timezone(timedelta(hours=8))


def _now_tag() -> str:
    return datetime.now(TZ).strftime("%Y%m%d-%H%M%S")


def _now_iso() -> str:
    return datetime.now(TZ).isoformat()


class VisionLatentDumper:
    """异步落盘 DreamZero video_pred latent（[B, C, T, H, W] VAE 空间）。

    forward hook 抓到 model_pred["video_pred"] 后 clone 到 CPU，
    单线程池写 .pt，不挡推理回包。离线用 Wan2.2 VAE decode 成 dream video。
    """

    SCHEMA = "dreamzero.vision_latent.v1"

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.session: Path | None = None
        self.idx = 0
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dz_lat_dump")

    def new_session(self, prompt: str = "") -> Path:
        path = self.root / f"session_{_now_tag()}"
        path.mkdir(parents=True, exist_ok=True)
        meta = {
            "schema": self.SCHEMA,
            "created_at": _now_iso(),
            "prompt": prompt,
            "model": "dreamzero",
            "decode_hint": "python recipes/dreamzero/offline_decode_latents.py --session " + str(path),
        }
        (path / "session.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        self.session = path
        self.idx = 0
        LOGGER.info("dreamzero latent session %s", path)
        return path

    def dump(self, latent: torch.Tensor, *, prompt: str = "",
             infer_sec: float | None = None, extra: dict | None = None) -> None:
        if self.session is None:
            self.new_session(prompt)
        assert self.session is not None
        idx = self.idx
        self.idx += 1
        cpu = latent.detach().to(device="cpu", dtype=torch.bfloat16).contiguous().clone()
        payload = {
            "schema": self.SCHEMA,
            "chunk_idx": idx,
            "recorded_at": _now_iso(),
            "latent": cpu,
            "latent_shape": list(cpu.shape),
            "latent_dtype": str(cpu.dtype).replace("torch.", ""),
            "prompt": prompt,
            "infer_sec": infer_sec,
            **(extra or {}),
        }
        pt_path = self.session / f"chunk_{idx:06d}.pt"
        self._pool.submit(self._write_chunk, pt_path, payload)

    @staticmethod
    def _write_chunk(pt_path: Path, payload: dict) -> None:
        try:
            torch.save(payload, pt_path)
            row = {
                "chunk_idx": payload.get("chunk_idx"),
                "recorded_at": payload.get("recorded_at"),
                "pt": pt_path.name,
                "latent_shape": payload.get("latent_shape"),
                "infer_sec": payload.get("infer_sec"),
            }
            with (pt_path.parent / "index.jsonl").open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception:
            LOGGER.exception("dreamzero latent 落盘失败 %s", pt_path)

    def close(self) -> None:
        self._pool.shutdown(wait=False)


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


def _to_hwc_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim != 3 or arr.shape[-1] not in (1, 3):
        raise ValueError(f"期望 HWC/CHW 图像，得到 shape={arr.shape}")
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if np.issubdtype(arr.dtype, np.floating):
        max_v = float(np.nanmax(arr)) if arr.size else 1.0
        if max_v <= 1.0 + 1e-3:
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)
    return arr


def _cam_frames(obs_list: list, key: str) -> list[np.ndarray]:
    """All HWC frames for one camera in this message (VA chunk may have 2 @ 5fps)."""
    frames: list[np.ndarray] = []
    for item in obs_list:
        if key not in item:
            raise KeyError(f"obs 缺相机 {key}；keys={list(item)}")
        frames.append(_to_hwc_uint8(item[key]))
    return frames


def _select_chunk_frames(buffer: list[np.ndarray], num_frames: int) -> np.ndarray:
    """Last ``num_frames`` frames, prepend-pad with the oldest if the buffer is short."""
    if not buffer:
        raise RuntimeError("相机帧缓冲为空")
    if len(buffer) >= num_frames:
        frames = buffer[-num_frames:]
    else:
        frames = list(buffer)
        while len(frames) < num_frames:
            frames.insert(0, buffer[0])
    return np.stack(frames, axis=0)


def build_model(ckpt: str, device: str, config_name: str = DEFAULT_CONFIG,
                overrides: list[str] | None = None):
    """overrides: hydra CLI 风格覆盖（``data.train_data_paths=...``），不用改 YAML 就能换字段。"""
    from hydra import compose, initialize_config_dir
    from rlinf.models.embodiment.dreamzero import get_model
    from infer_tong_dreamzero import _load_weights

    cfg_dir = os.environ.get(
        "DREAMZERO_CFG_DIR", "/home/xk/VLA/RLinf/examples/sft/config"
    )
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(config_name=config_name, overrides=list(overrides or []))

    model = get_model(cfg.actor.model)
    _load_weights(model, ckpt)
    model.eval()
    model.to(device=device, dtype=torch.bfloat16)
    return model


class DreamZeroTongPolicy:
    def __init__(self, model, instruction: str, device: str = "cuda",
                 dump_video_root: str | None = None):
        self.model = model
        self.instruction = instruction
        self.device = device
        self._frame_buffers: dict[str, list[np.ndarray]] = {
            CAM_LEFT: [],
            CAM_FRONT: [],
            CAM_RIGHT: [],
        }
        self._is_first_call = True
        # ---- dream video latent 落盘 ----
        self._latent_dumper: VisionLatentDumper | None = None
        self._hook_handle = None
        self._captured_video_pred: torch.Tensor | None = None
        if dump_video_root:
            self._latent_dumper = VisionLatentDumper(dump_video_root)
            self._hook_handle = self.model.register_forward_hook(self._capture_video_pred)
            LOGGER.info("dream video latent dump enabled → %s", dump_video_root)

    def _clear_causal_state(self) -> None:
        for key in self._frame_buffers:
            self._frame_buffers[key] = []
        self._is_first_call = True
        ah = getattr(self.model, "action_head", None)
        if ah is not None:
            ah.language = None
            if hasattr(ah, "current_start_frame"):
                ah.current_start_frame = 0

    def _capture_video_pred(self, module, args, output):
        """Forward hook: 从 lazy_joint_video_action_causal 的 BatchFeature 抓 video_pred。"""
        if hasattr(output, "__getitem__"):
            try:
                vp = output["video_pred"]
                if isinstance(vp, torch.Tensor):
                    self._captured_video_pred = vp.detach()
            except (KeyError, IndexError):
                pass

    def reset(self, prompt: str | None) -> None:
        if prompt:
            self.instruction = str(prompt).strip() or self.instruction
        self._clear_causal_state()
        if self._latent_dumper:
            self._latent_dumper.new_session(self.instruction)
        LOGGER.info("reset prompt=%s", self.instruction[:160])

    def _append_obs_frames(self, obs_list: list) -> None:
        for key in (CAM_LEFT, CAM_FRONT, CAM_RIGHT):
            buf = self._frame_buffers[key]
            buf.extend(_cam_frames(obs_list, key))
            if len(buf) > _MAX_BUFFER_FRAMES:
                del buf[: len(buf) - _MAX_BUFFER_FRAMES]

    def infer(self, msg: dict) -> dict:
        if msg.get("reset"):
            prompt = msg.get("prompt")
            if isinstance(prompt, (list, tuple)) and prompt:
                prompt = prompt[0]
            self.reset(prompt if isinstance(prompt, str) else None)
            return {}
        if msg.get("compute_kv_cache"):
            # VA client still sends this; DreamZero KV is the WAN causal cache.
            return {}

        obs_list = msg.get("obs")
        if not obs_list:
            raise KeyError("infer 消息缺少 obs")
        state = np.asarray(msg.get("state"), dtype=np.float32).reshape(-1)
        if state.size < 16:
            pad = np.zeros(16, dtype=np.float32)
            pad[: state.size] = state
            state = pad
        else:
            state = state[:16]

        self._append_obs_frames(obs_list)
        num_frames = 1 if self._is_first_call else FRAMES_PER_CHUNK
        left = _select_chunk_frames(self._frame_buffers[CAM_LEFT], num_frames)
        front = _select_chunk_frames(self._frame_buffers[CAM_FRONT], num_frames)
        right = _select_chunk_frames(self._frame_buffers[CAM_RIGHT], num_frames)
        h, w = left.shape[1:3]
        env_obs = {
            "main_images": left[None, ...],
            "extra_view_images": front[None, ...],
            "wrist_images": right[None, ...],
            "states": state[None, :],
            "task_descriptions": [self.instruction],
        }
        ah = getattr(self.model, "action_head", None)
        sf_before = getattr(ah, "current_start_frame", None)
        self._captured_video_pred = None  # hook 会在推理中填充
        t0 = time.perf_counter()
        with torch.inference_mode():
            actions, _ = self.model.predict_action_batch(env_obs, mode="eval")
        infer_sec = time.perf_counter() - t0
        # 落盘 video latent（hook 捕获的 video_pred）
        if self._latent_dumper is not None and self._captured_video_pred is not None:
            self._latent_dumper.dump(
                self._captured_video_pred,
                prompt=self.instruction,
                infer_sec=infer_sec,
                extra={"is_first_call": num_frames == 1, "num_input_frames": num_frames},
            )
            self._captured_video_pred = None
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 3:
            actions = actions[0]
        elif actions.ndim == 1:
            actions = actions[None, :]
        if actions.shape[-1] > 16:
            actions = actions[..., :16]
        self._is_first_call = False
        sf_after = getattr(ah, "current_start_frame", None)
        LOGGER.info(
            "infer T=%d first=%s start_frame %s→%s action=%s sec=%.2f img=%dx%d buf=%d prompt=%s%s",
            num_frames,
            num_frames == 1,
            sf_before,
            sf_after,
            tuple(actions.shape),
            infer_sec,
            h,
            w,
            len(self._frame_buffers[CAM_LEFT]),
            self.instruction[:80],
            " +dream" if self._latent_dumper else "",
        )
        return {"action": np.ascontiguousarray(actions)}


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
                action["server_timing"] = {"infer_ms": (time.monotonic() - t0) * 1000}
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--config-name", default=DEFAULT_CONFIG,
                   help="hydra config in RLinf/examples/sft/config（14B/5B 切换）")
    p.add_argument("--set", dest="cfg_set", action="append", default=None,
                   metavar="KEY=VAL",
                   help="hydra 字段覆盖，可重复；例：--set data.train_data_paths=/data/x "
                        "--set actor.model.metadata_json_path=/data/meta.json")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=1235)
    p.add_argument("--device", default="cuda")
    p.add_argument("--instruction", default=DEFAULT_PROMPT)
    _default_dump = os.environ.get("DREAMZERO_DUMP_VIDEO", "")
    p.add_argument("--dump-video", default=_default_dump,
                   help="dream video latent 落盘目录（非空即开启，每包捕获 video_pred .pt）")
    args = p.parse_args()

    if not os.path.isfile(args.ckpt):
        raise SystemExit(f"ckpt 不存在: {args.ckpt}")

    overrides = list(args.cfg_set or [])
    LOGGER.info("loading ckpt %s (config=%s overrides=%s)",
                args.ckpt, args.config_name, overrides or "无")
    model = build_model(args.ckpt, args.device, args.config_name, overrides)
    dump_video_root = args.dump_video or None
    policy = DreamZeroTongPolicy(model, args.instruction, args.device,
                                 dump_video_root=dump_video_root)
    LOGGER.info("model ready, serving on %s:%s", args.host, args.port)
    WebsocketPolicyServer(
        policy,
        host=args.host,
        port=args.port,
        metadata={
            # 家族名做身份校验（客户端 lingbot_expected_model 精确比对），
            # config 名单独给出，便于区分 14B/5B 与排查连错端口
            "model": model_family_id(args.config_name),
            "config": args.config_name,
            "ckpt": args.ckpt,
            # 实际生效的字段覆盖，便于台账/回放复现
            "overrides": overrides,
            "action_dim": 16,
            "action_horizon": 16,
            "frames_per_chunk": FRAMES_PER_CHUNK,
            "dump_video": bool(dump_video_root),
            "rtc_capable": False,
        },
    ).serve_forever()


if __name__ == "__main__":
    main()
