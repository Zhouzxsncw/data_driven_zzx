#!/usr/bin/env python3
"""Cosmos3-Nano TongBot websocket 服务（协议对齐 LingBot-VA / DreamZero）。

默认：Cosmo3OmniPipeline 加载 Nano，再叠本机最新 00073 SFT DCP
（iter_000003000 / net_ema）。三路相机拼 concat_view，quantile 反归一化
到 16D 关节弧度。

  bash /home/xk/VLA/cosmos3_staging/launch_cosmos3_server.sh
  bash /home/xk/VLA/cosmos3_staging/launch_cosmos3_server.sh --pretrained-only
"""
from __future__ import annotations

import argparse
import asyncio
import functools
import http
import json
import logging
import os
import re
import sys
import time
import traceback

import msgpack
import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import websockets
import websockets.asyncio.server as _server
import websockets.frames
from PIL import Image
from torch.distributed.checkpoint.filesystem import FileSystemReader

_STAGING = os.path.dirname(os.path.abspath(__file__))
if _STAGING not in sys.path:
    sys.path.insert(0, _STAGING)
from vision_latent_io import VisionLatentDumper

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("cosmos3_serve")

DEFAULT_MODEL = os.environ.get(
    "COSMOS3_MODEL", "/home/xk/VLA/cosmos_h100_setup/Cosmos3-Nano"
)
DEFAULT_CKPT = os.environ.get(
    "COSMOS3_CKPT", "/home/xk/VLA/h100_ckpts/cosmos3_00073/iter_000001500"
)
DEFAULT_STATS = os.environ.get(
    "COSMOS3_ACTION_STATS",
    "/home/xk/VLA/cosmos-framework/cosmos_framework/data/generator/action/normalizer_stats/tongbot_16d_00073.json",
)
DEFAULT_PROMPT = (
    "Using both hands, pick up the courier items from the green conveyor belt. "
    "Place the white courier bags into the left box and the brown courier boxes into the right box."
)
# UniPC + flow_shift=10；官方 action warmup 用 8 步。
# 20 步去噪约 1.6s；再加 VAE decode ≈ 2.3s。15fps 执行 16 步只要 1.07s。
# 默认 20 步 + 不解视频。要加速：COSMOS3_STEPS=10。
DEFAULT_STEPS = int(os.environ.get("COSMOS3_STEPS", "20"))
DEFAULT_GUIDANCE = float(os.environ.get("COSMOS3_GUIDANCE", "1.0"))
FLOW_SHIFT = 10.0
CAM_FRONT = "observation.images.front"
CAM_LEFT = "observation.images.left"
CAM_RIGHT = "observation.images.right"
TONGBOT_DIM = 16
ACTION_HORIZON = 16
IMAGE_SIZE = 256
TONGBOT_DOMAIN_ID = 30

_DIFFUSERS_KEY_MAPPING_RES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^transformer\."), ""),
    (re.compile(r"^action_proj_in\."), "action2llm."),
    (re.compile(r"^action_proj_out\."), "llm2action."),
    (re.compile(r"^audio_proj_in\."), "sound2llm."),
    (re.compile(r"^audio_proj_out\."), "llm2sound."),
    (re.compile(r"^audio_modality_embed$"), "sound_modality_embed"),
    (re.compile(r"^proj_in\."), "vae2llm."),
    (re.compile(r"^proj_out\."), "llm2vae."),
    (re.compile(r"^time_embedder\.linear_1\."), "time_embedder.mlp.0."),
    (re.compile(r"^time_embedder\.linear_2\."), "time_embedder.mlp.2."),
    (re.compile(r"\.self_attn\.to_q\."), ".self_attn.q_proj."),
    (re.compile(r"\.self_attn\.to_k\."), ".self_attn.k_proj."),
    (re.compile(r"\.self_attn\.to_v\."), ".self_attn.v_proj."),
    (re.compile(r"\.self_attn\.to_out\."), ".self_attn.o_proj."),
    (re.compile(r"\.self_attn\.norm_q\."), ".self_attn.q_norm."),
    (re.compile(r"\.self_attn\.norm_k\."), ".self_attn.k_norm."),
    (re.compile(r"\.self_attn\.add_q_proj\."), ".self_attn.q_proj_moe_gen."),
    (re.compile(r"\.self_attn\.add_k_proj\."), ".self_attn.k_proj_moe_gen."),
    (re.compile(r"\.self_attn\.add_v_proj\."), ".self_attn.v_proj_moe_gen."),
    (re.compile(r"\.self_attn\.to_add_out\."), ".self_attn.o_proj_moe_gen."),
    (re.compile(r"\.self_attn\.norm_added_q\."), ".self_attn.q_norm_moe_gen."),
    (re.compile(r"\.self_attn\.norm_added_k\."), ".self_attn.k_norm_moe_gen."),
    (re.compile(r"^lm_head\."), "language_model.lm_head."),
    (
        re.compile(r"^(embed_tokens\.|layers\.|norm(?:_moe_gen)?\.)(.*)$"),
        r"language_model.model.\1\2",
    ),
)


def _diffusers_to_net_key(name: str) -> str:
    for pattern, replacement in _DIFFUSERS_KEY_MAPPING_RES:
        name = pattern.sub(replacement, name)
    return name


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
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
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


def _as_numpy_f32(raw) -> np.ndarray:
    """Cosmos3 pipe 输出 bf16 Tensor，numpy 不能直接 asarray。"""
    if isinstance(raw, torch.Tensor):
        return raw.detach().to(dtype=torch.float32, device="cpu").numpy()
    return np.asarray(raw, dtype=np.float32)


def _to_tongbot_16d(raw) -> np.ndarray:
    arr = _as_numpy_f32(raw)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"action 期望 [T, D]，得到 {arr.shape}")
    out = np.zeros((arr.shape[0], TONGBOT_DIM), dtype=np.float32)
    width = min(arr.shape[1], TONGBOT_DIM)
    out[:, :width] = arr[:, :width]
    return out


def _concat_view(front: np.ndarray, left: np.ndarray | None, right: np.ndarray | None) -> Image.Image:
    top = Image.fromarray(front).resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    if left is None or right is None:
        LOGGER.warning("缺腕部相机，只用 front（SFT 训练是三路 concat_view）")
        return top
    half = IMAGE_SIZE // 2
    left_im = Image.fromarray(left).resize((half, half), Image.Resampling.BILINEAR)
    right_im = Image.fromarray(right).resize((half, half), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE + half))
    canvas.paste(top, (0, 0))
    canvas.paste(left_im, (0, IMAGE_SIZE))
    canvas.paste(right_im, (half, IMAGE_SIZE))
    return canvas


def _json_prompt(instruction: str, *, fps: int = 15) -> str:
    duration = (ACTION_HORIZON + 1) / float(fps)
    payload = {
        "cinematography": {
            "framing": (
                "The top row shows the front camera looking at the workspace. "
                "The bottom row contains two horizontally concatenated wrist-mounted "
                "camera views: the left-hand camera on the left and the right-hand "
                "camera on the right."
            )
        },
        "actions": [
            {
                "time": f"0.00-{duration:.2f}s",
                "description": instruction if instruction.endswith(".") else instruction + ".",
                "idle_frame": "0 out of 16",
            }
        ],
        "duration": f"{int(duration)}s",
        "fps": fps,
        "resolution": f"{IMAGE_SIZE + IMAGE_SIZE // 2}x{IMAGE_SIZE}",
        "aspect_ratio": "3,2",
    }
    return json.dumps(payload, ensure_ascii=False)


def _load_quantile_stats(path: str) -> tuple[np.ndarray, np.ndarray]:
    raw = json.loads(open(path, encoding="utf-8").read())
    stats = raw.get("global", raw)
    q01 = np.asarray(stats["q01"], dtype=np.float32)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    rng = np.clip(q99 - q01, 1e-6, None)
    if q01.shape[0] != TONGBOT_DIM:
        raise SystemExit(f"stats dim={q01.shape[0]}，期望 {TONGBOT_DIM}")
    return q01, rng


def _denorm_quantile(action: np.ndarray, q01: np.ndarray, rng: np.ndarray) -> np.ndarray:
    arr = np.asarray(action, dtype=np.float32)
    d = q01.shape[0]
    arr[..., :d] = (arr[..., :d] + 1.0) / 2.0 * rng + q01
    return arr


def _register_tongbot_domain() -> None:
    from diffusers.pipelines.cosmos import pipeline_cosmos3_omni as cosmos_pipe

    cosmos_pipe._EMBODIMENT_TO_DOMAIN_ID["tongbot"] = TONGBOT_DOMAIN_ID
    cosmos_pipe._EMBODIMENT_TO_RAW_ACTION_DIM["tongbot"] = TONGBOT_DIM
    LOGGER.info("registered domain tongbot id=%s dim=%s", TONGBOT_DOMAIN_ID, TONGBOT_DIM)


def _ensure_dist() -> None:
    if dist.is_available() and dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29571")
    dist.init_process_group(backend="gloo", rank=0, world_size=1)


def overlay_sft_dcp(pipe, ckpt: str, *, use_ema: bool = True) -> None:
    model_dir = os.path.join(ckpt, "model")
    prefix = "net_ema." if use_ema else "net."
    reader = FileSystemReader(model_dir)
    meta_keys = set(reader.read_metadata().state_dict_metadata.keys())
    state: dict[str, torch.Tensor] = {}
    missing = []
    for name, tensor in list(pipe.transformer.named_parameters()) + list(
        pipe.transformer.named_buffers()
    ):
        net_key = _diffusers_to_net_key(name)
        dcp_key = prefix + net_key
        if dcp_key not in meta_keys:
            missing.append((name, dcp_key))
            continue
        state[dcp_key] = tensor.data
    if not state:
        raise SystemExit(f"DCP 和 transformer 对不上，一个 key 都没中: {ckpt}")
    LOGGER.info(
        "overlay DCP %s prefix=%s matched=%s missing=%s",
        ckpt,
        prefix.rstrip("."),
        len(state),
        len(missing),
    )
    if missing[:8]:
        LOGGER.info("overlay miss sample: %s", missing[:8])
    t0 = time.perf_counter()
    dcp.load(state_dict=state, storage_reader=reader)
    LOGGER.info("DCP overlay done in %.1fs", time.perf_counter() - t0)


def load_pretrained_pipeline(model_dir: str):
    from diffusers import Cosmos3OmniPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

    pipe = Cosmos3OmniPipeline.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        enable_safety_checker=False,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=FLOW_SHIFT
    )
    if hasattr(pipe, "disable_safety_checker"):
        pipe.disable_safety_checker()
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)
    return pipe


class Cosmos3TongPolicy:
    def __init__(
        self,
        pipe,
        instruction: str,
        *,
        domain_name: str = "tongbot",
        chunk_size: int = ACTION_HORIZON,
        steps: int = DEFAULT_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE,
        view_point: str = "concat_view",
        denorm: tuple[np.ndarray, np.ndarray] | None = None,
        json_prompt: bool = True,
        concat_cameras: bool = True,
        decode_video: bool = False,
        latent_dumper: VisionLatentDumper | None = None,
    ) -> None:
        self.pipe = pipe
        self.instruction = instruction
        self.domain_name = domain_name
        self.chunk_size = int(chunk_size)
        self.steps = int(steps)
        self.guidance_scale = float(guidance_scale)
        self.view_point = view_point
        self.denorm = denorm
        self.json_prompt = json_prompt
        self.concat_cameras = concat_cameras
        self.decode_video = bool(decode_video)
        self.latent_dumper = latent_dumper
        self._generator = torch.Generator(device="cuda")

    def reset(self, prompt: str | None) -> None:
        if prompt:
            self.instruction = str(prompt).strip() or self.instruction
        LOGGER.info("reset prompt=%s domain=%s", self.instruction[:160], self.domain_name)
        if self.latent_dumper is not None:
            self.latent_dumper.new_session(self.instruction)

    def _prompt_text(self) -> str:
        if self.json_prompt:
            return _json_prompt(self.instruction)
        return self.instruction

    def _obs_image(self, last: dict) -> Image.Image:
        if CAM_FRONT not in last:
            raise KeyError(f"obs 缺 {CAM_FRONT}；keys={list(last)}")
        front = _to_hwc_uint8(last[CAM_FRONT])
        if not self.concat_cameras:
            return Image.fromarray(front)
        left = _to_hwc_uint8(last[CAM_LEFT]) if CAM_LEFT in last else None
        right = _to_hwc_uint8(last[CAM_RIGHT]) if CAM_RIGHT in last else None
        return _concat_view(front, left, right)

    def infer(self, msg: dict) -> dict:
        if msg.get("reset"):
            prompt = msg.get("prompt")
            if isinstance(prompt, (list, tuple)) and prompt:
                prompt = prompt[0]
            steps = msg.get("num_inference_steps")
            if steps is not None:
                self.steps = max(1, int(steps))
                LOGGER.info("denoise steps → %s (client override)", self.steps)
            self.reset(prompt if isinstance(prompt, str) else None)
            return {}
        if msg.get("compute_kv_cache"):
            return {}

        obs_list = msg.get("obs")
        if not obs_list:
            raise KeyError("infer 消息缺少 obs")
        image = self._obs_image(obs_list[-1])

        from diffusers import CosmosActionCondition

        t0 = time.perf_counter()
        self._generator.manual_seed(0)
        with torch.inference_mode():
            result = self.pipe(
                prompt=self._prompt_text(),
                action=CosmosActionCondition(
                    mode="policy",
                    chunk_size=self.chunk_size,
                    domain_name=self.domain_name,
                    resolution_tier=256,
                    image=image,
                    view_point=self.view_point,
                ),
                fps=15.0,
                num_inference_steps=self.steps,
                guidance_scale=self.guidance_scale,
                enable_sound=False,
                enable_safety_check=False,
                output_type="pil" if self.decode_video else "latent",
                generator=self._generator,
            )
        raw = None
        if hasattr(result, "action") and result.action:
            raw = result.action[0]
        elif isinstance(result, dict) and result.get("action"):
            raw = result["action"][0]
        if raw is None:
            raise RuntimeError("Cosmos3 policy 没有返回 action")
        raw_np = _as_numpy_f32(raw)
        actions = _to_tongbot_16d(raw_np)
        if self.denorm is not None:
            actions = _denorm_quantile(actions, self.denorm[0], self.denorm[1])
        infer_sec = time.perf_counter() - t0
        video = getattr(result, "video", None)
        if self.latent_dumper is not None and isinstance(video, torch.Tensor):
            self.latent_dumper.dump(
                video,
                prompt=self.instruction,
                action=actions,
                cond_image=image,
                infer_sec=infer_sec,
                extra={
                    "num_inference_steps": self.steps,
                    "guidance_scale": self.guidance_scale,
                    "output_type": "latent",
                    "decode_video": False,
                },
            )
        LOGGER.info(
            "infer action=%s raw=%s sec=%.2f steps=%s decode_video=%s dump_latent=%s domain=%s prompt=%s",
            tuple(actions.shape),
            tuple(raw_np.shape),
            infer_sec,
            self.steps,
            self.decode_video,
            self.latent_dumper is not None and isinstance(video, torch.Tensor),
            self.domain_name,
            self.instruction[:80],
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--vae", default="")
    p.add_argument("--action-stats", default=DEFAULT_STATS)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=1236)
    p.add_argument("--instruction", default=DEFAULT_PROMPT)
    p.add_argument("--domain-name", default="tongbot")
    p.add_argument("--chunk-size", type=int, default=ACTION_HORIZON)
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p.add_argument("--guidance-scale", type=float, default=DEFAULT_GUIDANCE)
    p.add_argument(
        "--decode-video",
        action="store_true",
        help="解码预测视频（真机只要 action，默认跳过 VAE decode，约省 0.6s/包）",
    )
    p.add_argument(
        "--dump-latent",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("COSMOS3_DUMP_LATENT", "1") not in ("0", "false", "False"),
        help="每包保存预测视频 latent，供后期 offline decode（默认开，不解 RGB）",
    )
    p.add_argument(
        "--latent-dir",
        default=os.environ.get("COSMOS3_LATENT_DIR", ""),
        help="vision latent 根目录，默认 <ckpt>/vision_latents",
    )
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--pretrained-only", action="store_true")
    args = p.parse_args()

    model = os.path.expanduser(args.model)
    if not os.path.isdir(model):
        raise SystemExit(f"模型目录不存在: {model}")

    _ensure_dist()
    LOGGER.info("loading Cosmos3-Nano %s", model)
    pipe = load_pretrained_pipeline(model)
    extra_meta: dict = {}
    denorm = None
    json_prompt = False
    concat_cameras = False
    domain = "robomind-franka-dual"
    ckpt_meta = model

    if not args.pretrained_only:
        ckpt = os.path.expanduser(args.ckpt)
        if not os.path.isdir(os.path.join(ckpt, "model")):
            raise SystemExit(f"SFT DCP 缺 model/: {ckpt}")
        _register_tongbot_domain()
        overlay_sft_dcp(pipe, ckpt, use_ema=True)
        denorm = _load_quantile_stats(os.path.expanduser(args.action_stats))
        json_prompt = True
        concat_cameras = True
        domain = args.domain_name
        ckpt_meta = ckpt
        extra_meta = {
            "sft": True,
            "action_normalization": "quantile",
            "view": "concat_view",
            "dcp_prefix": "net_ema",
        }
    else:
        extra_meta = {
            "pretrained_warning": (
                "Cosmos3-Nano 预训练域不是 TongBot 16D 关节；"
                "当前把模型输出 pad/truncate 到 16D。"
            )
        }

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    latent_dir = args.latent_dir or os.path.join(str(ckpt_meta), "vision_latents")
    latent_dumper = None
    if args.dump_latent and not args.decode_video:
        latent_dumper = VisionLatentDumper(
            latent_dir,
            meta={
                "ckpt": str(ckpt_meta),
                "num_inference_steps": args.steps,
                "guidance_scale": args.guidance_scale,
                "scheduler": "UniPCMultistep",
                "flow_shift": FLOW_SHIFT,
                "output_type": "latent",
                "decode_pred_video": False,
            },
        )
        LOGGER.info("dump vision latent every chunk → %s", latent_dir)
    elif args.decode_video:
        LOGGER.info("decode_video=True，不另存 latent（result.video 已是 RGB）")

    policy = Cosmos3TongPolicy(
        pipe,
        args.instruction,
        domain_name=domain,
        chunk_size=args.chunk_size,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        denorm=denorm,
        json_prompt=json_prompt,
        concat_cameras=concat_cameras,
        decode_video=args.decode_video,
        latent_dumper=latent_dumper,
    )
    LOGGER.info(
        "model ready, serving on %s:%s ckpt=%s steps=%s guidance=%s decode_video=%s dump_latent=%s",
        args.host,
        args.port,
        ckpt_meta,
        args.steps,
        args.guidance_scale,
        args.decode_video,
        latent_dumper is not None,
    )
    va_infer = {
        "source": "live_server",
        "pretrained_model_path": model,
        "param_dtype": "bfloat16",
        "num_inference_steps": args.steps,
        "action_num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "action_guidance_scale": args.guidance_scale,
        "decode_pred_video": bool(args.decode_video),
        "output_type": "pil" if args.decode_video else "latent",
        "dump_vision_latent": latent_dumper is not None,
        "vision_latent_dir": latent_dir if latent_dumper is not None else None,
        "offline_decode": (
            "python /home/xk/VLA/cosmos3_staging/offline_decode_latents.py --session <session_dir>"
        ),
        "scheduler": "UniPCMultistep",
        "flow_shift": FLOW_SHIFT,
        "snr_shift": FLOW_SHIFT,
        "action_snr_shift": FLOW_SHIFT,
        "height": IMAGE_SIZE,
        "width": IMAGE_SIZE,
        "action_dim": TONGBOT_DIM,
        "action_per_frame": args.chunk_size,
        "action_norm_method": "quantile" if denorm is not None else None,
        "env_type": domain,
    }
    WebsocketPolicyServer(
        policy,
        host=args.host,
        port=args.port,
        metadata={
            "model": "cosmos3_nano",
            "ckpt": ckpt_meta,
            "action_dim": TONGBOT_DIM,
            "action_horizon": args.chunk_size,
            "domain_name": domain,
            "va_infer": va_infer,
            **extra_meta,
            "rtc_capable": False,
        },
    ).serve_forever()


if __name__ == "__main__":
    main()
