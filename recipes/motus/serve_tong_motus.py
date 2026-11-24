#!/usr/bin/env python3
"""Motus TongBot websocket server, VA-client compatible.

Protocol (same as LingBot-VA / Cosmos3 / WALL-OSS / FastWAM):
  handshake metadata  {model: motus, ...}
  reset:              {reset: True, prompt: [str]}
  compute_kv_cache:   ignored
  infer:              {obs: [cam_dict, ...], state: (16,)}
  reply:              {action: (48, 16)}

"dream video" 落盘（--dump-video <dir>）：
  inference_step 的 video 分支去噪本身就在 latent space，落盘时永远保持
  skip_video_decode=True（真机控制环零额外开销，不再有数百 ms 的同步 VAE decode），
  只把去噪得到的原始 video_latent（含 index 0 的 condition 帧，teacher forcing）
  异步落盘到 <dir>/session_YYYYMMDD-HHMMSS/chunk_XXXXXX.pt + chunk_XXXXXX_cond.jpg，
  另写 index.jsonl。事后离线解码成 mp4：
    python recipes/motus/offline_decode_latents.py --session <dir>/session_*

与训练的差异对齐（jiangsuan-4 /nvme2n1/motus 旧 ckpt vs 之后的三路训练）：
  - TongBot 数据是 front/left/right 三路。旧 js4 loader 不认这套键名，fallback
    成 front 单目（Image source: single_view）。那批 ckpt（含 step_9000）推理
    必须 --cam-mode front。
  - 之后训练：loader 把 front/left/right 当 three_cam，T 形拼图后再
    resize_with_padding → 384×320。对应 ckpt 推理必须 --cam-mode three。
  - state/action 16D 用 stat.json 的 tongbot 键做 min-max：norm=(x-min)/(max-min)。
  - 输出 chunk=48（num_video_frames 8 × video_action_freq_ratio 6），已反归一化。
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
from pathlib import Path

import msgpack
import numpy as np
import torch
from PIL import Image
import websockets
import websockets.asyncio.server as _server
import websockets.frames
from datetime import datetime, timezone, timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("motus_serve")

CAM_FRONT = "observation.images.front"
CAM_LEFT = "observation.images.left"
CAM_RIGHT = "observation.images.right"
TONGBOT_DIM = 16

MOTUS_ROOT = Path(os.environ.get("MOTUS_ROOT", "/home/xk/VLA/Motus"))
MOTUS_INF = MOTUS_ROOT / "inference" / "real_world" / "Motus"
for _p in (str(MOTUS_ROOT), str(MOTUS_INF)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DEFAULT_CONFIG = str(MOTUS_INF / "utils" / "tongbot.yaml")
DEFAULT_CKPT = "/home/xk/VLA/h100_ckpts/motus_tongbot_jiangsuan4/checkpoint_step_9000"
DEFAULT_STAT = str(MOTUS_ROOT / "data" / "utils" / "stat.json")
DEFAULT_PROMPT = "Use both arms to complete the pick and place task."

TZ = timezone(timedelta(hours=8))


def _now_iso() -> str:
    return datetime.now(TZ).isoformat()


def _now_tag() -> str:
    return datetime.now(TZ).strftime("%Y%m%d-%H%M%S")


def _jpeg_bytes(image: "Image.Image", *, quality: int) -> bytes:
    import io

    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# msgpack helpers (same as other serve scripts)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Motus model wrapper
# ---------------------------------------------------------------------------

class MotusServeModel:
    """加载 Motus（tongbot.yaml + ckpt），T5/VLM processor 常驻。"""

    def __init__(self, args):
        import yaml
        from transformers import AutoProcessor

        from models.motus import Motus, MotusConfig  # MOTUS_INF 已在 sys.path

        self.device = torch.device(args.device)
        self.denoise = args.denoise
        self.cam_mode = str(getattr(args, "cam_mode", "front")).strip().lower()
        if self.cam_mode not in ("front", "three"):
            raise ValueError(f"cam_mode 必须是 front|three，收到 {self.cam_mode!r}")
        LOGGER.info("cam_mode=%s", self.cam_mode)

        with open(args.model_config, "r") as f:
            cfg = yaml.safe_load(f)
        common = cfg["common"]
        model_cfg = cfg["model"]
        self.video_h = int(common["video_height"])
        self.video_w = int(common["video_width"])
        self.state_dim = int(common["state_dim"])
        self.action_dim = int(common["action_dim"])

        LOGGER.info("building Motus (config=%s)", args.model_config)
        mc = MotusConfig(
            wan_checkpoint_path=model_cfg["wan"]["checkpoint_path"],
            vae_path=model_cfg["wan"]["vae_path"],
            wan_config_path=model_cfg["wan"]["config_path"],
            video_precision=model_cfg["wan"]["precision"],
            vlm_checkpoint_path=model_cfg["vlm"]["checkpoint_path"],
            und_expert_hidden_size=model_cfg.get("und_expert", {}).get("hidden_size", 512),
            und_expert_ffn_dim_multiplier=model_cfg.get("und_expert", {}).get("ffn_dim_multiplier", 4),
            und_expert_norm_eps=model_cfg.get("und_expert", {}).get("norm_eps", 1e-5),
            vlm_adapter_input_dim=model_cfg.get("und_expert", {}).get("vlm", {}).get("input_dim", 2048),
            vlm_adapter_projector_type=model_cfg.get("und_expert", {}).get("vlm", {}).get("projector_type", "mlp3x_silu"),
            num_layers=30,
            action_state_dim=self.state_dim,
            action_dim=self.action_dim,
            action_expert_dim=model_cfg["action_expert"]["hidden_size"],
            action_expert_ffn_dim_multiplier=model_cfg["action_expert"]["ffn_dim_multiplier"],
            action_expert_norm_eps=model_cfg["action_expert"].get("norm_eps", 1e-5),
            global_downsample_rate=common["global_downsample_rate"],
            video_action_freq_ratio=common["video_action_freq_ratio"],
            num_video_frames=common["num_video_frames"],
            video_height=self.video_h,
            video_width=self.video_w,
            batch_size=1,
            video_loss_weight=model_cfg["loss_weights"]["video_loss_weight"],
            action_loss_weight=model_cfg["loss_weights"]["action_loss_weight"],
            training_mode="finetune",
            load_pretrained_backbones=False,
        )
        self.model = Motus(mc).to(self.device)
        self.model.eval()
        self.model.load_checkpoint(args.ckpt, strict=False)
        LOGGER.info("checkpoint loaded: %s", args.ckpt)

        # T5 编码器常驻（reset 时重编码 prompt）
        from wan.modules.t5 import T5EncoderModel  # bak 路径由 models.motus 加入

        wan_base = Path(model_cfg["wan"]["config_path"])
        self.t5 = T5EncoderModel(
            text_len=512,
            dtype=torch.bfloat16,
            device=str(self.device),
            checkpoint_path=str(wan_base / "models_t5_umt5-xxl-enc-bf16.pth"),
            tokenizer_path=str(wan_base / "google" / "umt5-xxl"),
        )
        self.vlm_processor = AutoProcessor.from_pretrained(
            model_cfg["vlm"]["checkpoint_path"], trust_remote_code=True
        )

        # tongbot min-max 统计
        with open(args.stat_json, "r") as f:
            stat = json.load(f)["tongbot"]
        self.act_min = np.asarray(stat["min"], dtype=np.float64)
        self.act_max = np.asarray(stat["max"], dtype=np.float64)
        rng = self.act_max - self.act_min
        rng[rng == 0.0] = 1.0  # 防除零（夹爪 min=max=0 时保持原值）
        self._range = rng
        LOGGER.info("stat tongbot: dim=%d files=%s", len(self.act_min), stat.get("file_count"))

    # ---- 输入侧 ----

    def normalize_state(self, state16: np.ndarray) -> torch.Tensor:
        norm = (state16 - self.act_min) / self._range
        return torch.from_numpy(norm).unsqueeze(0).to(torch.bfloat16)

    def encode_prompt(self, prompt: str):
        out = self.t5([prompt], device=str(self.device))
        if isinstance(out, torch.Tensor):
            return [out.squeeze(0)]
        return out

    def build_vlm_inputs(self, instruction: str, image_pil):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image", "image": image_pil},
                ],
            }
        ]
        text = self.vlm_processor.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=False
        )
        encoded = self.vlm_processor(text=[text], images=[image_pil], return_tensors="pt")
        vlm_inputs = {
            "input_ids": encoded["input_ids"].to(self.device),
            "attention_mask": encoded["attention_mask"].to(self.device),
            "pixel_values": encoded["pixel_values"].to(self.device),
            "image_grid_thw": encoded.get("image_grid_thw", None),
        }
        if vlm_inputs["image_grid_thw"] is not None:
            vlm_inputs["image_grid_thw"] = vlm_inputs["image_grid_thw"].to(self.device)
        return vlm_inputs

    def preprocess_front(self, obs_last: dict) -> torch.Tensor:
        """front uint8 HWC → [1,3,384,320] float [0,1]（js4 旧 ckpt / --cam-mode front）。"""
        return self._frame_to_nchw(self._require_cam(obs_last, CAM_FRONT))

    def preprocess_obs(self, obs_last: dict) -> torch.Tensor:
        """与当前 ckpt 训练时的图像一致：front 单目，或 front/left/right T 拼图。"""
        if self.cam_mode == "front":
            return self.preprocess_front(obs_last)
        from data.utils.image_utils import resize_with_padding
        from data.lerobot.camera_views import stitch_three_views_hwc

        stitched = stitch_three_views_hwc(
            self._require_cam(obs_last, CAM_FRONT),
            self._require_cam(obs_last, CAM_LEFT),
            self._require_cam(obs_last, CAM_RIGHT),
            resize_with_padding,
        )
        return self._frame_to_nchw(stitched)

    @staticmethod
    def _require_cam(obs_last: dict, key: str) -> np.ndarray:
        if key not in obs_last:
            raise KeyError(f"obs 缺相机 {key}；keys={list(obs_last)}")
        img = np.asarray(obs_last[key])
        if img.ndim == 4:
            img = img[0]
        return np.ascontiguousarray(img, dtype=np.uint8)

    def _frame_to_nchw(self, img: np.ndarray) -> torch.Tensor:
        from data.utils.image_utils import resize_with_padding

        resized = resize_with_padding(img, (self.video_h, self.video_w))
        t = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        return t.unsqueeze(0)

    # ---- 推理 ----

    @torch.no_grad()
    def infer_actions(self, first_frame, state16, instruction, t5_embeds, image_pil):
        """永远 skip_video_decode=True：去噪本身就在 latent space，直接拿回最终
        video_latent（已在显存里，不额外花销），不在真机控制环里同步跑 VAE decode。
        需要像素帧时由离线脚本 recipes/motus/offline_decode_latents.py 解码。
        """
        vlm_inputs = self.build_vlm_inputs(instruction, image_pil)
        video_latent, actions = self.model.inference_step(
            first_frame=first_frame,
            state=state16,
            num_inference_steps=self.denoise,
            language_embeddings=t5_embeds,
            vlm_inputs=[vlm_inputs],
            skip_video_decode=True,
        )
        actions = actions.squeeze(0).float().cpu().numpy()  # (48, 16) normalized [0,1]
        return actions * self._range + self.act_min, video_latent


class VisionLatentDumper:
    """异步落盘 Motus 去噪后的原始 video_latent（不在真机控制环里解码）。

    去噪循环本身就在 latent space，latent 已在显存里、返回零额外计算。
    主线程只做一次 detach().cpu() 拷贝（小 tensor，不是像素），写盘交给单线程池。
    session 目录结构 / 字段对齐 Cosmos3 / DreamZero 的 VisionLatentDumper，
    事后用 recipes/motus/offline_decode_latents.py 离线 decode 成 dream video。
    """

    SCHEMA = "motus.vision_latent.v1"

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.session: Path | None = None
        self.idx = 0
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="latent_dump")

    def new_session(self, prompt: str = "") -> Path:
        path = self.root / f"session_{_now_tag()}"
        path.mkdir(parents=True, exist_ok=True)
        meta = {
            "schema": self.SCHEMA,
            "created_at": _now_iso(),
            "prompt": prompt,
            "model": "motus",
            "decode_hint": "python recipes/motus/offline_decode_latents.py --session " + str(path),
        }
        (path / "session.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        self.session = path
        self.idx = 0
        LOGGER.info("vision latent session %s", path)
        return path

    def dump(self, latent: torch.Tensor, *,
             prompt: str = "", cond_image: Image.Image | None = None,
             infer_sec: float | None = None) -> None:
        """latent: [B, C', T', H', W'] 去噪后的 video_latent（不是像素）。"""
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
        }
        cond_bytes = None
        if cond_image is not None:
            cond_bytes = _jpeg_bytes(cond_image, quality=80)
        pt_path = self.session / f"chunk_{idx:06d}.pt"
        jpg_path = self.session / f"chunk_{idx:06d}_cond.jpg" if cond_bytes else None
        self._pool.submit(self._write_chunk, pt_path, payload, jpg_path, cond_bytes)

    @staticmethod
    def _write_chunk(pt_path: Path, payload: dict, jpg_path: Path | None,
                     cond_bytes: bytes | None) -> None:
        try:
            torch.save(payload, pt_path)
            if jpg_path is not None and cond_bytes:
                jpg_path.write_bytes(cond_bytes)
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
            LOGGER.exception("vision latent 落盘失败 %s", pt_path)

    def close(self) -> None:
        self._pool.shutdown(wait=False)


class MotusServePolicy:
    def __init__(self, serve_model: MotusServeModel, instruction: str,
                 dump_video_root: str | None = None):
        self.m = serve_model
        self.instruction = instruction
        self._t5 = self.m.encode_prompt(instruction)
        self._dumper: VisionLatentDumper | None = None
        if dump_video_root:
            self._dumper = VisionLatentDumper(dump_video_root)
            LOGGER.info("dream video dump enabled → %s", dump_video_root)

    def reset(self, prompt: str | None) -> None:
        if prompt:
            self.instruction = str(prompt).strip() or self.instruction
            LOGGER.info("re-encode prompt: %s", self.instruction[:120])
            self._t5 = self.m.encode_prompt(self.instruction)
        if self._dumper:
            self._dumper.new_session(self.instruction)

    def infer(self, msg: dict) -> dict:
        if msg.get("reset"):
            prompt = msg.get("prompt")
            if isinstance(prompt, (list, tuple)) and prompt:
                prompt = prompt[0]
            self.reset(prompt if isinstance(prompt, str) else None)
            return {}
        if msg.get("compute_kv_cache"):
            return {}

        obs_list = msg.get("obs")
        if not obs_list:
            raise KeyError("infer 消息缺少 obs")
        last = obs_list[-1]

        m = self.m
        first_frame = m.preprocess_obs(last).to(m.device)
        state16 = m.normalize_state(_state16(msg.get("state"))).to(m.device)

        front_pil = Image.fromarray(
            (first_frame.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        )

        t0 = time.perf_counter()
        actions, video_latent = m.infer_actions(
            first_frame, state16, self.instruction, self._t5, front_pil,
        )
        infer_sec = time.perf_counter() - t0
        LOGGER.info(
            "infer action=%s denoise=%d sec=%.2f range=[%.3f,%.3f]%s",
            actions.shape, m.denoise, infer_sec,
            float(actions.min()), float(actions.max()),
            " +latent" if self._dumper else "",
        )
        # infer_actions 永远返回 video_latent（skip_video_decode=True 后不再丢），
        # 这里只是决定是否落盘，不影响真机控制环的耗时。
        if self._dumper is not None:
            self._dumper.dump(video_latent, prompt=self.instruction,
                              cond_image=front_pil, infer_sec=infer_sec)
        return {"action": np.ascontiguousarray(actions, dtype=np.float32)}


# ---------------------------------------------------------------------------
# Websocket server (identical to other serve scripts)
# ---------------------------------------------------------------------------

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
    p.add_argument("--ckpt", default=os.environ.get("MOTUS_CKPT", DEFAULT_CKPT))
    p.add_argument("--model-config", default=os.environ.get("MOTUS_MODEL_CONFIG", DEFAULT_CONFIG))
    p.add_argument("--stat-json", default=os.environ.get("MOTUS_STAT_JSON", DEFAULT_STAT))
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=int(os.environ.get("MOTUS_PORT", "1241")))
    p.add_argument("--instruction", default=os.environ.get("MOTUS_PROMPT", DEFAULT_PROMPT))
    p.add_argument("--denoise", type=int, default=int(os.environ.get("MOTUS_DENOISE", "10")))
    p.add_argument(
        "--cam-mode",
        default=os.environ.get("MOTUS_CAM_MODE", "front"),
        choices=("front", "three"),
        help="front=js4 旧 ckpt 单目；three=front/left/right T 拼图（之后训练的 ckpt）",
    )
    p.add_argument("--device", default=os.environ.get("MOTUS_DEVICE", "cuda"))
    p.add_argument("--load-only", action="store_true")
    # --dump-video: 开启后每包做 VAE decode 并落盘 dream video 帧（会增加推理延迟）
    _default_dump = os.environ.get("MOTUS_DUMP_VIDEO", "")
    p.add_argument("--dump-video", default=_default_dump,
                   help="dream video 落盘目录（非空即开启，每包 VAE decode + 写帧 jpg）")
    args = p.parse_args()

    serve_model = MotusServeModel(args)
    LOGGER.info("model ready denoise=%d", args.denoise)
    if args.load_only:
        LOGGER.info("load-only OK")
        return

    dump_video_root = args.dump_video or None
    policy = MotusServePolicy(serve_model, args.instruction,
                              dump_video_root=dump_video_root)
    meta = {
        "model": "motus",
        "ckpt": str(Path(args.ckpt).resolve()),
        "action_horizon": 48,
        "denoise": args.denoise,
        "action_dim": TONGBOT_DIM,
        "cam_mode": serve_model.cam_mode,
        "rtc_capable": False,
    }
    LOGGER.info("server listening on %s:%s", args.host, args.port)
    WebsocketPolicyServer(policy, args.host, args.port, metadata=meta).serve_forever()


if __name__ == "__main__":
    main()
