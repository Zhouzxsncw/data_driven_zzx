#!/usr/bin/env python3
"""
LingBot-VA 独立推理脚本
=======================
新建的推理代码，参考 infer/config/lingbot 配置格式。
不修改 LingBot-VA 原有推理代码，而是作为上层封装使用。

功能：
  1. 单 GPU 启动 LingBot-VA 推理服务器 (websocket 模式)
  2. 客户端连接测试
  3. 使用示例图片进行独立推理测试

用法：
  # 启动推理服务器 (单 GPU)
  python run_lingbotva_infer.py --config config.yaml --mode server

  # 运行客户端测试
  python run_lingbotva_infer.py --config config.yaml --mode client

  # 运行独立推理测试 (不需要客户端)
  python run_lingbotva_infer.py --config config.yaml --mode test
"""

import argparse
import os
import shutil
import sys
import time
import json
import yaml
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from PIL import Image

import torch
import torch.nn.functional as F


def _enable_gpu_fast_path() -> None:
    """本机 RTX PRO 6000 96GB：TF32 + Flash/cuDNN SDPA，Transformer/T5/VAE 全部留 GPU。"""
    if not torch.cuda.is_available():
        return
    torch.cuda.set_device(0)
    try:
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        torch.backends.cudnn.conv.fp32_precision = "tf32"
    except Exception:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    # Transformer 的 custom_sdpa 会在调用时强制 Flash/cuDNN。
    # 这里只把快路径打开；math 留给 T5/VAE 作兜底，避免小形状 SDPA 直接报错。
    torch.backends.cuda.enable_flash_sdp(True)
    try:
        torch.backends.cuda.enable_cudnn_sdp(True)
    except Exception:
        pass


def _module_device(module) -> str:
    try:
        return str(next(module.parameters()).device)
    except StopIteration:
        return "empty"
    except Exception as exc:
        return f"unknown({exc})"


def _log_gpu_placement(model, logger) -> None:
    names = ("transformer", "text_encoder", "vae")
    lines = []
    for name in names:
        mod = getattr(model, name, None)
        if mod is None:
            lines.append(f"{name}=missing")
            continue
        lines.append(f"{name}={_module_device(mod)}")
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    props = torch.cuda.get_device_properties(0)
    logger.info(
        "GPU placement: %s | allocated=%.1fGiB reserved=%.1fGiB | %s (%.0fGiB)",
        " ".join(lines),
        allocated,
        reserved,
        props.name,
        props.total_memory / (1024 ** 3),
    )
    sdp = []
    for name, fn in (
        ("flash", getattr(torch.backends.cuda, "flash_sdp_enabled", None)),
        ("cudnn", getattr(torch.backends.cuda, "cudnn_sdp_enabled", None)),
        ("mem_efficient", getattr(torch.backends.cuda, "mem_efficient_sdp_enabled", None)),
        ("math", getattr(torch.backends.cuda, "math_sdp_enabled", None)),
    ):
        if fn is None:
            continue
        try:
            sdp.append(f"{name}={fn()}")
        except Exception:
            sdp.append(f"{name}=?")
    if sdp:
        logger.info("SDPA backends: %s (attn_mode=torch custom_sdpa)", " ".join(sdp))
    off_t5 = bool(getattr(model, "enable_offload", True))
    off_vae = bool(getattr(model, "offload_vae", off_t5))
    if off_t5 or off_vae:
        logger.warning("仍有模块 offload 到 CPU: T5=%s VAE=%s（本机 96GB 应全关）", off_t5, off_vae)
    else:
        logger.info("全部权重在 GPU：Transformer + T5 + VAE，无 CPU offload")

# 训练 ckpt 通常只有 transformer/。推理还需要基座的 vae/tokenizer/text_encoder，
# 并且必须把 attn_mode 从训练用的 flex 改成 torch。
_COMPONENT_DIRS = ("vae", "tokenizer", "text_encoder")
_VLA_ROOT = Path("/home/xk/VLA")
_DEFAULT_BASE_CANDIDATES = (
    Path("/home/xk/VLA/lingbot-va-pretrained/lingbot-va-base"),
    Path("/home/xk/VLA/lingbot-va-pretrained/00036_step2000"),
    Path("/home/cigai22/lingbot-va-pretrained/00036_step2000"),
    Path("/home/cigai22/lingbot-va-pretrained/lingbot-va-base"),
)
_RESOLVED_ROOT = Path(
    os.environ.get(
        "LINGBOT_VA_RESOLVED_ROOT",
        "/home/xk/VLA/lingbot-va-pretrained/_resolved",
    )
)


def _lingbot_va_code_root() -> Path:
    env = os.environ.get("LINGBOT_VA_CODE")
    if env:
        p = Path(env).expanduser().resolve()
        if p.exists():
            return p
    here = Path(__file__).resolve().parent
    for cand in (
        here / "lingbot-va",
        here.parent / "lingbot-va",
        _VLA_ROOT / "lingbot-va",
        Path("/home/xk/VLA/lingbot-va"),
        Path("/home/cigai22/lingbot-va"),
    ):
        if cand.exists():
            return cand.resolve()
    raise FileNotFoundError("找不到 lingbot-va 代码目录，请设置 LINGBOT_VA_CODE")


def _has_transformer_weights(d: Path) -> bool:
    if not (d / "config.json").is_file():
        return False
    if any(
        (d / name).exists()
        for name in (
            "diffusion_pytorch_model.safetensors",
            "diffusion_pytorch_model.bin",
            "pytorch_model.bin",
        )
    ):
        return True
    return any(d.glob("*.safetensors"))


def _find_transformer_dir(raw: Path) -> Path:
    p = raw.expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"模型路径不存在: {p}")
    if p.name == "transformer" and _has_transformer_weights(p):
        return p
    if _has_transformer_weights(p / "transformer"):
        return p / "transformer"
    if _has_transformer_weights(p):
        return p
    raise FileNotFoundError(
        f"在 {p} 找不到 transformer 权重 (需要 config.json + *.safetensors)"
    )


def _find_base_model(explicit: Optional[str]) -> Path:
    cands = []
    if explicit:
        cands.append(Path(explicit).expanduser())
    cands.extend(_DEFAULT_BASE_CANDIDATES)
    for c in cands:
        if all((c / name).exists() for name in _COMPONENT_DIRS):
            return c.resolve()
    raise FileNotFoundError(
        "找不到完整基座模型 (vae/tokenizer/text_encoder)。"
        f" 试过: {[str(c) for c in cands]}"
    )


def _is_under_removable_media(path: Path) -> bool:
    parts = path.resolve().parts
    return len(parts) >= 2 and parts[1] == "media"


def _link_or_copy(src: Path, dst: Path, copy_file: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_file and src.is_file():
        print(f"  拷贝权重到本地 SSD: {src} -> {dst} ({src.stat().st_size / 1e9:.1f} GB)")
        shutil.copy2(src, dst)
    else:
        os.symlink(src, dst)


def _reuse_complete_pack(transformer_dir: Path) -> Optional[str]:
    """本地已有完整模型根且 attn_mode=torch 时直接用，避免再拼目录、再加载。"""
    root = transformer_dir.parent if transformer_dir.name == "transformer" else transformer_dir
    if not all((root / name).exists() for name in _COMPONENT_DIRS):
        return None
    if not (root / "transformer").exists():
        return None
    try:
        cfg = json.loads((transformer_dir / "config.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if cfg.get("attn_mode") != "torch":
        return None
    if _is_under_removable_media(root):
        return None
    print(f"使用已有完整模型目录: {root}")
    return str(root)


def resolve_pretrained_model_path(
    raw_path: str,
    base_model_path: Optional[str] = None,
) -> str:
    """允许 yaml 指向完整模型根、checkpoint 目录、或只有 transformer 的子目录。

    拼出可加载的根目录:
      {resolved}/vae, tokenizer, text_encoder  ← 基座 symlink
      {resolved}/transformer/config.json       ← attn_mode 改成 torch
      {resolved}/transformer/*.safetensors     ← 训练权重；若在 /media 则拷到 SSD
    """
    transformer_dir = _find_transformer_dir(Path(raw_path))
    reused = _reuse_complete_pack(transformer_dir)
    if reused is not None:
        return reused

    ckpt_name = transformer_dir.parent.name
    if ckpt_name in ("", ".", "/"):
        ckpt_name = transformer_dir.name
    resolved = _RESOLVED_ROOT / ckpt_name
    resolved.mkdir(parents=True, exist_ok=True)
    tf_dst = resolved / "transformer"
    tf_dst.mkdir(parents=True, exist_ok=True)

    same_dir_complete = all(
        (transformer_dir.parent / name).exists() for name in _COMPONENT_DIRS
    )
    if same_dir_complete:
        base = transformer_dir.parent
    else:
        base = _find_base_model(base_model_path)

    for name in _COMPONENT_DIRS:
        src = (base / name).resolve()
        dst = resolved / name
        if dst.exists() or dst.is_symlink():
            if dst.resolve() == src:
                continue
            dst.unlink()
        os.symlink(src, dst)

    assets_src = base / "assets"
    if assets_src.exists():
        assets_dst = resolved / "assets"
        if assets_dst.exists() or assets_dst.is_symlink():
            if not (assets_dst.exists() and assets_dst.resolve() == assets_src.resolve()):
                assets_dst.unlink()
                os.symlink(assets_src.resolve(), assets_dst)
        else:
            os.symlink(assets_src.resolve(), assets_dst)

    cfg = json.loads((transformer_dir / "config.json").read_text())
    cfg["attn_mode"] = "torch"
    (tf_dst / "config.json").write_text(json.dumps(cfg, indent=2))

    weight_src = None
    for name in (
        "diffusion_pytorch_model.safetensors",
        "diffusion_pytorch_model.bin",
        "pytorch_model.bin",
    ):
        cand = transformer_dir / name
        if cand.exists():
            weight_src = cand
            break
    if weight_src is None:
        safes = sorted(transformer_dir.glob("*.safetensors"))
        if not safes:
            raise FileNotFoundError(f"{transformer_dir} 里没有权重文件")
        weight_src = safes[0]
    weight_src = weight_src.resolve()
    weight_dst = tf_dst / weight_src.name
    # USB 上的 9.5G 权重默认只做 symlink；拷到 SSD 设 LINGBOT_VA_COPY_USB_WEIGHTS=1
    copy_weights = (
        _is_under_removable_media(weight_src)
        and os.environ.get("LINGBOT_VA_COPY_USB_WEIGHTS", "") == "1"
    )
    need_refresh = True
    if weight_dst.exists() and not weight_dst.is_symlink():
        if weight_dst.stat().st_size == weight_src.stat().st_size:
            need_refresh = False
    if need_refresh:
        _link_or_copy(weight_src, weight_dst, copy_file=copy_weights)

    print(f"模型根目录: {resolved}")
    print(f"  transformer: {weight_dst} -> {weight_src}")
    print(f"  基座组件: {base}")
    print(f"  attn_mode: torch (原 ckpt 可能是 flex)")
    return str(resolved)


# ============================================================
# 配置加载 (参考 infer/config 格式)
# ============================================================

def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def _apply_resolved_model_path(cfg: dict) -> dict:
    raw = cfg['model']['pretrained_model_path']
    base = cfg['model'].get('base_model_path')
    cfg['model']['pretrained_model_path'] = resolve_pretrained_model_path(raw, base)
    return cfg


def config_to_va_config(cfg: dict):
    """
    将 YAML 配置转换为 wan_va_server 所需的 EasyDict 格式。
    这是关键桥梁：将新的 YAML 配置格式映射到 LingBot-VA 原始配置接口。
    """
    from easydict import EasyDict

    va_cfg = EasyDict()

    # 模型路径
    va_cfg.wan22_pretrained_model_name_or_path = cfg['model']['pretrained_model_path']

    # 精度
    dtype_map = {
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
        'float32': torch.float32,
    }
    va_cfg.param_dtype = dtype_map.get(cfg['model']['param_dtype'], torch.bfloat16)
    va_cfg.enable_offload = cfg['model'].get('enable_offload', True)
    va_cfg.offload_vae = bool(cfg['model'].get('offload_vae', va_cfg.enable_offload))

    # 服务器
    va_cfg.host = cfg['server']['host']
    va_cfg.port = cfg['server']['port']
    va_cfg.infer_mode = cfg['server']['infer_mode']

    # 架构
    va_cfg.env_type = cfg['architecture']['env_type']
    va_cfg.attn_window = cfg['architecture']['attn_window']
    va_cfg.frame_chunk_size = cfg['architecture']['frame_chunk_size']
    va_cfg.height = cfg['architecture']['height']
    va_cfg.width = cfg['architecture']['width']
    va_cfg.patch_size = tuple(cfg['architecture']['patch_size'])

    # 动作
    va_cfg.action_dim = cfg['action']['action_dim']
    va_cfg.action_per_frame = cfg['action']['action_per_frame']
    va_cfg.used_action_channel_ids = cfg['action']['used_action_channel_ids']
    va_cfg.action_norm_method = cfg['action']['action_norm_method']

    # 占位值必须是 used 通道数（14），不是 action_dim（30）。
    # preprocess_action 把 (C_used, F, H) pad 成 C_used+1 后用该索引填回 30 维；
    # 写成 30 会 IndexError，KV cache 失败后每一包都变成第一帧。
    inverse_ids = [len(va_cfg.used_action_channel_ids)] * va_cfg.action_dim
    for i, j in enumerate(va_cfg.used_action_channel_ids):
        inverse_ids[j] = i
    va_cfg.inverse_used_action_channel_ids = inverse_ids

    # 推理参数
    va_cfg.num_inference_steps = cfg['inference']['num_inference_steps']
    va_cfg.video_exec_step = cfg['inference']['video_exec_step']
    va_cfg.action_num_inference_steps = cfg['inference']['action_num_inference_steps']
    va_cfg.snr_shift = cfg['inference']['snr_shift']
    va_cfg.action_snr_shift = cfg['inference']['action_snr_shift']
    va_cfg.guidance_scale = cfg['inference']['guidance_scale']
    va_cfg.action_guidance_scale = cfg['inference']['action_guidance_scale']
    va_cfg.num_chunks_to_infer = cfg['inference']['num_chunks_to_infer']

    # 相机
    va_cfg.obs_cam_keys = cfg['camera']['obs_cam_keys']

    # 归一化统计
    va_cfg.norm_stat = cfg['norm_stat']

    # 保存路径
    va_cfg.save_root = cfg['test']['save_root']
    va_cfg.decode_pred_video = bool(cfg.get('inference', {}).get('decode_pred_video', False))
    va_cfg.decode_pred_video_max_chunks = int(cfg.get('inference', {}).get('decode_pred_video_max_chunks', 8))

    # 测试用
    va_cfg.input_img_path = cfg['test']['input_img_path']
    va_cfg.prompt = cfg['test']['prompt']

    # 分布式参数 (单 GPU 模式)
    va_cfg.rank = 0
    va_cfg.local_rank = 0
    va_cfg.world_size = 1

    return va_cfg


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def snapshot_va_infer(config_path: str, cfg: dict, va_cfg) -> dict:
    """服务端实际加载的采样/架构参数，随 websocket metadata 发给客户端写入实验产物。"""
    resolved = str(Path(config_path).expanduser().resolve())
    try:
        config_yaml = Path(resolved).read_text(encoding="utf-8")
    except OSError:
        config_yaml = yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)
    model = cfg.get("model") or {}
    return _jsonable({
        "config_path": resolved,
        "config_yaml": config_yaml,
        "pretrained_model_path": va_cfg.wan22_pretrained_model_name_or_path,
        "base_model_path": model.get("base_model_path"),
        "param_dtype": model.get("param_dtype", "bfloat16"),
        "enable_offload": bool(va_cfg.enable_offload),
        "env_type": va_cfg.env_type,
        "attn_window": va_cfg.attn_window,
        "frame_chunk_size": va_cfg.frame_chunk_size,
        "height": va_cfg.height,
        "width": va_cfg.width,
        "patch_size": list(va_cfg.patch_size),
        "action_dim": va_cfg.action_dim,
        "action_per_frame": va_cfg.action_per_frame,
        "used_action_channel_ids": list(va_cfg.used_action_channel_ids),
        "action_norm_method": va_cfg.action_norm_method,
        "num_inference_steps": va_cfg.num_inference_steps,
        "action_num_inference_steps": va_cfg.action_num_inference_steps,
        "video_exec_step": va_cfg.video_exec_step,
        "snr_shift": va_cfg.snr_shift,
        "action_snr_shift": va_cfg.action_snr_shift,
        "guidance_scale": va_cfg.guidance_scale,
        "action_guidance_scale": va_cfg.action_guidance_scale,
        "num_chunks_to_infer": va_cfg.num_chunks_to_infer,
        "decode_pred_video": bool(getattr(va_cfg, "decode_pred_video", False)),
        "obs_cam_keys": list(va_cfg.obs_cam_keys),
        "server_prompt": va_cfg.prompt,
        "source": "live_server",
        "started_at": datetime.now().astimezone().isoformat(),
    })


# ============================================================
# 服务器模式：单 GPU 启动 LingBot-VA 推理服务
# ============================================================

def run_server(config_path: str):
    """
    以单 GPU 模式启动 LingBot-VA 推理服务器。
    不需要 torch.distributed，直接在单卡上运行。
    """
    cfg = _apply_resolved_model_path(load_config(config_path))
    va_cfg = config_to_va_config(cfg)

    # 添加 lingbot-va 代码路径
    lingbot_va_root = _lingbot_va_code_root()
    sys.path.insert(0, str(lingbot_va_root))

    # 确保 wan_va 模块可导入
    wan_va_path = str(lingbot_va_root)
    if wan_va_path not in sys.path:
        sys.path.insert(0, wan_va_path)

    from wan_va.wan_va_server import VA_Server
    from wan_va.utils.Simple_Remote_Infer.deploy.websocket_policy_server import WebsocketPolicyServer
    from wan_va.utils import init_logger, logger

    init_logger()
    _enable_gpu_fast_path()

    logger.info(f"=" * 60)
    logger.info(f"LingBot-VA 推理服务器 (单 GPU 模式)")
    logger.info(f"模型路径: {va_cfg.wan22_pretrained_model_name_or_path}")
    logger.info(f"监听地址: {va_cfg.host}:{va_cfg.port}")
    logger.info(f"推理模式: {va_cfg.infer_mode}")
    logger.info(f"Offload: T5={va_cfg.enable_offload} VAE={getattr(va_cfg, 'offload_vae', va_cfg.enable_offload)}")
    logger.info(
        f"infer: video_steps={va_cfg.num_inference_steps} action_steps={va_cfg.action_num_inference_steps} "
        f"guidance={va_cfg.guidance_scale}/{va_cfg.action_guidance_scale} attn_window={va_cfg.attn_window}"
    )
    try:
        from va_debug_log import debug_log_path as _va_dbg_path
        logger.info(f"结构化 debug jsonl: {_va_dbg_path()}")
    except Exception:
        logger.info("结构化 debug jsonl: 见环境变量 LINGBOT_VA_DEBUG_LOG")
    logger.info(f"=" * 60)

    # 创建模型（T5/VAE/Transformer 按 YAML 全部放到 GPU）
    model = VA_Server(va_cfg)
    _log_gpu_placement(model, logger)

    if va_cfg.infer_mode == 'server':
        # WebSocket 服务模式
        logger.info(f"启动 WebSocket 服务器 {va_cfg.host}:{va_cfg.port}")
        va_infer = snapshot_va_infer(config_path, cfg, va_cfg)
        logger.info(
            "实验记录 snapshot: video_steps=%s action_steps=%s guidance=%s/%s "
            "attn_window=%s video_exec_step=%s snr=%s/%s ckpt=%s",
            va_infer.get("num_inference_steps"),
            va_infer.get("action_num_inference_steps"),
            va_infer.get("guidance_scale"),
            va_infer.get("action_guidance_scale"),
            va_infer.get("attn_window"),
            va_infer.get("video_exec_step"),
            va_infer.get("snr_shift"),
            va_infer.get("action_snr_shift"),
            va_infer.get("pretrained_model_path"),
        )
        model_server = WebsocketPolicyServer(
            model,
            host=va_cfg.host,
            port=va_cfg.port,
            metadata={
                'model': 'lingbot-va',
                'device': 'cuda:0',
                'va_infer': va_infer,
            },
        )
        model_server.serve_forever()
    elif va_cfg.infer_mode == 'i2va':
        # 图生视频-动作模式
        logger.info("运行 I2VA 模式 (图生视频-动作)")
        model.generate()
    else:
        raise ValueError(f"未知推理模式: {va_cfg.infer_mode}")

    logger.info("推理完成!")


# ============================================================
# 客户端模式：连接到 LingBot-VA 服务器进行测试
# ============================================================

def run_client(config_path: str):
    """
    作为客户端连接到 LingBot-VA 推理服务器并发送测试请求。
    """
    cfg = load_config(config_path)

    lingbot_va_root = _lingbot_va_code_root()
    sys.path.insert(0, str(lingbot_va_root))

    from wan_va.utils.Simple_Remote_Infer.deploy.websocket_client_policy import WebsocketClientPolicy
    from wan_va.utils.Simple_Remote_Infer.deploy.image_tools import convert_to_uint8

    host = cfg['server']['host']
    port = cfg['server']['port']
    if host == '0.0.0.0':
        host = '127.0.0.1'

    print(f"连接到 LingBot-VA 服务器 {host}:{port} ...")
    client = WebsocketClientPolicy(host=host, port=port)
    print("连接成功!")

    # 准备测试观测
    obs_cam_keys = cfg['camera']['obs_cam_keys']
    img_path = cfg['test']['input_img_path']
    prompt = cfg['test']['prompt']

    # 加载图片
    images = {}
    for cam_key in obs_cam_keys:
        img_file = os.path.join(img_path, f"{cam_key}.png")
        if os.path.exists(img_file):
            img = np.array(Image.open(img_file).convert("RGB"))
            images[cam_key] = img
            print(f"  加载图片: {cam_key} -> {img.shape}")
        else:
            # 使用随机图片作为后备
            img = np.random.randint(0, 256, size=(256, 256, 3), dtype=np.uint8)
            images[cam_key] = img
            print(f"  [警告] 未找到 {img_file}，使用随机图片")

    # 步骤 1: 重置 (发送 prompt)
    print(f"\n[1/3] 重置服务器，prompt: '{prompt}'")
    reset_obs = {
        'reset': True,
        'prompt': [prompt],
    }
    client.infer(reset_obs)
    print("  重置完成!")

    # 步骤 2: 计算 KV Cache
    print("[2/3] 计算 KV Cache...")
    state = np.random.rand(1, len(cfg['action']['used_action_channel_ids'])).astype(np.float32)
    kv_obs = {
        'obs': [images],
        'state': state,
        'compute_kv_cache': True,
    }
    client.infer(kv_obs)
    print("  KV Cache 计算完成!")

    # 步骤 3: 推理一个 chunk
    print("[3/3] 推理一个动作 chunk...")
    infer_obs = {
        'obs': [images],
        'state': state,
    }
    result = client.infer(infer_obs)
    action = result.get('action', None)
    if action is not None:
        print(f"  动作形状: {action.shape}")
        print(f"  动作值 (前5维): {action.flatten()[:5]}")
    else:
        print("  [警告] 未返回动作")

    print("\n客户端测试完成!")


# ============================================================
# 独立测试模式：直接加载模型并运行推理
# ============================================================

def run_test(config_path: str):
    """
    独立测试模式：直接加载模型进行推理，不需要 websocket。
    用于验证模型是否正确安装和加载。
    """
    cfg = _apply_resolved_model_path(load_config(config_path))
    va_cfg = config_to_va_config(cfg)

    # 添加 lingbot-va 代码路径
    lingbot_va_root = _lingbot_va_code_root()
    sys.path.insert(0, str(lingbot_va_root))

    from wan_va.wan_va_server import VA_Server
    from wan_va.utils import init_logger, logger

    init_logger()
    _enable_gpu_fast_path()

    logger.info("=" * 60)
    logger.info("LingBot-VA 独立推理测试")
    logger.info("=" * 60)

    # 创建模型
    logger.info("加载模型...")
    model = VA_Server(va_cfg)
    logger.info("模型加载完成!")

    # 准备测试数据
    obs_cam_keys = va_cfg.obs_cam_keys
    img_path = va_cfg.input_img_path

    # 加载或生成测试图片
    images = {}
    for cam_key in obs_cam_keys:
        img_file = os.path.join(img_path, f"{cam_key}.png")
        if os.path.exists(img_file):
            img = np.array(Image.open(img_file).convert("RGB"))
            images[cam_key] = img
            logger.info(f"加载图片: {cam_key} -> {img.shape}")
        else:
            img = np.random.randint(0, 256, size=(256, 256, 3), dtype=np.uint8)
            images[cam_key] = img
            logger.warning(f"未找到 {img_file}，使用随机图片")

    # 步骤 1: Reset
    logger.info(f"[1/3] Reset (prompt: '{va_cfg.prompt}')")
    reset_obs = {
        'reset': True,
        'prompt': [va_cfg.prompt],
    }
    model.infer(reset_obs)

    # 步骤 2: 直接推理 (第一帧会自动初始化 init_latent)
    logger.info("[2/3] 推理动作 (第一帧)...")
    state = np.random.rand(1, len(va_cfg.used_action_channel_ids)).astype(np.float32)
    infer_obs = {
        'obs': [images],
        'state': state,
    }
    result = model.infer(infer_obs)
    action = result.get('action', None)

    if action is not None:
        logger.info(f"动作形状: {action.shape}")
        logger.info(f"动作值 (前10维): {action.flatten()[:10]}")
        logger.info(f"动作范围: [{action.min():.4f}, {action.max():.4f}]")
    else:
        logger.warning("未返回动作!")

    # 步骤 3: 重置 VAE 缓存后再推理第二帧
    logger.info("[3/3] 重置并推理第二帧...")
    model.infer({'reset': True, 'prompt': [va_cfg.prompt]})

    infer_obs2 = {
        'obs': [images],
        'state': state,
    }
    result2 = model.infer(infer_obs2)
    action2 = result2.get('action', None)
    if action2 is not None:
        logger.info(f"第二帧动作形状: {action2.shape}")
        logger.info(f"第二帧动作值 (前10维): {action2.flatten()[:10]}")

    # 保存结果
    save_root = va_cfg.save_root
    os.makedirs(save_root, exist_ok=True)
    if action is not None:
        np.save(os.path.join(save_root, 'test_action.npy'), action)
        logger.info(f"动作已保存到: {save_root}/test_action.npy")

    logger.info("=" * 60)
    logger.info("测试完成!")
    logger.info("=" * 60)


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='LingBot-VA 推理脚本')
    parser.add_argument('--config', type=str, required=True,
                        help='YAML 配置文件路径')
    parser.add_argument('--mode', type=str, default='test',
                        choices=['server', 'client', 'test'],
                        help='运行模式: server=启动服务, client=客户端测试, test=独立测试')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"错误: 配置文件不存在: {args.config}")
        sys.exit(1)

    print(f"配置文件: {args.config}")
    print(f"运行模式: {args.mode}")
    print()

    if args.mode == 'server':
        run_server(args.config)
    elif args.mode == 'client':
        run_client(args.config)
    elif args.mode == 'test':
        run_test(args.config)


if __name__ == '__main__':
    main()
