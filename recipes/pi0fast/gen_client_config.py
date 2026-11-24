#!/usr/bin/env python3
"""从 PI05 websocket 客户端 YAML 派生 PI0-FAST 客户端 YAML。

改动点:
  - policy.lingbot_port: 1243（FAST 服务端口）
  - policy.lingbot_expected_model: pi0_fast
  - policy.ckpt_dir / openpi_config_name: 指向 PI0-FAST run
  - policy.openpi_action_horizon: 15
  - policy.n_action_steps_override: 15（FAST chunk 即 15，可整段执行）
  - policy.rtc_enabled: false（FAST 无 RTC）
  - experiment_name 前缀 pi0fast

用法: python3 gen_client_config.py [--src ...00092_ws_client.yaml] [--out ...pi0fast_ws_client.yaml]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SRC = (
    REPO_ROOT / "infer/infer/config/lerobot/pi05_openpi_tongbot_16d_00092_ws_client.yaml"
)
DEFAULT_OUT = (
    REPO_ROOT / "infer/infer/config/lerobot/pi0fast_openpi_tongbot_16d_00036_ws_client.yaml"
)

FAST_CKPT_ROOT = (
    "/home/xk/VLA/openpi/cigai-agilex-train-openpi/checkpoints/"
    "00036_pi0fast_bs16_lr2.5e-5_ah15_1gpu"
)
DEFAULT_PROMPT = (
    "With both hands, pick up the yellow banana and the yellow lemon and put them "
    "into the black box; then take the green vegetables and place them into the blue box."
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=str(DEFAULT_SRC))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    args = p.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    if not src.is_file():
        raise SystemExit(f"错误: 源配置不存在: {src}")

    cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
    pol = cfg.get("policy") or {}

    cfg["experiment_name"] = "pi0fast_openpi_tongbot_16d_00036_ws"
    pol["type"] = "pi05"  # websocket 通道复用（客户端不加载模型，仅转发 obs）
    pol["pi05_backend"] = "websocket"
    pol["lingbot_port"] = 1243
    pol["lingbot_expected_model"] = "pi0_fast"
    pol["ckpt_dir"] = FAST_CKPT_ROOT
    pol["openpi_config_name"] = "pi0_fast_tongbot"
    pol["openpi_action_dim"] = 16
    pol["openpi_action_horizon"] = 15
    pol["openpi_adapt_to_pi"] = False
    pol["openpi_delta_action_dims"] = [7, 7, -1, -1]
    pol["task_prompt"] = DEFAULT_PROMPT
    pol["n_action_steps_override"] = 15
    pol["use_action_chunk"] = True
    pol["rtc_enabled"] = False
    pol.pop("rtc_config", None)
    cfg["policy"] = pol

    if (cfg.get("batch_infer") or {}).get("action_steps") is not None:
        cfg["batch_infer"]["action_steps"] = [15]

    header = (
        "# =============================================================================\n"
        "# 本机 gpu-master OpenPI π0-FAST × 00036 websocket 客户端（连 127.0.0.1:1243）\n"
        "# 由 recipes/pi0fast/gen_client_config.py 从 PI05 00092 客户端配置派生。\n"
        "#\n"
        "#   1) bash recipes/pi0fast/launch_pi0fast_server.sh\n"
        "#   2) 日志出现 listening 后:\n"
        "#      bash recipes/pi0fast/launch_pi0fast_client.sh\n"
        "#\n"
        "# FAST: chunk=15（训练 action_horizon），无 RTC；客户端复用 pi05 websocket 通道。\n"
        "# =============================================================================\n"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(header + yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"[ok] 已生成 {out}")


if __name__ == "__main__":
    main()
