# Motus 适配器。YAML policy.type: motus
# 经 websocket 调 recipes/motus/serve_tong_motus.py（默认端口 1241）。
# 协议与 LingBot-VA / FastWAM 相同。
# action_horizon=48（8 帧 × 6），denoise=10，front 单目 384×320（训练即单目）。

from __future__ import annotations

from infer.core.types import PolicyType
from infer.policies import register_policy
from infer.policies.lingbot_va_adapter import LingbotVaAdapter


@register_policy(PolicyType.MOTUS)
class MotusAdapter(LingbotVaAdapter):
    name = "motus"
    _display_name = "Motus"
    _backend_key = "motus_websocket"
    _default_port = 1241
    _default_horizon = 48
    _pin_exec_horizon_always = True
    _default_expected_model = "motus"
    # Motus tongbot 训练数据 15fps、global_downsample_rate=1
    _default_obs_fps = 15.0
    _wrong_server_hint = "Motus 用端口 1241，FastWAM 用 1240，WALL-OSS 用 1239。"
