# FastWAM 适配器。YAML policy.type: fastwam
# 经 websocket 调 serve_tong_fastwam.py（默认端口 1240）。
# 协议与 LingBot-VA / Cosmos3 相同。
# action_horizon=32，denoise=10，三相机 robotwin 拼接。

from __future__ import annotations

from infer.core.types import PolicyType
from infer.policies import register_policy
from infer.policies.lingbot_va_adapter import LingbotVaAdapter


@register_policy(PolicyType.FASTWAM)
class FastWAMAdapter(LingbotVaAdapter):
    name = "fastwam"
    _display_name = "FastWAM"
    _backend_key = "fastwam_websocket"
    _default_port = 1240
    _default_horizon = 32
    _pin_exec_horizon_always = True
    _default_expected_model = "fastwam"
    _default_obs_fps = 5.0
    _wrong_server_hint = "FastWAM 用端口 1240，WALL-OSS 用 1239，MemoryVLA 用 1238。"
