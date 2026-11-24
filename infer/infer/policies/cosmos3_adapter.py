# Cosmos3 适配器。YAML policy.type: cosmos3
# 经 websocket 调 serve_tong_cosmos3.py（默认端口 1236）。
# 协议与 LingBot-VA / DreamZero 相同。

from __future__ import annotations

from infer.core.types import PolicyType
from infer.policies import register_policy
from infer.policies.lingbot_va_adapter import LingbotVaAdapter


@register_policy(PolicyType.COSMOS3)
class Cosmos3Adapter(LingbotVaAdapter):
    name = "cosmos3"
    _display_name = "Cosmos3"
    _backend_key = "cosmos3_websocket"
    _default_port = 1236
    _default_horizon = 16
    _pin_exec_horizon_always = True
    _default_expected_model = "cosmos3_nano"
    _default_obs_fps = 15.0
    _wrong_server_hint = "Cosmos3 用端口 1236，DreamZero 用 1235，LingBot-VA 用 1234。"
