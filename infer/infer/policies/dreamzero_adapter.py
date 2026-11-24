# DreamZero 5B 适配器。YAML policy.type: dreamzero
# 经 websocket 调 serve_tong_dreamzero.py（默认端口 1235）。
# 协议与 LingBot-VA 相同，但模型、端口、action_horizon=16 独立。

from __future__ import annotations

from infer.core.types import PolicyType
from infer.policies import register_policy
from infer.policies.lingbot_va_adapter import LingbotVaAdapter


@register_policy(PolicyType.DREAMZERO)
class DreamZeroAdapter(LingbotVaAdapter):
    name = "dreamzero"
    _display_name = "DreamZero"
    _backend_key = "dreamzero_websocket"
    _default_port = 1235
    _default_horizon = 16
    _pin_exec_horizon_always = True
    _default_expected_model = "dreamzero_5b"
    _wrong_server_hint = "DreamZero 用端口 1235，LingBot-VA 用 1234。"
