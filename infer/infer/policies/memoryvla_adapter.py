# MemoryVLA 适配器。YAML policy.type: memoryvla
# 经 websocket 调 serve_tong_memoryvla.py（默认端口 1238）。
# 协议与 LingBot-VA / DreamZero / Cosmos3 相同。

from __future__ import annotations

from infer.core.types import PolicyType
from infer.policies import register_policy
from infer.policies.lingbot_va_adapter import LingbotVaAdapter


@register_policy(PolicyType.MEMORYVLA)
class MemoryVlaAdapter(LingbotVaAdapter):
    name = "memoryvla"
    _display_name = "MemoryVLA"
    _backend_key = "memoryvla_websocket"
    _default_port = 1238
    _default_horizon = 16
    _pin_exec_horizon_always = True
    _default_expected_model = "memoryvla"
    _wrong_server_hint = "MemoryVLA 用端口 1238，InternVLA 用 1237，Cosmos3 用 1236，DreamZero 用 1235，LingBot-VA 用 1234。"
