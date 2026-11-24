# InternVLA-A1.5 适配器。YAML policy.type: internvla
# 经 websocket 调 serve_tong_internvla.py（默认端口 1237）。
# 协议与 LingBot-VA / DreamZero / Cosmos3 相同。

from __future__ import annotations

from infer.core.types import PolicyType
from infer.policies import register_policy
from infer.policies.lingbot_va_adapter import LingbotVaAdapter


@register_policy(PolicyType.INTERNVLA)
class InternVlaAdapter(LingbotVaAdapter):
    name = "internvla"
    _display_name = "InternVLA"
    _backend_key = "internvla_websocket"
    _default_port = 1237
    _default_horizon = 50
    _pin_exec_horizon_always = False
    _default_expected_model = "internvla_a15"
    _wrong_server_hint = "InternVLA 用端口 1237，MemoryVLA 用 1238，Cosmos3 用 1236，DreamZero 用 1235，LingBot-VA 用 1234。"
