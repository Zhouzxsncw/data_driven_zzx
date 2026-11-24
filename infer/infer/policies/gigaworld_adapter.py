# GigaWorld-Policy-0.5 适配器。YAML policy.type: gigaworld / gwp / gwp05
# 经 websocket 调 recipes/gigaworld_policy/serve_tong_gwp.py（默认端口 1243）。
# 协议与 LingBot-VA / FastWAM / Motus 相同。
# action_horizon=48，denoise=10，三相机 T 形 320×384。

from __future__ import annotations

from infer.core.types import PolicyType
from infer.policies import register_policy
from infer.policies.lingbot_va_adapter import LingbotVaAdapter


@register_policy(PolicyType.GIGAWORLD)
class GigaWorldAdapter(LingbotVaAdapter):
    name = "gigaworld"
    _display_name = "GigaWorld-Policy-0.5"
    _backend_key = "gigaworld_websocket"
    _default_port = 1243
    _default_horizon = 48
    _pin_exec_horizon_always = True
    _default_expected_model = "gwp05"
    _default_obs_fps = 15.0
    _wrong_server_hint = "GWP 用端口 1243，Motus 用 1241，FastWAM 用 1240。"
