# τ0-WM 适配器。YAML policy.type: tau0wm / tau0 / tau-0-wm
# 经 websocket 调 recipes/tau0_wm/serve_tong_tau.py（默认端口 1244）。
# GPU：16D FK → play 相对 EE → 绝对 EE20。客户端读完后同一条 ws ``solve_ik`` → 16D。
# 模型 chunk=33；菜单 action-step 范围 1~33；所选值 = IK 路点数 n_exec。
# rot6d 必须是列约定，不能用 WALL-OSS 行约定。

from __future__ import annotations

import logging

from infer.core.types import PolicyType
from infer.policies import register_policy
from infer.policies.client_ee_ik import ClientAbsEeIkMixin
from infer.policies.lingbot_va_adapter import LingbotVaAdapter

LOGGER = logging.getLogger(__name__)


@register_policy(PolicyType.TAU0WM)
class Tau0WmAdapter(ClientAbsEeIkMixin, LingbotVaAdapter):
    name = "tau0wm"
    _display_name = "τ0-WM"
    _backend_key = "tau0wm_websocket"
    _default_port = 1244
    _default_horizon = 33
    _pin_exec_horizon_always = False
    _default_expected_model = "tau0wm"
    _default_obs_fps = 15.0
    _wrong_server_hint = (
        "τ0-WM 用端口 1244，GWP 用 1243，Motus 用 1241，FastWAM 用 1240，WALL-OSS 用 1239。"
    )

    def __init__(self, cfg, device=None) -> None:
        super().__init__(cfg, device=device)
        self.action_space = "ee"
        self.control = "ik"
        self.action_space_source = "tau0wm_solve_ik"
        self.action_space_layout = (
            "τ0-WM 相对 EE20（列约定）；infer 回绝对 EE20；ws solve_ik → 16D"
        )
        self.robot_config_path = None
        self._model_chunk_size: int | None = None

    def _sync_n_exec(self) -> None:
        if self._runtime is None:
            return
        n_exec = int(self._n_action_steps)
        if self._model_chunk_size is not None:
            n_exec = min(n_exec, int(self._model_chunk_size))
        self._runtime.n_exec = n_exec
        LOGGER.info(
            "τ0-WM n_exec=%s（solve_ik 路点数=菜单 action-step，chunk=%s）",
            self._runtime.n_exec,
            self._model_chunk_size,
        )

    def _resolve_model_chunk_size(self) -> int:
        meta = {}
        if self._runtime is not None:
            meta = self._runtime.get_server_metadata() or {}
        raw = meta.get("action_horizon")
        if isinstance(raw, (int, float)) and int(raw) >= 1:
            return int(raw)
        from tools.policy.policy_batch_support import read_action_horizon

        return read_action_horizon(self.cfg.ckpt_dir, PolicyType.TAU0WM)

    def load(self) -> None:
        super().load()
        horizon = self._resolve_model_chunk_size()
        self._model_chunk_size = horizon
        self._chunk_size = horizon
        if self._n_action_steps > horizon:
            LOGGER.warning(
                "τ0-WM n_action_steps=%s 超过 chunk_size=%s，截到 chunk",
                self._n_action_steps,
                horizon,
            )
            self._n_action_steps = horizon
        self._sync_n_exec()
        self._bind_client_ik(column_rot6d=True)
        LOGGER.info(
            "ACTION SPACE=%s → 控制=末端 IK (source=%s; %s) chunk_size=%s",
            self.action_space,
            self.action_space_source,
            self.action_space_layout,
            horizon,
        )

    def set_n_action_steps(self, n: int) -> None:
        n = int(n)
        if self._model_chunk_size is not None:
            n = min(n, int(self._model_chunk_size))
        super().set_n_action_steps(n)
        self._sync_n_exec()
