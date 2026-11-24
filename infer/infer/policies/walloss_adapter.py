# WALL-OSS 适配器。YAML policy.type: walloss
# 经 websocket 调 serve_tong_walloss.py（默认端口 1239）。
# GPU：相对 EE → 绝对 EE20。客户端读完后同一条 ws ``solve_ik`` → 16D。
# 模型 chunk 以 ckpt config.yml / 服务端握手 action_horizon 为准。
# 菜单 action-step 范围 1~chunk；所选值 = 客户端 IK 路点数 n_exec。

from __future__ import annotations

import logging

from infer.core.types import PolicyType
from infer.policies import register_policy
from infer.policies.client_ee_ik import ClientAbsEeIkMixin
from infer.policies.lingbot_va_adapter import LingbotVaAdapter

LOGGER = logging.getLogger(__name__)


@register_policy(PolicyType.WALLOSS)
class WallOssAdapter(ClientAbsEeIkMixin, LingbotVaAdapter):
    name = "walloss"
    _display_name = "WALL-OSS"
    _backend_key = "walloss_websocket"
    _default_port = 1239
    _default_horizon = 32
    _pin_exec_horizon_always = False
    _default_expected_model = "walloss"
    _wrong_server_hint = (
        "WALL-OSS 用端口 1239，MemoryVLA 用 1238，InternVLA 用 1237，"
        "Cosmos3 用 1236，DreamZero 用 1235，LingBot-VA 用 1234。"
    )

    def __init__(self, cfg, device=None) -> None:
        super().__init__(cfg, device=device)
        self.action_space = "ee"
        self.control = "ik"
        self.action_space_source = "walloss_solve_ik"
        self.action_space_layout = "WALL-OSS 相对 EE20；infer 回绝对 EE20；ws solve_ik → 16D"
        self.robot_config_path = None
        self._model_chunk_size: int | None = None

    def _sync_n_exec(self) -> None:
        """把当前 action-step 同步成服务端截断的绝对 EE 路点数。"""
        if self._runtime is None:
            return
        n_exec = int(self._n_action_steps)
        if self._model_chunk_size is not None:
            n_exec = min(n_exec, int(self._model_chunk_size))
        self._runtime.n_exec = n_exec
        LOGGER.info(
            "WALL-OSS n_exec=%s（solve_ik 路点数=菜单 action-step，chunk=%s）",
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

        return read_action_horizon(self.cfg.ckpt_dir, PolicyType.WALLOSS)

    def load(self) -> None:
        super().load()
        horizon = self._resolve_model_chunk_size()
        self._model_chunk_size = horizon
        self._chunk_size = horizon
        if self._n_action_steps > horizon:
            LOGGER.warning(
                "WALL-OSS n_action_steps=%s 超过 chunk_size=%s，截到 chunk",
                self._n_action_steps,
                horizon,
            )
            self._n_action_steps = horizon
        self._sync_n_exec()
        self._bind_client_ik()
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
