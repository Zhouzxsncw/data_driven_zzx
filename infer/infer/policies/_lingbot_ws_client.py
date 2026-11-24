"""Minimal LingBot-VLA websocket client（不依赖 lingbotvla 包）。"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Tuple

import websockets.sync.client

from infer.policies._lingbot_msgpack_numpy import Packer, unpackb

LOGGER = logging.getLogger(__name__)


class WebsocketClientPolicy:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: Optional[int] = None,
        api_key: Optional[str] = None,
        wait_timeout_sec: float = 300.0,
    ) -> None:
        self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server(wait_timeout_sec)

    def get_server_metadata(self) -> Dict[str, Any]:
        return self._server_metadata

    def _wait_for_server(
        self, wait_timeout_sec: float
    ) -> Tuple[websockets.sync.client.ClientConnection, Dict[str, Any]]:
        LOGGER.info("Waiting for LingBot server at %s ...", self._uri)
        deadline = time.time() + max(1.0, float(wait_timeout_sec))
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                )
                metadata = unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                if time.time() >= deadline:
                    raise TimeoutError(
                        f"连接 LingBot 服务超时（{wait_timeout_sec:.0f}s）: {self._uri}；"
                        "请先在 lingbotvla 环境启动 "
                        "`python -m deploy.lingbot_vla_v2_policy --port ...`"
                    )
                LOGGER.info("Still waiting for LingBot server...")
                time.sleep(2.0)

    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in LingBot inference server:\n{response}")
        return unpackb(response)

    def reset(self, robo_name: str) -> None:
        self.infer(dict(reset=True, robo_name=robo_name))
