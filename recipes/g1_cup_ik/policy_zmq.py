"""Isaac Lab 侧薄客户端：只依赖 pyzmq + msgpack_numpy，不 import gr00t。"""

from __future__ import annotations

from typing import Any

import msgpack
import msgpack_numpy as mnp
import zmq


class PolicyZmqClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 5550, timeout_ms: int = 120000):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.sock.connect(f"tcp://{host}:{port}")

    def ping(self) -> bool:
        try:
            self._call("ping", requires_input=False)
            return True
        except zmq.error.ZMQError:
            return False

    def get_action(self, observation: dict[str, Any], options: dict[str, Any] | None = None) -> dict:
        # PolicyServer 回 (action, info)；action 各键是 (B, T, D)，[0] 不是「只取 env0」。
        resp = self._call("get_action", {"observation": observation, "options": options})
        if isinstance(resp, (list, tuple)):
            return resp[0]
        return resp

    def _call(self, endpoint: str, data: dict | None = None, requires_input: bool = True) -> Any:
        req: dict[str, Any] = {"endpoint": endpoint}
        if requires_input:
            req["data"] = data
        self.sock.send(msgpack.packb(req, default=mnp.encode))
        raw = self.sock.recv()
        if raw == b"ERROR":
            raise RuntimeError("PolicyServer ERROR")
        resp = msgpack.unpackb(raw, object_hook=mnp.decode, raw=False)
        if isinstance(resp, dict) and "error" in resp:
            raise RuntimeError(resp["error"])
        return resp

    def close(self) -> None:
        try:
            self.sock.close(linger=0)
        except Exception:
            pass
        try:
            self.ctx.term()
        except Exception:
            pass
