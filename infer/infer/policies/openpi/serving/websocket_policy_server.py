import asyncio
import http
import logging
import sys
import time
import traceback
from pathlib import Path

# 共享服务端请求级日志模块（infer/recording/server_logger.py）
_RECORDING_DIR = Path(__file__).resolve().parents[3] / "recording"
if str(_RECORDING_DIR) not in sys.path:
    sys.path.insert(0, str(_RECORDING_DIR))
try:
    from server_logger import ServerRequestLogger  # noqa: E402
except ImportError:
    ServerRequestLogger = None  # type: ignore[assignment,misc]

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
        rtc_config: dict | None = None,
        srv_logger: "ServerRequestLogger | None" = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._rtc_config = rtc_config or {}
        self._srv_logger = srv_logger
        self._request_id = 0
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        if self._srv_logger is not None:
            self._srv_logger.start()
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        client_addr = str(websocket.remote_address)
        logger.info(f"Connection from {client_addr} opened")
        if self._srv_logger:
            self._srv_logger.log_connection(client_addr=client_addr, event="connect",
                                            metadata=self._metadata)
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        
        while True:
            try:
                start_time = time.monotonic()
                request_data = msgpack_numpy.unpackb(await websocket.recv())
                self._request_id += 1
                
                # 处理请求数据：可能包含obs、prev_action_chunk和executed_steps
                if isinstance(request_data, dict) and "obs" in request_data:
                    obs = request_data["obs"]
                    prev_action_chunk = request_data.get("prev_action_chunk", None)
                    executed_steps = request_data.get("executed_steps", 0)
                else:
                    # 向后兼容：直接是observation
                    obs = request_data
                    prev_action_chunk = None
                    executed_steps = 0

                infer_time = time.monotonic()
                rtc_used = False
                
                # RTC推理或标准推理
                if self._rtc_config.get("use_rtc_guidance", False) and prev_action_chunk is not None:
                    # 使用RTC引导推理
                    if hasattr(self._policy, 'infer_with_rtc_guidance'):
                        # 创建delta action mask（ALOHA双臂：前6维和后6维是关节，中间2维是夹爪）
                        import numpy as np
                        from openpi.transforms import make_bool_mask
                        delta_action_mask = make_bool_mask(6, -1, 6, -1)
                        
                        # 计算RTC参数
                        inference_delay = self._rtc_config.get("rtc_inference_delay", 4)
                        action_horizon = getattr(self._policy._model, 'action_horizon', 8)
                        prefix_attention_horizon = action_horizon - self._rtc_config.get("rtc_execute_horizon", 1)
                        prefix_attention_schedule = self._rtc_config.get("rtc_prefix_attention_schedule", "exp")
                        max_guidance_weight = self._rtc_config.get("rtc_max_guidance_weight", 1.0)
                        
                        # 调用策略的RTC推理，参数顺序与模型函数签名保持一致
                        action = self._policy.infer_with_rtc_guidance(
                            obs,                          # dict: 观测数据
                            prev_action_chunk,            # np.ndarray: 前一次动作序列
                            inference_delay,              # int: 推理延迟步数
                            prefix_attention_horizon,     # int: 前缀注意力范围
                            prefix_attention_schedule,    # str: 注意力调度方式
                            max_guidance_weight,          # float: 最大引导权重
                            delta_action_mask,            # np.ndarray|None: 增量动作掩码
                            executed_steps                # int: 已执行步数
                        )
                        rtc_used = True
                        logger.info(f"RTC Guided Inference: delay={inference_delay}, horizon={prefix_attention_horizon}")
                    else:
                        action = self._policy.infer(obs)
                        logger.warning("Policy does not support RTC guidance, falling back to standard inference")
                else:
                    # 标准推理
                    start_time = time.time()
                    action = self._policy.infer(obs)
                    if self._rtc_config.get("use_rtc_guidance", False) and prev_action_chunk is None:
                        logger.info("RTC First Inference (no guidance)")
                
                infer_time = time.monotonic() - infer_time
                infer_ms = infer_time * 1000

                action["server_timing"] = {
                    "infer_ms": infer_ms,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                # 网络发送
                t_send = time.monotonic()
                await websocket.send(packer.pack(action))
                send_ms = (time.monotonic() - t_send) * 1000
                prev_total_time = time.monotonic() - start_time

                # 请求级日志
                if self._srv_logger:
                    # 构建 timing 分解
                    log_timing: dict = {"total_ms": round(infer_ms, 2), "network_send_ms": round(send_ms, 2)}
                    # 如果 action 携带内部 timing 分解（_timing），合并到日志
                    inner_timing = action.pop("_timing", None)
                    if isinstance(inner_timing, dict):
                        log_timing.update(inner_timing)
                    self._srv_logger.log_request(
                        request_id=self._request_id,
                        client_addr=client_addr,
                        infer_ms=infer_ms,
                        request_data=request_data if isinstance(request_data, dict) else None,
                        response_data=action,
                        timing=log_timing,
                        extra={"rtc_used": rtc_used} if rtc_used else None,
                    )

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {client_addr} closed")
                if self._srv_logger:
                    self._srv_logger.log_connection(client_addr=client_addr, event="disconnect")
                break
            except Exception:
                tb = traceback.format_exc()
                if self._srv_logger:
                    self._srv_logger.log_request(
                        request_id=self._request_id,
                        client_addr=client_addr,
                        infer_ms=0,
                        error=tb.splitlines()[-1] if tb else "unknown",
                    )
                await websocket.send(tb)
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

    def close_logger(self) -> dict | None:
        """关闭请求级日志，返回统计摘要。"""
        if self._srv_logger:
            return self._srv_logger.close()
        return None


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
