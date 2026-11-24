"""LongWAM uses the full raw observation history, not the VA two-frame KV protocol."""
from collections import deque
import numpy as np

from infer.core.types import PolicyType
from infer.policies import register_policy
from infer.policies.lingbot_va_adapter import LingbotVaAdapter
from infer.policies.lingbot_va_runtime import LingbotVaWebsocketRuntime


class LongWAMWebsocketRuntime(LingbotVaWebsocketRuntime):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._fps5_frames = deque(maxlen=49)

    def build_observation(self, images, state):
        fallback = self._images_to_dict(images)
        with self._fps5_lock:
            frames = [frame for _, frame in self._fps5_frames]
        # The camera buffer is timestamped at 15 Hz. Use its last anchor; do not
        # append an extra inference-time frame and change the temporal stride.
        return {"obs": frames or [fallback], "state": np.asarray(state, dtype=np.float32).reshape(-1)}

    def _infer_locked(self, obs, t0):
        # Each request contains its own history. Never send padded VA KV messages.
        self._is_first_infer = True
        return super()._infer_locked(obs, t0)


@register_policy(PolicyType.LONGWAM)
class LongWAMAdapter(LingbotVaAdapter):
    name = "longwam"
    _display_name = "LongWAM"
    _backend_key = "longwam_websocket"
    _runtime_class = LongWAMWebsocketRuntime
    _default_port = 1246
    _default_horizon = 32
    _pin_exec_horizon_always = True
    _default_expected_model = "longwam"
    _default_obs_fps = 15.0
    _wrong_server_hint = "LongWAM uses port 1246 and its own trained MoT checkpoint."
