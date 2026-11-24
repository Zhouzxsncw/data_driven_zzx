"""torchvision>=0.27 移除了 VideoReader；用 PyAV 补回，供 LeRobot 数据集解码（不改 src/lerobot）。"""
from __future__ import annotations

import torchvision


class _AvVideoReader:
    def __init__(self, path: str, stream: str = "video") -> None:
        import av

        self._av = av
        self.container = av.open(path)
        self.stream = stream
        self._video = self.container.streams.video[0]
        self._time_base = float(self._video.time_base)
        self._iter = None

    def seek(self, timestamp: float, keyframes_only: bool = False) -> None:
        ts = max(0.0, float(timestamp))
        self.container.seek(int(ts / self._time_base), stream=self._video, backward=True)

    def __iter__(self):
        for frame in self.container.decode(self._video):
            pts = float(frame.pts * self._time_base) if frame.pts is not None else 0.0
            data = torch_from_av_frame(frame)
            yield {"pts": pts, "data": data}


def torch_from_av_frame(frame):
    import torch
    import numpy as np

    arr = frame.to_ndarray(format="rgb24")
    return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0


def install() -> None:
    torchvision.io.VideoReader = _AvVideoReader  # type: ignore[attr-defined]


install()
