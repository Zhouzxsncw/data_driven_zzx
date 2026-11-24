"""外骨骼数采相机元数据与推理分辨率对齐。"""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field

from infer.exo.exo_collect import ExoLeRobotCollector


@dataclass
class _CollectCameraStub:
    key: str
    serial: str
    width: int
    height: int
    fps: int = 30


@dataclass
class _CollectCamerasStub:
    enabled: bool = True
    devices: list = field(default_factory=list)


@dataclass
class _CollectCfgStub:
    cameras: _CollectCamerasStub
    state_modalities: object = None
    action_modalities: object = None
    dataset: object = None
    sync: object = None

    def resolve_repo_id(self) -> str:
        return "test_repo"


class ExoCollectCameraShapeTests(unittest.TestCase):
    def test_sync_camera_shapes_from_infer_devices(self) -> None:
        collect_cfg = _CollectCfgStub(
            cameras=_CollectCamerasStub(
                devices=[
                    _CollectCameraStub("observation.cam_left", "s0", 640, 480),
                    _CollectCameraStub("observation.cam_mid", "s1", 640, 480),
                ]
            ),
        )
        collector = ExoLeRobotCollector.__new__(ExoLeRobotCollector)
        collector.cfg = collect_cfg
        collector._camera_shapes = {
            d.key: (d.height, d.width, 3) for d in collect_cfg.cameras.devices
        }

        @dataclass
        class _InferDev:
            serial: str
            width: int
            height: int
            model_width: int | None = None
            model_height: int | None = None

        collector.sync_camera_shapes_from_infer_devices(
            [_InferDev("s0", 640, 480, 960, 540), _InferDev("s1", 640, 480, 960, 540)]
        )
        self.assertEqual(
            collector._camera_shapes["observation.cam_left"],
            (540, 960, 3),
        )
        self.assertEqual(
            collector._camera_shapes["observation.cam_mid"],
            (540, 960, 3),
        )

    def test_sync_camera_shapes_prefers_model_resolution(self) -> None:
        collect_cfg = _CollectCfgStub(
            cameras=_CollectCamerasStub(
                devices=[
                    _CollectCameraStub("observation.cam_left", "s0", 640, 480),
                ]
            ),
        )
        collector = ExoLeRobotCollector.__new__(ExoLeRobotCollector)
        collector.cfg = collect_cfg
        collector._camera_shapes = {
            d.key: (d.height, d.width, 3) for d in collect_cfg.cameras.devices
        }

        @dataclass
        class _InferDev:
            serial: str
            width: int
            height: int
            model_width: int
            model_height: int

        collector.sync_camera_shapes_from_infer_devices(
            [_InferDev("s0", 640, 480, 224, 224)]
        )
        self.assertEqual(
            collector._camera_shapes["observation.cam_left"],
            (224, 224, 3),
        )


if __name__ == "__main__":
    unittest.main()
