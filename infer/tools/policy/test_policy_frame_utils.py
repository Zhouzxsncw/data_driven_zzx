"""policy_frame_utils 单元测试。"""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class _FeatureStub:
    shape: list[int]


@dataclass
class _PolicyConfigStub:
    resize_shape: Optional[tuple[int, int]] = None
    image_features: Optional[Dict[str, Any]] = None
    input_features: Optional[Dict[str, Any]] = None


class ResolveImageTargetHwTests(unittest.TestCase):
    def test_ignores_resize_shape_when_input_features_present(self) -> None:
        from tools.policy.policy_frame_utils import resolve_image_target_hw

        cfg = _PolicyConfigStub(
            resize_shape=(224, 224),
            input_features={
                "observation.cam_left": _FeatureStub([540, 960, 3]),
                "observation.cam_mid": _FeatureStub([540, 960, 3]),
            },
        )
        self.assertEqual(resolve_image_target_hw(cfg), (540, 960))

    def test_prefers_input_features_over_image_features(self) -> None:
        """ACT：input_features=960x540（数采），image_features=640x480（preprocess）。"""
        from tools.policy.policy_frame_utils import resolve_image_target_hw

        cfg = _PolicyConfigStub(
            image_features={
                "observation.cam_left": _FeatureStub([480, 640, 3]),
                "observation.cam_mid": _FeatureStub([480, 640, 3]),
            },
            input_features={
                "observation.cam_left": _FeatureStub([540, 960, 3]),
                "observation.cam_mid": _FeatureStub([540, 960, 3]),
            },
        )
        self.assertEqual(resolve_image_target_hw(cfg), (540, 960))

    def test_falls_back_to_image_features_without_input_cameras(self) -> None:
        from tools.policy.policy_frame_utils import resolve_image_target_hw

        cfg = _PolicyConfigStub(
            image_features={
                "observation.cam_left": _FeatureStub([480, 640, 3]),
            },
            input_features={"observation.state": _FeatureStub([8])},
        )
        self.assertEqual(resolve_image_target_hw(cfg), (480, 640))

    def test_returns_preprocess_input_features(self) -> None:
        from tools.policy.policy_frame_utils import resolve_image_target_hw

        cfg = _PolicyConfigStub(
            input_features={
                "observation.cam_left": _FeatureStub([3, 224, 224]),
                "observation.cam_mid": _FeatureStub([3, 224, 224]),
            },
        )
        self.assertEqual(resolve_image_target_hw(cfg), (224, 224))

    def test_returns_none_without_camera_features(self) -> None:
        from tools.policy.policy_frame_utils import resolve_image_target_hw

        cfg = _PolicyConfigStub(
            resize_shape=(224, 224),
            input_features={"observation.state": _FeatureStub([8])},
        )
        self.assertIsNone(resolve_image_target_hw(cfg))


if __name__ == "__main__":
    unittest.main()
