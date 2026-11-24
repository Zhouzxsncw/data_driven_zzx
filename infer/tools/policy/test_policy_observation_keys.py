"""policy_observation_keys 单元测试。"""

from __future__ import annotations

import unittest

from tools.policy.policy_observation_keys import (
    order_images_for_model,
    resolve_device_camera_keys,
    resolve_infer_camera_keys,
    resolve_policy_image_keys,
)


class _DeviceStub:
    def __init__(self, serial: str, key: str | None = None) -> None:
        self.serial = serial
        self.key = key


class PolicyObservationKeysTests(unittest.TestCase):
    def test_resolve_infer_camera_keys_from_checkpoint_subset(self) -> None:
        keys = resolve_infer_camera_keys(
            None,
            {
                "observation.state": {"shape": [8]},
                "observation.cam_mid": {"shape": [540, 960, 3]},
            },
        )
        self.assertEqual(keys, ["observation.cam_mid"])

    def test_resolve_infer_camera_keys_explicit_empty(self) -> None:
        keys = resolve_infer_camera_keys(
            [],
            {"observation.cam_mid": {"shape": [540, 960, 3]}},
        )
        self.assertEqual(keys, [])

    def test_resolve_infer_camera_keys_images_namespace(self) -> None:
        keys = resolve_infer_camera_keys(
            None,
            {
                "observation.images.cam_left": {"shape": [540, 960, 3]},
                "observation.images.cam_mid": {"shape": [540, 960, 3]},
            },
        )
        self.assertEqual(
            keys,
            ["observation.images.cam_left", "observation.images.cam_mid"],
        )

    def test_resolve_policy_image_keys_prefers_yaml(self) -> None:
        class _Cfg:
            input_features = {"observation.cam_mid": {"shape": [540, 960, 3]}}

        keys = resolve_policy_image_keys(["observation.cam_left"], _Cfg())
        self.assertEqual(keys, ["observation.cam_left"])

    def test_order_images_for_model_by_device_key(self) -> None:
        images = ["mid_img", "left_img"]
        ordered = order_images_for_model(
            images,
            ["observation.cam_left", "observation.cam_mid"],
            ["observation.cam_mid", "observation.cam_left"],
        )
        self.assertEqual(ordered, ["left_img", "mid_img"])

    def test_resolve_device_camera_keys_fallback_positional(self) -> None:
        keys = resolve_device_camera_keys(
            [_DeviceStub("s0"), _DeviceStub("s1")],
            ["observation.cam_left", "observation.cam_mid"],
        )
        self.assertEqual(
            keys,
            ["observation.cam_left", "observation.cam_mid"],
        )

    def test_capture_index_for_recording_slot(self) -> None:
        from tools.policy.policy_observation_keys import capture_index_for_recording_slot

        device_keys = ["observation.cam_mid"]
        self.assertEqual(capture_index_for_recording_slot(device_keys, "cam_mid"), 0)
        self.assertIsNone(capture_index_for_recording_slot(device_keys, "cam_left"))


if __name__ == "__main__":
    unittest.main()
