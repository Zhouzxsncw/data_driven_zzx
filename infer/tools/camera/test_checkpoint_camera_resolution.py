"""checkpoint_camera_resolution 单元测试。"""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field

import numpy as np

from data_collect.sync import TimestampedBuffer
from infer.io.camera_payload import pack_camera_payload
from tools.camera.checkpoint_camera_resolution import (
    apply_checkpoint_camera_resolution,
    device_output_hw,
    format_camera_resolution_mismatch,
    is_model_preprocess_image_shape,
    latest_camera_hw_from_image_buf,
    needs_output_resize,
    verify_camera_capture_resolution,
)


@dataclass
class _CameraDeviceStub:
    serial: str
    width: int | None = None
    height: int | None = None
    model_width: int | None = None
    model_height: int | None = None
    fps: int = 30


@dataclass
class _CameraSenderStub:
    enabled: bool = True
    transport: str = "local"
    devices: list = field(default_factory=list)


@dataclass
class _PolicyStub:
    ckpt_dir: object = None
    type: object = None
    camera_keys: list | None = None


@dataclass
class _CfgStub:
    policy: _PolicyStub
    camera_sender: _CameraSenderStub = field(default_factory=_CameraSenderStub)


class CheckpointCameraResolutionTests(unittest.TestCase):
    def test_is_model_preprocess_image_shape(self) -> None:
        self.assertTrue(is_model_preprocess_image_shape([3, 224, 224]))
        self.assertTrue(is_model_preprocess_image_shape([224, 224, 3]))
        self.assertFalse(is_model_preprocess_image_shape([3, 540, 960]))
        self.assertFalse(is_model_preprocess_image_shape([480, 640, 3]))
        self.assertFalse(is_model_preprocess_image_shape([540, 960, 3]))

    def test_apply_checkpoint_keeps_manual_resolution_for_preprocess_shape(self) -> None:
        device = _CameraDeviceStub("a", width=640, height=480)
        cfg = _CfgStub(
            policy=_PolicyStub(type="act"),
            camera_sender=_CameraSenderStub(devices=[device]),
        )
        ckpt_cfg = {
            "input_features": {
                "observation.cam_left": {"shape": [3, 224, 224]},
            }
        }
        apply_checkpoint_camera_resolution(cfg, ckpt_cfg=ckpt_cfg)
        self.assertEqual(device.model_width, 224)
        self.assertEqual(device.model_height, 224)
        self.assertEqual(device.width, 640)
        self.assertEqual(device.height, 480)
        self.assertTrue(needs_output_resize(device))
        self.assertEqual(device_output_hw(device), (224, 224))

    def test_latest_camera_hw_from_image_buf(self) -> None:
        buf = TimestampedBuffer(maxlen=10)
        buf.append(
            1.0,
            pack_camera_payload(
                [
                    np.zeros((540, 960, 3), dtype=np.uint8),
                    np.zeros((540, 960, 3), dtype=np.uint8),
                ]
            ),
        )
        self.assertEqual(
            latest_camera_hw_from_image_buf(buf),
            [(540, 960), (540, 960)],
        )

    def test_verify_camera_capture_resolution_ok(self) -> None:
        buf = TimestampedBuffer(maxlen=10)
        buf.append(
            1.0,
            pack_camera_payload([np.zeros((480, 640, 3), dtype=np.uint8)]),
        )
        cfg = _CfgStub(
            policy=_PolicyStub(),
            camera_sender=_CameraSenderStub(
                devices=[_CameraDeviceStub("a", 640, 480)],
            ),
        )
        ok, detail = verify_camera_capture_resolution(cfg, buf)
        self.assertTrue(ok)
        self.assertIn("一致", detail)

    def test_verify_camera_capture_resolution_mismatch(self) -> None:
        buf = TimestampedBuffer(maxlen=10)
        buf.append(
            1.0,
            pack_camera_payload([np.zeros((540, 960, 3), dtype=np.uint8)]),
        )
        cfg = _CfgStub(
            policy=_PolicyStub(),
            camera_sender=_CameraSenderStub(
                devices=[_CameraDeviceStub("a", 640, 480)],
            ),
        )
        ok, detail = verify_camera_capture_resolution(cfg, buf)
        self.assertFalse(ok)
        self.assertIn("cam0", detail)

    def test_format_camera_resolution_mismatch(self) -> None:
        text = format_camera_resolution_mismatch(
            [(540, 960)],
            [(480, 640)],
        )
        self.assertIn("960x540", text)
        self.assertIn("640x480", text)


if __name__ == "__main__":
    unittest.main()
