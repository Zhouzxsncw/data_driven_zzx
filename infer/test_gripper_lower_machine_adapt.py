"""数采夹爪/action 包解析与下位机协议对齐测试。"""

from __future__ import annotations

import unittest

import numpy as np

from data_collect.modalities import (
    ModalityAssembler,
    ModalityGroupConfig,
    ModalityKind,
    ModalityToggle,
    StreamKind,
    normalize_gripper_state_packet,
    normalize_robot_action_packet,
)


def _dual_gripper_group(*, right_on: bool = True) -> ModalityGroupConfig:
    on = ModalityToggle(enabled=True, required=True, fallback="drop_frame")
    off = ModalityToggle(enabled=False, required=False, fallback="drop_frame")
    return ModalityGroupConfig(
        left_arm=on,
        right_arm=on,
        left_gripper=on,
        right_gripper=on if right_on else off,
        right_hand=off,
    )


class TestNormalizePackets(unittest.TestCase):
    def test_gripper_state_5d(self) -> None:
        raw = np.asarray([0.2, 0.8, 1710000000.0, 12.5, -3.0], dtype=np.float64)
        out = normalize_gripper_state_packet(raw)
        self.assertEqual(out.size, 2)
        self.assertAlmostEqual(float(out[0]), 0.2, places=9)
        self.assertAlmostEqual(float(out[1]), 0.8, places=9)

    def test_gripper_state_3d(self) -> None:
        raw = np.asarray([0.1, 0.9, 1710000000.0], dtype=np.float64)
        out = normalize_gripper_state_packet(raw)
        np.testing.assert_allclose(out, [0.1, 0.9])

    def test_robot_action_17d(self) -> None:
        raw = np.zeros(17, dtype=np.float64)
        raw[14] = 0.3
        raw[15] = 0.7
        raw[16] = 1710000000.0
        out = normalize_robot_action_packet(raw)
        self.assertEqual(out.size, 16)
        self.assertAlmostEqual(float(out[14]), 0.3, places=9)
        self.assertAlmostEqual(float(out[15]), 0.7, places=9)

    def test_robot_action_legacy_15d(self) -> None:
        raw = np.zeros(15, dtype=np.float64)
        raw[14] = 0.4
        out = normalize_robot_action_packet(raw)
        self.assertEqual(out.size, 16)
        self.assertAlmostEqual(float(out[14]), 0.4, places=9)
        self.assertAlmostEqual(float(out[15]), 0.4, places=9)


class TestAssemblerDualGripper(unittest.TestCase):
    def test_source0_uses_gripper_state_both(self) -> None:
        assembler = ModalityAssembler(
            _dual_gripper_group(),
            _dual_gripper_group(),
            gripper_source=0,
        )
        packets = {
            StreamKind.ROBOT_STATE: np.zeros(14, dtype=np.float64),
            StreamKind.ROBOT_ACTION: np.zeros(16, dtype=np.float64),
            StreamKind.GRIPPER_STATE: np.asarray([0.25, 0.75], dtype=np.float64),
        }
        packets = assembler.normalize_gripper_packets(packets)
        # filter may binarize; with open defaults and raw far from open, after enough updates
        # use disabled-filter path via direct extract after setting filter off
        assembler = ModalityAssembler(
            _dual_gripper_group(),
            _dual_gripper_group(),
            gripper_source=0,
            gripper_value_filter=type(assembler.gripper_value_filter_cfg)(enabled=False),
        )
        packets = {
            StreamKind.ROBOT_STATE: np.zeros(14, dtype=np.float64),
            StreamKind.ROBOT_ACTION: np.zeros(16, dtype=np.float64),
            StreamKind.GRIPPER_STATE: np.asarray([0.25, 0.75], dtype=np.float64),
        }
        state, err = assembler.assemble_vector(packets, for_action=False)
        self.assertIsNone(err)
        assert state is not None
        # left7 + right7 + left_g + right_g
        self.assertEqual(state.size, 16)
        self.assertAlmostEqual(float(state[14]), 0.25, places=5)
        self.assertAlmostEqual(float(state[15]), 0.75, places=5)

    def test_source1_uses_action_both(self) -> None:
        from data_collect.modalities import GripperValueFilterConfig

        assembler = ModalityAssembler(
            _dual_gripper_group(),
            _dual_gripper_group(),
            gripper_source=1,
            gripper_value_filter=GripperValueFilterConfig(enabled=False),
        )
        action = np.zeros(16, dtype=np.float64)
        action[14] = 0.11
        action[15] = 0.22
        packets = {
            StreamKind.ROBOT_STATE: np.zeros(14, dtype=np.float64),
            StreamKind.ROBOT_ACTION: action,
        }
        state, err = assembler.assemble_vector(packets, for_action=False)
        self.assertIsNone(err)
        assert state is not None
        self.assertAlmostEqual(float(state[14]), 0.11, places=5)
        self.assertAlmostEqual(float(state[15]), 0.22, places=5)
        self.assertIn(StreamKind.ROBOT_ACTION, assembler.required_streams())
        self.assertNotIn(StreamKind.GRIPPER_STATE, assembler.required_streams())


if __name__ == "__main__":
    unittest.main()
