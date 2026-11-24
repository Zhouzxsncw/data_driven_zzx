"""PICO-relative RM75 targets using the inference session's arm handles."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from infer.io.pico_input import PicoFrame


_OPERATOR_TO_RIGHT_BASE = np.array([[0., 0., 1.], [-1., 0., 0.], [0., 1., 0.]])
_LEFT_MIRROR = np.diag([1., -1., -1.])


def _base_maps(frame: PicoFrame) -> tuple[np.ndarray, np.ndarray]:
    forward = Rotation.from_quat(frame.headset_pose[3:]).apply([0., 0., -1.])
    forward[1] = 0.
    norm = np.linalg.norm(forward)
    if norm < 0.2:
        raise ValueError("PICO headset has no stable horizontal heading")
    forward /= norm
    up = np.array([0., 1., 0.])
    view = np.array([forward, np.cross(forward, up), up])
    right = _OPERATOR_TO_RIGHT_BASE @ view
    return _LEFT_MIRROR @ right, right


class PicoRm75Targets:
    """Anchor both controllers once, then return guarded physical joint angles."""

    def __init__(self, arms, frame: PicoFrame, joints: np.ndarray, active_sides=None) -> None:
        self.arms = arms
        self.controllers = (frame.left, frame.right)
        self.active_sides = frozenset(active_sides or ("left", "right"))
        self.anchor_joints = joints.copy()
        self.maps = _base_maps(frame)
        self.poses = []
        self.limits = []
        for arm, seed in zip(arms, joints):
            rc_lo, lo = arm.rm_get_joint_min_pos()
            rc_hi, hi = arm.rm_get_joint_max_pos()
            lo = np.asarray(lo, dtype=float)
            hi = np.asarray(hi, dtype=float)
            if (rc_lo or rc_hi or lo.shape != (7,) or hi.shape != (7,)
                    or not np.isfinite([lo, hi]).all() or np.any(hi - lo <= 6.)):
                raise RuntimeError("Could not read RM75 joint limits")
            pose = np.asarray(arm.rm_algo_forward_kinematics(seed.tolist(), 0), dtype=float)
            if pose.shape != (7,) or not np.isfinite(pose).all():
                raise RuntimeError("Could not anchor RM75 tool pose")
            self.limits.append((lo + 3., hi - 3.))
            self.poses.append(pose)
        self.sequence = frame.sequence

    def target(self, frame: PicoFrame, joints: np.ndarray, params_type) -> np.ndarray | None:
        if frame.sequence == self.sequence:
            return None
        self.sequence = frame.sequence
        output = joints.copy()
        for i, (arm, current, anchor) in enumerate(
            zip(self.arms, (frame.left, frame.right), self.controllers)
        ):
            side = "left" if i == 0 else "right"
            if side not in self.active_sides:
                continue
            base = self.maps[i]
            pose = self.poses[i]
            shift = 0.30 * (base @ (np.array(current.position) - anchor.position))
            shift_len = np.linalg.norm(shift)
            if shift_len > 0.10:
                shift *= 0.10 / shift_len

            relative = Rotation.from_quat(current.orientation) * Rotation.from_quat(anchor.orientation).inv()
            vector = relative.as_rotvec()
            # Axial vectors acquire the determinant when mapped through a reflection.
            vector = 0.4 * np.linalg.det(base) * (base @ vector)
            angle = np.linalg.norm(vector)
            if angle > 0.45:
                vector *= 0.45 / angle
            nominal = Rotation.from_quat(pose[[4, 5, 6, 3]])
            target_q = (Rotation.from_rotvec(vector) * nominal).as_quat()
            target = np.r_[pose[:3] + shift, target_q[[3, 0, 1, 2]]]
            if shift_len < 0.0005 and angle < 0.01:
                # A return to the anchored hand pose still needs a target after motion.
                if np.allclose(joints[i], self.anchor_joints[i], atol=0.1):
                    continue
            rc, solved = arm.rm_algo_inverse_kinematics(
                params_type(q_in=joints[i].tolist(), q_pose=target.tolist(), flag=0)
            )
            solved = np.asarray(solved, dtype=float)
            if rc or solved.shape != (7,) or not np.isfinite(solved).all():
                continue
            delta = solved - joints[i]
            # Compare physical angles, including J7's multi-turn range.
            peak = np.max(np.abs(delta))
            if peak > 25.:
                continue
            candidate = joints[i] + delta * min(1., 1. / max(peak, 1e-9))
            lo, hi = self.limits[i]
            # Allow recovery from a joint already inside the limit margin.
            if (np.any(candidate < np.minimum(lo, joints[i]))
                    or np.any(candidate > np.maximum(hi, joints[i]))):
                continue
            elbow = joints[i, 3]
            if (abs(candidate[3]) < min(10., abs(elbow)) or candidate[3] * elbow < 0):
                continue
            output[i] = candidate
        return output if not np.array_equal(output, joints) else None
