import numpy as np
import time 
import math
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from servo import MultiDynamixel, Dynamixel

class TGExo:
    def __init__(self, l_gripper_position, r_gripper_position, params):
        self.joint = MultiDynamixel(**params)
        self.joint.set_torque_mode([0]*19)  # 设置为力矩模式
        # self.joint.set_torque_mode([0]*15)
        self.joint.down_torque() # 下力矩，不需要控制外骨骼
        self.l_gripper_position = l_gripper_position
        # self.r_gripper_position = r_gripper_position
        
    def set_gripper_torque(self, torque):
        self.joint.single_torque_feedback(self.joint.DXL_IDs[17], torque[0])
        # self.joint.single_torque_feedback(self.joint.DXL_IDs[14], torque[0])
        # self.joint.single_torque_feedback(self.joint.DXL_IDs[18], torque[1])
        # self.joint.single_torque_feedback(self.joint.DXL_IDs[15], torque[1])

    def read(self):
        angles = self.joint.present_angle_read()
        angles = self._limit_angle(angles)
        l_gripper_angle = angles[17]
        # l_gripper_angle = angles[14]
        # r_gripper_angle = angles[18]
        l_gripper_ratio = (l_gripper_angle + 180)/0.088 - self.l_gripper_position[0]
        l_gripper_ratio /= (self.l_gripper_position[1] - self.l_gripper_position[0])
        # r_gripper_ratio = (r_gripper_angle + 180)/0.088 - self.r_gripper_position[0]
        # r_gripper_ratio /= (self.r_gripper_position[1] - self.r_gripper_position[0])
        return ([math.radians(angle) for angle in angles[:17]], l_gripper_ratio, 0)

    def end(self):
        self.joint.exit_dxl_control()

    def _limit_angle(self, angles):
        limited_angles = []
        for angle in angles:
            angle = (angle + 180) % 360 - 180  # 将角度限制在[-180, 180]范围内
            limited_angles.append(angle)
        return limited_angles
