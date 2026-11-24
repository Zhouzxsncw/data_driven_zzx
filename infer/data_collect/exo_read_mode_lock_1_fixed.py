'''
用于从机器人外骨骼处获取数据传输到UDP特定端口中
python scripts/exo_read.py
'''

# Third party
import socket
import time
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

# Tong Genesis
from TG_exo import TGExo

def q_control_send(q_control, s, port_args):
    try:
        q_control = np.float64(q_control)
        s.sendto(q_control.tobytes(), port_args)
    except Exception as e:
        print(f"UDP发送出现错误: {e}")

def main():
    TGexo_params= {
        'dxl_name': 'XL330',  # 舵机型号
        'Device_name': '/dev/ttyUSB2',  # 串口号，可能需要根据实际情况修改
        'dxl_ids': [0, 1, 2,
               3, 4, 5, 6, 7, 8, 9,
               10, 11, 12, 13, 14, 15, 16,
               20],  # 舵机ID列表
        'baudrate': 1000000,  # 波特率
        'feedback_threshold': 0.05,  # 扭矩反馈阈值，反馈高于该值则
    }
    l_gripper_position = [1853, 2543]  # [夹爪闭合位置, 夹爪张开位置]
    r_gripper_position = [1853, 2543]  # [夹爪闭合位置, 夹爪张开位置]
    exo = TGExo(l_gripper_position, r_gripper_position, TGexo_params)
    exo.set_gripper_torque([0.3, 0.3])  # 设置夹爪扭矩

    fps = 240

    # 真实机械臂
    host = '192.168.1.13'
    port = 30005  # 控制端口
    s_c = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    port_args_c = (host, port)

    gripper_port = 30015
    s_g = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    port_args_g = (host, gripper_port)

    # 数据采集端口
    data_host = '192.168.1.12'
    data_port = 18131
    s_d = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    port_args_d = (data_host, data_port)

    q_control = np.zeros(19)
    gripper_control = np.zeros(3)
    # [修改] data_to_collect 扩为 20 维，索引19为时间戳
    data_to_collect = np.zeros(20)
    try:
        input("Calibration done. Press Enter to start sending data...")
        time_start = time.time()
        running_times = 0
        while True:
            once_start = time.time()
            angles, l_gripper_ratio, r_gripper_ratio = exo.read()

            q_control[1:18] = angles
            q_control[18] = time.time()  # 设置时间戳

            gripper_control[1] = l_gripper_ratio  # 左夹爪位置
            gripper_control[2] = r_gripper_ratio  # 右夹爪位置

            # [修改] 补齐右夹爪 + 时间戳在末尾，共17个值
            data_to_collect[:17] = angles              # 索引 0..16:  17个关节
            data_to_collect[17] = l_gripper_ratio      # 索引 17:    左夹爪
            data_to_collect[18] = r_gripper_ratio      # 索引 18:    右夹爪（已取消注释）
            data_to_collect[19] = time.time()          # 索引 19:    时间戳（末尾）

            q_control[0] = 2  # 正常控制信号
            q_control_send(q_control, s_c, port_args_c)  # 发送控制信号到机械臂

            # [修改] data_to_collect[3:] = 索引3到19共17个值
            # 布局: angles[3:17](14关节) + l_gripper + r_gripper + timestamp
            print("data_to_collect", data_to_collect[3:])
            print("q_control", q_control)
            q_control_send(data_to_collect[3:], s_d, port_args_d)  # 发送到采集端口 18131

            q_control_send(gripper_control, s_g, port_args_g)  # 发送夹爪位置
            running_times += 1
            once_end = time.time()
            sleep_time = max(0, (1.0 / fps) - (once_end - once_start))
            time.sleep(sleep_time)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        time_end = time.time()
        print(f"平均频率: {running_times/(time_end - time_start)} Hz")
        q_control[0] = 1  # 退出信号
        q_control_send(q_control, s_c, port_args_c)

        gripper_control[0] = 1  # 退出信号
        q_control_send(gripper_control, s_g, port_args_g)

        exo.end()
        s_c.close()
        s_g.close()
        s_d.close()
        print("退出控制，关闭舵机连接。")

if __name__ == "__main__":
    main()
