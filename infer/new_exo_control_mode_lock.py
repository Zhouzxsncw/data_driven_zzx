'''
用于从UDP特定端口中接受机器人控制命令，并将机器人状态通过UDP发送到上位机。

运行机械臂控制脚本：
python scripts/new_exo_robot_control.py

explanation of the title:
- new_exo: 适用于新版外骨骼的机械臂控制代码，与旧版本的区别在于处理数据时机器人腰部第2关节和第3关节
    的正负方向不同。（此问题在URDF中也有出现，需要注意）
'''

# third-party
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
import socket
import time
import numpy as np
import threading
import math
from queue import Queue

# TG_control
from TG_control import RoboTG_g #('g' for gripper)

# 全局队列和停止事件
data_queue = Queue(1)  # 机器人控制命令队列
time_queue = Queue(1)  # 时间戳队列
gripper_queue = Queue(1)  # 夹爪控制指令队列
now_torso_pos = Queue(1)  # 腰部当前关节位置队列
stop_event = threading.Event()
reset_event = threading.Event()
INIT_FLAG = 9

def clear_control_queues():
    while not data_queue.empty():
        data_queue.get()
    while not time_queue.empty():
        time_queue.get()
    while not gripper_queue.empty():
        gripper_queue.get()

def reset_robot_to_init(robot, robot_init_pos):
    print("[INFO] 正在回到标准初始位...")
    reset_event.set()

    clear_control_queues()

    # 回到固定初始位
    robot.movej_10(robot_init_pos.tolist(), 10, block=1)

    # 固定夹爪开口，避免每次起始夹爪状态不一样
    robot.set_gripper_position([0.7, 0.7])

    time.sleep(1.5)
    print("[INFO] 已到达标准初始位")
    reset_event.clear()

def ask_reset_robot_to_init(robot, robot_init_pos):
    print("[INFO] 左臂标准初始位:")
    print(robot_init_pos.tolist())
    answer = input("是否将左臂移动到标准初始位？输入 y/yes 确认，其他输入跳过: ").strip().lower()
    if answer in ("y", "yes"):
        reset_robot_to_init(robot, robot_init_pos)
        return True
    robot.set_gripper_position([0.7, 0.7])
    time.sleep(1.0)
    print("[INFO] 已跳过左臂初始化，仅初始化夹爪到 [0.7, 0.7]")
    return False

class RobotDataSender(threading.Thread):
    '''
    作为线程运行：读取并发送机器人状态数据到上位机
    '''
    def __init__(self, robot, high_host, joint_port):
        '''
        params:
            robot: RoboTG_g实例
            high_host: 上位机ip地址
            joint_port: 机器人状态数据发送端口
        '''
        super().__init__()
        self.robot = robot
        self.high_host = high_host 
        self.joint_port = joint_port
        
    def run(self):
        # 创建UDP socket
        joint_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            while not stop_event.is_set():
                # 读取机器人当前关节位置ssh
                robot_state = self.robot.get_robot_state()
                # if now_torso_pos.full():
                #     now_torso_pos.get()
                # now_torso_pos.put(robot_state[:3])  # 存储腰部当前关节位置用于腰部电机的控制（类似步进控制）
                
                # 数据处理部分，主要处理接受端（如数据采集时的外骨骼）角度定义与机器人定义不同的部分
                robot_state[9-3] -= 120
                # robot_state[16-3] -= 120
                robot_state[16-3] -= 120

                # 转换为弧度并添加时间戳
                joint_pose = [math.radians(degree) for degree in robot_state[:-1]]
                robot_state = joint_pose + robot_state[-1:] + [time.time()]
                state_data = np.float64(robot_state).tobytes()
                
                # 发送数据到上位机
                joint_socket.sendto(state_data, (self.high_host, self.joint_port))
        except (BrokenPipeError, ConnectionResetError) as e:
            print(f"Error: {e}.")
        finally:
            # Close sockets
            joint_socket.close()
            print("Sockets closed.")

def robot_control_listener(high_host, port, buffer_size, robot, robot_init_pos):
    """作为线程运行：监听并接收机械臂控制命令，放入队列中"""
    global stop_flag
    mode_two_locked = False
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.bind((high_host, port))
            print('等待控制命令数据...')
            while not stop_event.is_set():
                s.settimeout(1.0)  # 避免阻塞以便定期检查 stop_event
                try:
                    data, _ = s.recvfrom(buffer_size)
                    if data:
                        # 解析接收到的数据
                        robot_control = np.frombuffer(data, dtype=np.float64)
                
                        # 停止标志处理
                        robot_control_flag = robot_control[0]
                        if robot_control_flag == 1:
                            stop_flag = True
                            print("机器人控制通道接收到停止标志(1)")
                            break
                        if robot_control_flag == INIT_FLAG:
                            print("机器人控制通道接收到回初始位命令(9)")
                            reset_robot_to_init(robot, robot_init_pos)
                            continue
                        if robot_control_flag == 2:
                            if not mode_two_locked:
                                mode_two_locked = True
                                print("机器人控制通道进入2锁定模式，仅处理flag=2数据")
                        elif mode_two_locked:
                            continue
                        timestamp = robot_control[18]
                        robot_control = [math.degrees(rad) for rad in robot_control[1:18]]

                        # 数据处理部分，主要处理接受到的数据角度定义与机器人定义不同的部分
                        robot_control[9] += 120
                        # robot_control[16] += 120
                        # robot_control[9] += 180
                        robot_control[16] += 120
                        # 限制关节角度范围
                        robot_control = set_range(robot_control) 

                        # 放入队列
                        if data_queue.full():
                            data_queue.get()
                            time_queue.get()
                            # print("机器人数据队列已满，丢弃旧数据")
                        data_queue.put(robot_control)
                        time_queue.put(timestamp) 
                except socket.timeout:
                    continue
    except Exception as e:
        print(f"接收机械臂控制命令时发生错误: {e}")

def _clamp_gripper_ratio(value):
    return float(max(0.0, min(1.0, value)))


def gripper_control_listener(high_host, port, buffer_size):
    """作为线程运行：监听并接收夹爪控制命令，放入队列中。"""
    global stop_flag
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.bind((high_host, port))
            print(f'等待夹爪控制命令数据: {high_host}:{port}')
            while not stop_event.is_set():
                s.settimeout(1.0)  # 避免阻塞以便定期检查 stop_event
                try:
                    data, _ = s.recvfrom(buffer_size)
                    if data:
                        # 解析接收到的数据
                        gripper_control = np.frombuffer(data, dtype=np.float64)
                        print(
                            "[GRIPPER_RECV] "
                            f"flag={gripper_control[0]:.3f}, "
                            f"left_cmd={gripper_control[1]:.3f}, "
                            f"right_cmd={gripper_control[2]:.3f}",
                            flush=True,
                        )
                        
                        # 停止标志处理
                        gripper_control_flag = gripper_control[0]
                        if gripper_control_flag == 1:
                            stop_flag = True
                            print("夹爪控制通道接收到停止标志(1)")
                            break
                        if gripper_control_flag not in (0, 2):
                            print(f"夹爪控制通道忽略未知 flag={gripper_control_flag}")
                            continue
                        
                        # 处理数据并放入队列

                        gripper_control_sig = [
                            _clamp_gripper_ratio(gripper_control[1]),
                            _clamp_gripper_ratio(gripper_control[2]),
                        ]
                        if gripper_queue.full():
                            gripper_queue.get()
                            # print("夹爪数据队列已满，丢弃旧数据")
                        gripper_queue.put(gripper_control_sig)
                except socket.timeout:
                    continue
    except Exception as e:
        print(f"接收夹爪控制命令时发生错误: {e}")

def set_range(angle_list):
    for i, angle in enumerate(angle_list):
        if i == 9 or i == 16:  # 特殊处理9和16号关节
            angle_list[i] = (angle + 360) % 720 - 360
        else:
            angle_list[i] = (angle + 180) % 360 - 180
        if i == 1:
            angle_list[i] = limit_torso_pos_2(angle_list[i])  # 限制腰部位置
        elif i == 2:
            angle_list[i] = limit_torso_pos(angle_list[i])  # 限制腰部位置
    return angle_list

def limit_torso_pos(torso_pos):
    """
    限制腰部位置在[-20, 20]范围内
    :param torso_pos: 腰部位置
    :return: 限制后的腰部位置
    """
    torso_pos = max(-20, min(torso_pos, 20))
    return torso_pos

def limit_torso_pos_2(torso_pos):
    """
    限制腰部位置在[-15, 20]范围内
    :param torso_pos: 腰部位置
    :return: 限制后的腰部位置
    """
    torso_pos = max(-15, min(torso_pos, 20))
    return torso_pos

def main():
    robot_r_host = "192.168.1.17"  # 示例ip地址，替换为实际的机器人ip
    robot_l_host = "192.168.1.18"
    # gripper参数设置
    dxl = 'XM430'
    devicename = '/dev/ttyUSB0'
    baudrate = 4000000  # 波特率
    left_gripper_kwargs = {
        'dxl_name': dxl,  
        'Device_name': devicename,  
        'dxl_id': 1,  
        'baudrate': baudrate,  
        'min_position': 711,  # 闭合位置
        'max_position': 1773,  # 张开位置
    }
    # right_gripper_kwargs = {
    #     'dxl_name': dxl, 
    #     'Device_name': devicename,  
    #     'dxl_id': 1,  
    #     'baudrate': baudrate,  
    #     'min_position': 1380,536  # 闭合位置
    #     'max_position': 2271,  # 张开位置
    # }
    # left_gripper_kwargs = None
    right_gripper_kwargs = None
    robot = RoboTG_g(
        robot_r_host,
        robot_l_host,
        enable_joint=[False, False, False],
        port_r=8080,
        port_l=8080,
        left_gripper_kwargs=left_gripper_kwargs,
        right_gripper_kwargs=right_gripper_kwargs
        )
    
    robot_zero_pos = np.zeros(17, dtype=np.float32)
    robot_zero_pos[4] = -90
    robot_zero_pos[11] = 90
    robot_zero_pos[9] = 210
    robot_zero_pos[16] = 30 
    # robot_init_pos = np.array([0, 2.5, 0,  # 0 2.5 0   
    #            -29.6, -58.6, -52.6, -47.3, -45.7, -27.4, 210, 
    #            29.6, 58.6, 52.6, 47.3, 45.7, 27.4, 30], dtype=np.float32) 
    # robot_init_pos = np.array([0, 2.5, 0.0, 
    #            -29.6, -58.6, -52.6, -47.3, -45.7, -27.4, 210,
    #            29.6, 58.6, 52.6, 47.3, 45.7, 117.4, 120], dtype=np.float32)
    # 采集数据，推理测试的初始化位置 --单左臂
    robot_init_pos = np.array([0, 2.5, 0.0, 
            -103.0, -73.67, 89, -8.31, -22.0, -75.5, 76.4,
            9.8, 82.1, -14.9, 12.1, 2.59, -29.4, 200.6], dtype=np.float32)
    
    global stop_flag
    stop_flag = False

    #TODO 实际使用时修改上位机ip
    high_host = "192.168.1.12"
    local_host = "192.168.1.13"
    ee_port = 11235
    # joint_port = 18132
    joint_port = 18132
    control_port = 30005
    gripper_port = 30015

    arm_initialized = ask_reset_robot_to_init(robot, robot_init_pos)

    gripper_listener_thread = threading.Thread(
        target=gripper_control_listener,
        args=(local_host, gripper_port, 1024)
    )
    gripper_listener_thread.start()

    if arm_initialized:
        print("左臂和夹爪初始化完成，开始接收夹爪控制命令...")
    else:
        print("夹爪初始化完成，开始接收夹爪控制命令...")

    try:
        while True:
            if stop_flag:
                print("接收到停止信号，正在停止...")
                break
            if reset_event.is_set():
                time.sleep(0.01)
                continue
            handled = False
            if not gripper_queue.empty():
                gripper_control_data = gripper_queue.get()
                print(f"[GRIPPER_EXEC] set_gripper_position({gripper_control_data})", flush=True)
                robot.set_gripper_position(gripper_control_data)
                handled = True

            if not handled:
                time.sleep(0.01)
    except KeyboardInterrupt:
        print("夹爪控制中断，正在退出...")
    finally:
        stop_event.set() 
        robot.exit()
        gripper_listener_thread.join()
        print("夹爪控制程序已退出。")

if __name__ == "__main__":
    main()
