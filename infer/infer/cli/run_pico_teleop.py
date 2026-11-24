"""Standalone PICO control of the RM75 arms and grippers."""

from __future__ import annotations

import argparse
import signal
import socket
import time
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

from infer.config_loader import load_infer_config
from infer.core.pico_recovery import PicoMode, PicoRecovery
from infer.io.direct_robot_io import DirectTongGenesisRobot
from infer.io.pico_gripper import PicoGripper
from infer.io.pico_input import PicoSubscriber


def follow_arms(robot, subscriber, gripper=None) -> None:
    recovery = PicoRecovery()
    teleop = None
    # PICO 输入接收线程持续保留最新帧，这里用固定周期消费，避免
    # 通过低频 sleep 人为增加控制延迟。实际周期由 SDK 下发耗时决定。
    control_period = 0.01  # 100 Hz target loop
    next_tick = time.perf_counter()
    try:
        while True:
            now = time.perf_counter()
            if now < next_tick:
                time.sleep(next_tick - now)
            next_tick += control_period
            # 处理偶发的 SDK 或系统调度抖动，避免落后后连续空转追赶。
            if next_tick < time.perf_counter() - control_period:
                next_tick = time.perf_counter() + control_period
            status = subscriber.poll()
            previous = recovery.mode
            recovery.observe(status, allowed=True, wall_time=time.time())
            if recovery.mode is PicoMode.RETURNING:
                recovery.hold()
            if recovery.mode is not previous:
                print(f"[PICO] {previous.value} -> {recovery.mode.value}", flush=True)
                if PicoMode.MANUAL in (previous, recovery.mode) or PicoMode.INTERVENING in (previous, recovery.mode):
                    robot.stop_pico_motion()
                teleop = None
                if gripper is not None:
                    gripper.reset()
            frame = recovery.frame
            if frame is not None and (teleop is None or teleop.sequence != frame.sequence):
                def tracking_is_fresh():
                    return subscriber.monitor.status().frame is frame

                teleop, _ = robot.pico_motion_step(
                    teleop,
                    frame,
                    tracking_is_fresh,
                    active_sides=recovery.active_sides,
                )
                if gripper is not None:
                    try:
                        goals = gripper.tick(frame, tracking_is_fresh)
                    except RuntimeError as exc:
                        print(f"[PICO] {exc}，双臂已请求保持，按 X 重新接管。", flush=True)
                        recovery.hold()
                        robot.stop_pico_motion()
                        teleop = None
                        gripper.reset()
                    else:
                        if goals is not None:
                            print(f"[PICO] 左右夹爪目标 {goals}", flush=True)
    finally:
        if recovery.mode in (PicoMode.MANUAL, PicoMode.INTERVENING):
            robot.stop_pico_motion()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_infer_config(args.config)
    if not cfg.direct_robot.enabled:
        parser.error("Standalone PICO control requires direct_robot.enabled")

    settings = cfg.pico_recovery
    settings.validate_grippers(cfg.direct_robot.gripper_mode)
    use_grippers = settings.gripper_open is not None
    use_udp = use_grippers and cfg.direct_robot.gripper_mode == "udp"
    robot_cfg = cfg.direct_robot if use_grippers else replace(cfg.direct_robot, gripper_mode="none")
    robot = DirectTongGenesisRobot(
        robot_cfg,
        gripper_state_port=cfg.network.gripper_state_port if use_udp else None,
        gripper_state_stale_sec=cfg.network.gripper_state_stale_sec,
    )

    def stop(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        with ExitStack() as stack:
            subscriber = PicoSubscriber(settings.endpoint, settings.timeout_sec)
            stack.callback(subscriber.close)
            udp_socket = stack.enter_context(socket.socket(socket.AF_INET, socket.SOCK_DGRAM)) if use_udp else None
            udp_address = (cfg.network.low_host, cfg.network.gripper_port) if use_udp else None
            gripper = PicoGripper(settings, robot, udp_socket, udp_address) if use_grippers else None
            stack.callback(robot.close)
            robot.connect()
            print("[PICO] X 接管，Y 保持，Ctrl+C 退出。", flush=True)
            if gripper is not None:
                print("[PICO] 按住对应侧食指扳机闭合，松开张开。", flush=True)
            follow_arms(robot, subscriber, gripper)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
