#!/usr/bin/env python3
"""PICO 头显 -> ZMQ 发布端。

链路:
    PICO 头显 --(USB/WiFi)--> RoboticsServiceProcess (127.0.0.1:60061)
        --> xrobotoolkit_sdk (本进程)
        --> ZMQ PUB tcp://127.0.0.1:5565 --> 本机推理客户端

用法:
    bash run_pico_zzx.sh            # 60 Hz
    bash run_pico_zzx.sh --hz 90

Ctrl+C 退出（会调用 xrt.close() 释放设备）。
"""
import argparse
import json
import time

import zmq

try:
    import xrobotoolkit_sdk as xrt
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "导入 xrobotoolkit_sdk 失败，请通过 run_pico_zzx.sh 使用本目录的 PICO 环境\n"
        f"原始错误: {e}"
    )


def safe(fn, default=0.0):
    """SDK 某些接口在设备未连接时可能抛异常，统一兜底。"""
    try:
        v = fn()
        if v is None:
            return default
        if isinstance(v, (list, tuple)):
            return [float(x) for x in v]
        return float(v)
    except Exception:
        return default


def pose7(fn):
    """统一成 [x, y, z, qx, qy, qz, qw]。"""
    v = safe(fn, [0.0] * 7)
    if not isinstance(v, list) or len(v) < 7:
        return [0.0] * 7
    return v[:7]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5565)
    ap.add_argument("--hz", type=float, default=60.0)
    ap.add_argument("--quiet", action="store_true", help="不打状态行")
    args = ap.parse_args()

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.bind(f"tcp://127.0.0.1:{args.port}")

    xrt.init()
    period = 1.0 / args.hz
    seq = 0
    zero_frames = 0
    warned = False
    last_report = time.time()
    sent = 0

    print(f"PUB tcp://127.0.0.1:{args.port}  {args.hz:.0f}Hz   Ctrl+C 退出")
    try:
        while True:
            t0 = time.perf_counter()

            hs = pose7(xrt.get_headset_pose)
            rc = pose7(xrt.get_right_controller_pose)
            lc = pose7(xrt.get_left_controller_pose)

            msg = {
                "seq": seq,
                "timestamp_ns": int(safe(xrt.get_time_stamp_ns, 0.0)),
                "headset_pose": hs,
                "right_controller": {
                    "position": rc[:3],
                    "orientation": rc[3:7],
                    "trigger": safe(xrt.get_right_trigger),
                    "grip": safe(xrt.get_right_grip),
                    "axis": safe(xrt.get_right_axis, [0.0, 0.0]),
                },
                "left_controller": {
                    "position": lc[:3],
                    "orientation": lc[3:7],
                    "trigger": safe(xrt.get_left_trigger),
                    "grip": safe(xrt.get_left_grip),
                    "axis": safe(xrt.get_left_axis, [0.0, 0.0]),
                },
                "buttons": {
                    "A": bool(safe(xrt.get_A_button)),
                    "B": bool(safe(xrt.get_B_button)),
                    "X": bool(safe(xrt.get_X_button)),
                    "Y": bool(safe(xrt.get_Y_button)),
                },
            }
            sock.send_string(json.dumps(msg))
            seq += 1
            sent += 1

            # 设备健康检查：姿态长时间全 0 说明头显没在推流
            if all(abs(v) < 1e-9 for v in hs):
                zero_frames += 1
                if zero_frames == 60 and not warned:
                    warned = True
                    print("⚠️  头显姿态全 0（设备没在推流）：检查头显是否休眠/切后台、"
                          "应用是否在前台、服务端是否认到设备")
            else:
                if warned:
                    print("✅ 收到头显数据，恢复正常发布")
                warned = False
                zero_frames = 0

            if not args.quiet and time.time() - last_report >= 2.0:
                last_report = time.time()
                print(f"  seq={seq:<6d} headset={['%.2f' % v for v in hs[:3]]} "
                      f"right={['%.2f' % v for v in rc[:3]]}")
                sent = 0

            dt = time.perf_counter() - t0
            if period > dt:
                time.sleep(period - dt)
    except KeyboardInterrupt:
        print("\n中断，正在收尾…")
    finally:
        try:
            xrt.close()
        finally:
            sock.close(0)
            ctx.term()


if __name__ == "__main__":
    main()
