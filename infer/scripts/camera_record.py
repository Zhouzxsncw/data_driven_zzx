#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WHEELTEC C100/C70 摄像头录制脚本
支持实时预览、按键控制录制/停止、分辨率切换

用法:
    python3 camera_record.py                      # 默认 /dev/video16, 1080p
    python3 camera_record.py -i 16                # 指定设备号
    python3 camera_record.py -r 720p              # 720p 录制
    python3 camera_record.py -r 640x480           # 640x480 录制
    python3 camera_record.py -o /path/to/output   # 指定输出目录
    python3 camera_record.py --no-preview         # 无预览模式（纯录制）

按键说明（预览窗口激活时）:
    r / 空格  — 开始/停止录制
    q         — 退出程序
    p         — 截图保存当前帧
    1         — 切换到 1080p
    2         — 切换到 720p
    3         — 切换到 640x480
    h         — 显示帮助
"""

import cv2
import os
import sys
import time
import argparse
import signal
import subprocess
from collections import deque
from datetime import datetime


# ============================================================
# 配置
# ============================================================
DEFAULT_DEVICE = 16
DEFAULT_OUTPUT_DIR = os.path.expanduser("~/Videos/camera_recordings")
DEFAULT_RESOLUTION = "1080p"
DEFAULT_FPS = 30
DEFAULT_FOURCC = "MJPG"   # C100/C70 推荐 MJPEG
OUTPUT_FOURCC = "mp4v"    # 输出编码: mp4v / avc1 / X264

RESOLUTION_MAP = {
    "1080p":   (1920, 1080),
    "720p":    (1280, 720),
    "640x480": (640, 480),
}


def get_timestamp():
    """生成时间戳字符串"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def detect_camera_devices():
    """检测系统中的摄像头设备"""
    try:
        result = subprocess.run(['ls', '/dev/video*'],
                                capture_output=True, text=True)
        devices = result.stdout.strip().split('\n')
        video_devices = [d for d in devices if d.startswith('/dev/video')]
        return sorted(video_devices)
    except Exception:
        return []


def check_v4l2_info(device_num):
    """通过 v4l2-ctl 检查设备信息"""
    dev_path = f"/dev/video{device_num}"
    print(f"\n{'='*60}")
    print(f"检查设备: {dev_path}")
    print(f"{'='*60}")

    if not os.path.exists(dev_path):
        print(f"✗ 设备 {dev_path} 不存在")
        return False

    # 检查权限
    if not os.access(dev_path, os.R_OK | os.W_OK):
        print(f"✗ 设备 {dev_path} 没有读写权限")
        print(f"  尝试修复: sudo chmod 666 {dev_path}")
        return True  # 不阻止继续尝试

    # v4l2-ctl 获取信息
    try:
        result = subprocess.run(
            ['v4l2-ctl', '-d', dev_path, '--all'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            print(result.stdout[:1500])
        else:
            print(f"v4l2-ctl 无法获取信息 (可能未安装): apt install v4l-utils")
    except FileNotFoundError:
        print("v4l2-ctl 未安装 (可选): apt install v4l-utils")
    except Exception as e:
        print(f"v4l2-ctl 错误: {e}")

    return True


class CameraRecorder:
    """摄像头录制器"""

    def __init__(
        self,
        device_id=DEFAULT_DEVICE,
        resolution=DEFAULT_RESOLUTION,
        fps=DEFAULT_FPS,
        output_dir=DEFAULT_OUTPUT_DIR,
        show_preview=True,
        camera_fourcc=DEFAULT_FOURCC,
        output_fourcc=OUTPUT_FOURCC,
        flip_horizontal=False,
        flip_vertical=False,
    ):
        self.device_id = device_id
        self.device_path = f"/dev/video{device_id}"
        self.resolution = resolution
        self.target_width, self.target_height = RESOLUTION_MAP.get(
            resolution, RESOLUTION_MAP["1080p"]
        )
        self.target_fps = fps          # 用户配置的目标/上限帧率
        self.output_dir = output_dir
        self.show_preview = show_preview
        self.camera_fourcc = camera_fourcc
        self.output_fourcc = output_fourcc
        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical

        # 状态
        self.cap = None
        self.writer = None
        self.is_recording = False
        self.recording_start_time = None
        self.frame_count = 0
        self.recording_frame_count = 0
        self.current_output_file = None
        self.actual_width = 0
        self.actual_height = 0
        self.running = False

        # ---- 实时帧率测量（解决视频加速问题的关键） ----
        # 记录最近 N 帧的时间戳，用于计算实际采集帧率
        self._frame_times = deque(maxlen=30)
        self._measured_fps = float(fps)  # 实测采集帧率，初始用目标值

        # FPS 显示统计（每秒刷新）
        self._fps_timer = time.time()
        self._fps_counter = 0
        self._display_fps = 0.0

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

    # ----------------------------------------------------------
    # 相机操作
    # ----------------------------------------------------------

    def open_camera(self):
        """打开摄像头"""
        print(f"\n正在打开摄像头 {self.device_path} ...")

        # 使用 V4L2 后端（兼容性更好）
        self.cap = cv2.VideoCapture(self.device_id, cv2.CAP_V4L2)

        if not self.cap.isOpened():
            # 回退到默认后端
            self.cap = cv2.VideoCapture(self.device_id)
            if not self.cap.isOpened():
                raise RuntimeError(f"无法打开摄像头 {self.device_path}")

        # 设置 MJPEG 格式（C100/C70 推荐）
        fourcc_code = cv2.VideoWriter_fourcc(*self.camera_fourcc)
        self.cap.set(cv2.CAP_PROP_FOURCC, fourcc_code)

        # 设置分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)

        # 向摄像头请求目标帧率
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        # ---- 预热 + 实测帧率 ----
        print("  预热中，测量实际帧率...")
        warmup_start = time.time()
        for _ in range(30):
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.actual_height, self.actual_width = frame.shape[:2]
                # 记录时间戳用于测帧率
                self._frame_times.append(time.time())
            time.sleep(0.02)
        warmup_elapsed = time.time() - warmup_start

        # 用预热期间的帧时间戳估算实际帧率
        if len(self._frame_times) >= 2:
            dt = self._frame_times[-1] - self._frame_times[0]
            if dt > 0:
                self._measured_fps = (len(self._frame_times) - 1) / dt
        print(f"  预热完成 ({warmup_elapsed:.1f}s), "
              f"{len(self._frame_times)} 帧")

        # 获取驱动报告的参数
        reported_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        print(f"✓ 摄像头已就绪")
        print(f"  设备:       {self.device_path}")
        print(f"  分辨率:     {self.actual_width:.0f}x{self.actual_height:.0f}")
        print(f"  驱动报告帧率: {reported_fps:.1f}")
        print(f"  实测采集帧率: {self._measured_fps:.1f} fps  ← 输出视频将用此帧率")
        print(f"  编码:       {self.camera_fourcc}")
        print(f"  输出目录:   {self.output_dir}")
        print(f"  预览:       {'开启' if self.show_preview else '关闭'}")

    def read_frame(self):
        """读取一帧，同时更新实测帧率"""
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None

        # 翻转处理
        if self.flip_horizontal and self.flip_vertical:
            frame = cv2.flip(frame, -1)
        elif self.flip_horizontal:
            frame = cv2.flip(frame, 1)
        elif self.flip_vertical:
            frame = cv2.flip(frame, 0)

        self.frame_count += 1

        # ---- 记录帧时间戳，持续更新实测帧率 ----
        now = time.time()
        self._frame_times.append(now)
        if len(self._frame_times) >= 2:
            dt = self._frame_times[-1] - self._frame_times[0]
            if dt > 0.5:  # 至少有 0.5 秒的数据才更新
                self._measured_fps = (len(self._frame_times) - 1) / dt

        # 显示用 FPS（每秒更新一次）
        self._fps_counter += 1
        elapsed = now - self._fps_timer
        if elapsed >= 1.0:
            self._display_fps = self._fps_counter / elapsed
            self._fps_counter = 0
            self._fps_timer = now

        return frame

    # ----------------------------------------------------------
    # 录制操作
    # ----------------------------------------------------------

    def start_recording(self):
        """开始录制，使用实测帧率作为输出帧率"""
        if self.is_recording:
            print("已经在录制中")
            return

        timestamp = get_timestamp()
        filename = f"recording_{timestamp}_{self.resolution}.mp4"
        self.current_output_file = os.path.join(self.output_dir, filename)

        fourcc = cv2.VideoWriter_fourcc(*self.output_fourcc)
        frame_size = (int(self.actual_width), int(self.actual_height))

        # ---- 关键修复：用实测帧率而不是目标帧率 ----
        # 确保输出帧率至少为 1，最多不超过目标帧率
        output_fps = max(1.0, min(self._measured_fps, self.target_fps))
        output_fps = round(output_fps, 1)

        self.writer = cv2.VideoWriter(
            self.current_output_file, fourcc, output_fps, frame_size
        )

        if not self.writer.isOpened():
            print(f"✗ 无法创建视频文件: {self.current_output_file}")
            self.writer = None
            return

        self.is_recording = True
        self.recording_start_time = time.time()
        self.recording_frame_count = 0
        self._output_fps = output_fps   # 记录本次录制用的输出帧率

        print(f"\n● 开始录制: {self.current_output_file}")
        print(f"  分辨率: {frame_size[0]}x{frame_size[1]}")
        print(f"  实测采集帧率: {self._measured_fps:.1f} fps")
        print(f"  输出视频帧率: {output_fps} fps  ← 匹配实际采集速率")
        print(f"  按空格键或 'r' 键停止录制")

    def stop_recording(self):
        """停止录制"""
        if not self.is_recording:
            return

        self.is_recording = False

        if self.writer:
            self.writer.release()
            self.writer = None

        elapsed = time.time() - self.recording_start_time
        file_size_mb = 0
        if self.current_output_file and os.path.exists(self.current_output_file):
            file_size_mb = os.path.getsize(self.current_output_file) / (1024 * 1024)

        actual_record_fps = self.recording_frame_count / elapsed if elapsed > 0 else 0
        output_fps = getattr(self, '_output_fps', self.target_fps)

        print(f"\n■ 停止录制")
        print(f"  文件: {self.current_output_file}")
        print(f"  时长: {elapsed:.1f} 秒")
        print(f"  帧数: {self.recording_frame_count}")
        print(f"  大小: {file_size_mb:.1f} MB")
        print(f"  实际写入帧率: {actual_record_fps:.1f} fps")
        print(f"  输出文件帧率: {output_fps} fps")
        if abs(actual_record_fps - output_fps) / max(output_fps, 1) < 0.15:
            print(f"  ✓ 帧率匹配，播放速度正常")
        else:
            print(f"  ⚠ 帧率偏差较大，播放可能略有不均")
        print(f"  按空格键或 'r' 键开始新的录制")

    def toggle_recording(self):
        """切换录制状态"""
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def capture_screenshot(self, frame):
        """保存截图"""
        timestamp = get_timestamp()
        filename = f"screenshot_{timestamp}.jpg"
        filepath = os.path.join(self.output_dir, filename)
        cv2.imwrite(filepath, frame)
        print(f"\n📷 截图已保存: {filepath}")

    # ----------------------------------------------------------
    # 分辨率切换
    # ----------------------------------------------------------

    def switch_resolution(self, res_name):
        """切换分辨率"""
        if res_name not in RESOLUTION_MAP:
            print(f"未知分辨率: {res_name}")
            return

        # 停止录制
        if self.is_recording:
            self.stop_recording()

        self.resolution = res_name
        self.target_width, self.target_height = RESOLUTION_MAP[res_name]

        print(f"\n切换分辨率到 {res_name} ({self.target_width}x{self.target_height})...")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
        time.sleep(0.3)

        # 清空帧时间戳缓存（分辨率变化后需重新测量）
        self._frame_times.clear()

        # 稳定帧 + 重新测量帧率
        for _ in range(30):
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.actual_height, self.actual_width = frame.shape[:2]
                self._frame_times.append(time.time())
            time.sleep(0.02)

        if len(self._frame_times) >= 2:
            dt = self._frame_times[-1] - self._frame_times[0]
            if dt > 0:
                self._measured_fps = (len(self._frame_times) - 1) / dt

        print(f"✓ 实际分辨率: {self.actual_width:.0f}x{self.actual_height:.0f}")
        print(f"  实测帧率: {self._measured_fps:.1f} fps")

    # ----------------------------------------------------------
    # 预览叠加信息
    # ----------------------------------------------------------

    def draw_overlay(self, frame):
        """在帧上绘制状态信息"""
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 2
        color = (0, 255, 0)
        red = (0, 0, 255)
        yellow = (0, 255, 255)

        y = 22
        dy = 20

        # 设备 + 分辨率
        cv2.putText(frame, f"Device: {self.device_path}  |  {w}x{h}",
                    (10, y), font, font_scale, color, thickness)
        y += dy

        # 实测帧率
        cv2.putText(frame, f"Capture FPS: {self._display_fps:.1f} (measured: {self._measured_fps:.1f})",
                    (10, y), font, font_scale, yellow, thickness)
        y += dy

        # 录制状态
        if self.is_recording:
            elapsed = time.time() - self.recording_start_time
            mins = int(elapsed // 60)
            secs = int(elapsed % 60)
            out_fps = getattr(self, '_output_fps', self._measured_fps)
            cv2.putText(frame, f"● REC {mins:02d}:{secs:02d}  "
                        f"Frames: {self.recording_frame_count}  "
                        f"Output: {out_fps} fps",
                        (10, y), font, font_scale * 1.05, red, thickness + 1)

            # 录制指示灯
            if int(time.time() * 2) % 2 == 0:
                cv2.circle(frame, (w - 25, 22), 9, (0, 0, 255), -1)
            y += dy
        else:
            cv2.putText(frame, "○ STANDBY (SPACE/r to record)",
                        (10, y), font, font_scale, (200, 200, 200), thickness)
            y += dy

        # 操作提示
        cv2.putText(frame, "SPACE/r:Record | q:Quit | p:Screenshot | 1/2/3:Res | h:Help",
                    (10, h - 10), font, 0.4, (255, 255, 0), 1)

        return frame

    # ----------------------------------------------------------
    # 帮助信息
    # ----------------------------------------------------------

    @staticmethod
    def print_help():
        """打印帮助信息"""
        print(f"\n{'='*60}")
        print("按键说明（预览窗口激活时）")
        print(f"{'='*60}")
        print("  空格 / r  — 开始 / 停止录制")
        print("  q         — 退出程序")
        print("  p         — 截图保存当前帧")
        print("  1         — 切换到 1080p")
        print("  2         — 切换到 720p")
        print("  3         — 切换到 640x480")
        print("  h         — 显示本帮助")
        print(f"{'='*60}")

    # ----------------------------------------------------------
    # 主循环
    # ----------------------------------------------------------

    def run(self):
        """运行主循环"""
        self.running = True

        print(f"\n{'='*60}")
        print(f"WHEELTEC C100/C70 摄像头录制工具")
        print(f"{'='*60}")
        self.print_help()
        print("\n提示: 按 空格键 或 r键 开始录制，按 q键 退出")
        print("输出视频帧率会自动匹配实际采集帧率，避免播放加速\n")

        # 打开摄像头
        self.open_camera()

        if self.show_preview:
            cv2.namedWindow('Camera Recorder', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Camera Recorder', 960, 640)

        signal.signal(signal.SIGINT, self._signal_handler)

        try:
            while self.running:
                frame = self.read_frame()

                if frame is None:
                    print("✗ 读取帧失败，尝试重连...")
                    time.sleep(2)
                    self.cap.release()
                    self.cap = cv2.VideoCapture(self.device_id, cv2.CAP_V4L2)
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
                    continue

                # 录制中写入帧
                if self.is_recording and self.writer:
                    self.writer.write(frame)
                    self.recording_frame_count += 1

                # 预览模式
                if self.show_preview:
                    display_frame = self.draw_overlay(frame.copy())
                    cv2.imshow('Camera Recorder', display_frame)

                    key = cv2.waitKey(1) & 0xFF

                    # ---- 按键处理 ----
                    if key == ord('q'):
                        self.running = False
                    elif key == ord('r') or key == 32:  # r 或 空格
                        self.toggle_recording()
                    elif key == ord('p'):
                        self.capture_screenshot(frame)
                    elif key == ord('1'):
                        self.switch_resolution("1080p")
                    elif key == ord('2'):
                        self.switch_resolution("720p")
                    elif key == ord('3'):
                        self.switch_resolution("640x480")
                    elif key == ord('h'):
                        self.print_help()

        except KeyboardInterrupt:
            print("\n\n收到中断信号，正在退出...")
        finally:
            self.cleanup()

    def _signal_handler(self, signum, frame):
        """信号处理"""
        print("\n收到中断信号...")
        self.running = False

    def cleanup(self):
        """清理资源"""
        print("正在清理...")
        self.running = False

        if self.is_recording:
            self.stop_recording()

        if self.cap:
            self.cap.release()
            self.cap = None

        cv2.destroyAllWindows()

        # 列出本次录制的文件
        print(f"\n输出目录: {self.output_dir}")
        if os.path.exists(self.output_dir):
            files = sorted(
                [f for f in os.listdir(self.output_dir)
                 if f.endswith('.mp4') or f.endswith('.jpg')],
                key=lambda f: os.path.getmtime(os.path.join(self.output_dir, f)),
                reverse=True
            )
            if files:
                print("最近的文件:")
                for f in files[:10]:
                    fpath = os.path.join(self.output_dir, f)
                    size_mb = os.path.getsize(fpath) / (1024 * 1024)
                    mtime = datetime.fromtimestamp(os.path.getmtime(fpath))
                    print(f"  {f}  ({size_mb:.1f} MB, {mtime.strftime('%Y-%m-%d %H:%M:%S')})")

        print("✓ 已退出")


# ============================================================
# 命令行入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="WHEELTEC C100/C70 摄像头录制脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python3 camera_record.py                          # 默认 /dev/video16, 1080p
  python3 camera_record.py -i 0                     # 指定 /dev/video0
  python3 camera_record.py -r 720p                  # 720p 录制
  python3 camera_record.py -r 640x480               # 640x480 录制
  python3 camera_record.py -o ~/my_videos           # 自定义输出目录
  python3 camera_record.py --no-preview             # 无预览纯录制模式
  python3 camera_record.py --list-devices           # 列出所有摄像头设备
  python3 camera_record.py --flip-horizontal        # 水平翻转画面
        """,
    )

    parser.add_argument(
        '-i', '--id', type=int, default=DEFAULT_DEVICE,
        help=f'摄像头设备号 (默认: {DEFAULT_DEVICE}，即 /dev/video{DEFAULT_DEVICE})'
    )
    parser.add_argument(
        '-r', '--resolution', type=str, default=DEFAULT_RESOLUTION,
        choices=['1080p', '720p', '640x480'],
        help=f'录制分辨率 (默认: {DEFAULT_RESOLUTION})'
    )
    parser.add_argument(
        '-f', '--fps', type=int, default=DEFAULT_FPS,
        help=f'目标帧率上限，输出文件会自动匹配实测采集帧率 (默认: {DEFAULT_FPS})'
    )
    parser.add_argument(
        '-o', '--output', type=str, default=DEFAULT_OUTPUT_DIR,
        help=f'输出目录 (默认: {DEFAULT_OUTPUT_DIR})'
    )
    parser.add_argument(
        '--no-preview', action='store_true',
        help='关闭实时预览窗口（纯录制模式）'
    )
    parser.add_argument(
        '--flip-horizontal', action='store_true',
        help='水平翻转画面'
    )
    parser.add_argument(
        '--flip-vertical', action='store_true',
        help='垂直翻转画面'
    )
    parser.add_argument(
        '--camera-fourcc', type=str, default=DEFAULT_FOURCC,
        help=f'摄像头采集编码 (默认: {DEFAULT_FOURCC})'
    )
    parser.add_argument(
        '--output-fourcc', type=str, default=OUTPUT_FOURCC,
        help=f'输出视频编码 (默认: {OUTPUT_FOURCC})'
    )
    parser.add_argument(
        '--list-devices', action='store_true',
        help='列出所有摄像头设备后退出'
    )

    args = parser.parse_args()

    # 列出设备
    if args.list_devices:
        devices = detect_camera_devices()
        if devices:
            print("找到以下摄像头设备:")
            for dev in devices:
                print(f"  {dev}")
        else:
            print("未找到任何 /dev/video* 设备")
        return

    # 检查设备信息
    check_v4l2_info(args.id)

    # 输出目录
    output_dir = os.path.expanduser(args.output)

    # 创建录制器并运行
    recorder = CameraRecorder(
        device_id=args.id,
        resolution=args.resolution,
        fps=args.fps,
        output_dir=output_dir,
        show_preview=not args.no_preview,
        camera_fourcc=args.camera_fourcc,
        output_fourcc=args.output_fourcc,
        flip_horizontal=args.flip_horizontal,
        flip_vertical=args.flip_vertical,
    )
    recorder.run()


if __name__ == "__main__":
    main()
