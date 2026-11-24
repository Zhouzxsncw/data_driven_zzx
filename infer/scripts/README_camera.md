# Camera Record Script

WHEELTEC C100/C70 摄像头录制脚本

## 依赖安装

```bash
pip install opencv-python
```

## 使用方法

```bash
# 查看可用摄像头设备号
python3 camera_record.py --list-devices

# 指定设备号开始录制（替换 X 为实际设备号）
python3 camera_record.py -i X

# 查看所有选项
python3 camera_record.py -h
```

## 按键

| 按键 | 功能 |
|------|------|
| 空格/r | 开始/停止录制 |
| q | 退出 |
| p | 截图 |
| 1/2/3 | 切换 1080p/720p/640x480 |

## 录制的视频

默认保存在 `~/Videos/camera_recordings/`
