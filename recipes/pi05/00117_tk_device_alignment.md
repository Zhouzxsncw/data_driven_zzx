# infer-tk 现场通信记录（2026-10-29）

SSH：`tk@10.151.91.152`。机器人局域网上位机为 `192.168.1.12`，到下位机/机械臂走 `enp128s31f6`。不能用办公网 IP 替代设备回传目标。

现场实际采集进程使用 `/home/tk/infer/config/dual_arm_gripper_2cam.yaml`，不是同名 `data_collect/config/` 文件。新推理代码根为 `/home/tk/data_Driven/infer`。

| 信号 | 通道方向 | 当前用途/证据 |
|---|---|---|
| 采集遥操作 action | 下位机 → `192.168.1.12:18131` UDP | 采集 action 输入，不是模型控制输出；当前采集进程占用 |
| 左/右机械臂状态与控制 | 上位机 → 左 `192.168.1.18:8080` / 右 `192.168.1.17:8080` SDK | 117 direct_robot 模式；网络可达和 SDK 导入通过，未验证控制 |
| 机械臂速度推送 | 两臂 → 上位机 `8098` UDP | 部署代码 JOINT_SPEED_PUSH_PORT；待实际订阅验证 |
| 夹爪控制 | 上位机 → `192.168.1.13:30015` UDP | 下位机舵机后端；未下发命令 |
| 夹爪状态 | 下位机 → `192.168.1.12:30016` UDP | 曾收到实测报文；目前采集进程占用 |
| 外骨骼状态 | 下位机 → `192.168.1.12:18140` UDP | 先前短时监听无包；接管未验证 |
| 外骨骼模式/控制 | 上位机 → `192.168.1.13:30026` UDP | 配置/代码核对；未发送验证 |
| 外骨骼对齐目标 | 上位机 → `192.168.1.13:30027` UDP | align_target_port；未发送验证 |
| 旧机械臂状态/命令 | 上位机 `18132` / 下位机 `30005` UDP | 117 direct_robot 不使用；不要与 18131 混用 |
| 相机 | RealSense 本地 USB | transport=local，不使用旧网络相机 TCP 18888 |
| 模型服务 | 客户端 → `127.0.0.1:1297` WebSocket | 117 step001302，已验证三次 `(50,16)` 有限值动作 |

物理相机与采集顺序：left=`f1422917`、front/cam_mid=`f0263949`、right=`f1370871`。现场实际采集三路 RGB 均为 `960×540@30`。此前取帧验证了前/左 `960×540`、右 `640×480`；右设备枚举支持 `960×540`，改为与采集一致后需在采集停止、相机空闲时验证三路同时取帧。

采集与推理 devices 必须都为 left → mid → right：接管共享图像缓冲并按下标解释。policy.camera_keys 仍可为 front → left → right，通过键名映射送模型。

117 权重：16D `[left7,right7,gripperL,gripperR]`、horizon=50、训练 FPS=15、delay=+0.2s、delta mask=`[7,7,-1,-1]`、不 interleave。SDK：`/home/tk/RM_API2-main/Python`。

设备专用配置：

- 推理：`/home/tk/data_Driven/infer/infer/config/lerobot/pi05_00117_step001302_tk_ws.yaml`
- 接管采集：`/home/tk/data_Driven/infer/data_collect/config/dual_arm_gripper_2cam_tk.yaml`
- 启动：`/home/tk/data_Driven/recipes/pi05/run_00117_tk.sh`

发现现场采集正在运行时，不再打开相机或启动第二套机械臂控制。下位机默认 SSH 登录未成功，未因此改动服务；夹爪反馈可用不代表外骨骼接管全部可用。
