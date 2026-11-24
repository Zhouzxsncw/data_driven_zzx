# cigai22 主机 PICO 双臂遥操与推理接管

部署目录：/home/cigai22/data_Driven_zzx。PICO PC Service 和发布器使用独立 Docker 镜像 data-driven-zzx/pico:1.0.0；推理 Python 使用已有 Conda 环境 openpi-pi05-infer。

## 三个启动入口

### 1. 独立 PICO 遥操

```bash
cd /home/cigai22/data_Driven_zzx
bash run_pico_teleop_zzx.sh
```

头显和左右手柄追踪正常后，单按 **X** 开始遥操，单按 **Y** 暂停并保持当前位置，再按 X 可继续。按住对应手柄的食指扳机闭合夹爪，松开扳机张开夹爪。先空载、小幅验证双臂和夹爪，结束时按 Ctrl+C。

推理接管模式增加单臂纠偏。按住左手柄 **Grip** 只接管左臂，按住右手柄 **Grip** 只接管右臂，另一侧继续执行策略动作；松开 Grip 后该侧交回策略。Grip 不驱动夹爪，夹爪仍由对应手柄的食指扳机控制。X/Y 继续保留双臂接管和交回机制。

### 2. PI05 推理服务端

服务端和客户端读取同一份配置 `infer/infer/config/lerobot/pi05_current_tk_ws.yaml`，两端使用的 ckpt 都取自该文件的 `policy.ckpt_dir`，换模型只改这一处，服务端与客户端自动保持一致。

当前模型（3 号盘 T9_3）：`/media/cigai22/T9_3/checkpoint/mix_108_36_91_73_112_PI05_cty18_bs768_ah50_fps15_d0p2_action_cty18_step007379/pretrained_model`（混合 00108/00036/00091/00073/00112，ah50，15Hz）。启动前确认 T9_3 已挂载，且该目录下有 `model.safetensors`，同级目录有 `assets/norm_stats.json`。然后单独打开一个终端启动服务端：

```bash
cd /home/cigai22/data_Driven_zzx
bash recipes/pi05/run_current_tk.sh server
```

服务端默认监听配置中的 `127.0.0.1:1297`。看到服务进入 listening 状态后，保持这个终端运行。服务端启动不会自动启动 PICO 遥操。

### 3. 推理客户端和人工接管

服务端出现 listening 后，再打开一个终端启动接管客户端：

```bash
cd /home/cigai22/data_Driven_zzx
bash recipes/pi05/run_current_tk.sh client --pico-recovery
```

客户端连接服务端并启动相机、双臂推理。运行中单按 **X** 接管，人工修正后单按 **Y** 交回，无需长按。交回后依据最新观测重新推理，不续播旧动作块。夹爪操作仍为按住扳机闭合、松开张开。独立遥操不能和推理客户端同时运行。

不需要人工接管时使用：

```bash
bash recipes/pi05/run_current_tk.sh client
```

## 部署说明

- 相机序列号已按目标机配置：f1480854、f1420947、f1380315。
- RM 网络地址沿用：192.168.1.12、192.168.1.13。
- 推理配置：infer/infer/config/lerobot/pi05_current_tk_ws.yaml（服务端与客户端共用，模型只在此一处指定）。当前模型为 /media/cigai22/T9_3/checkpoint/mix_108_36_91_73_112_PI05_cty18_bs768_ah50_fps15_d0p2_action_cty18_step007379/pretrained_model，位于 3 号盘 T9_3，已确认 model.safetensors 与 assets/norm_stats.json 齐全。该模型对应的原 data_Driven 配置使用 init_pose_source: standard，不使用 dataset_root，标准初始位已同步到配置。模型 action horizon 和客户端动作步数均为 50。默认 prompt 为 108（香蕉柠檬/青菜分盒），batch_infer.task_prompt_presets 内已备好该混合模型覆盖的 073/091/108/112 四句，换任务只需切预设。
- PICO 日志：pico/state/pc_service.log、pico/state/publisher.log；发布端口 5565，PC Service 端口 60061、63901。

## 迁移后的离线检查记录

2026-11-19 在 cigai22 主机完成以下检查，过程中没有启动模型服务、PICO 服务或机器人控制：

- 307 个 YAML 文件解析通过。
- `infer` 和 `recipes` Python 文件编译通过。
- PICO Docker 镜像内 `zmq`、`xrobotoolkit_sdk` 导入通过。
- `vr_pub.py --help` 和独立遥操入口帮助信息通过。
- 所有迁移入口脚本通过 `bash -n`，RM 地址和目标机相机序列号已写入当前配置。
- 下位机 `.13` 的夹爪和外骨骼服务已确认运行，夹爪 UDP 端口为 `30015`，外骨骼端口为 `30026` 和 `30027`。

当前仍有两项代码检查问题：

- `recipes/gr00t_sonic/collect.sh` 在第 27 行存在 Bash 语法错误。这是迁移代码中已有的独立脚本问题，不影响当前 PI05/PICO 入口。
- 关键 pytest 文件因仓库的双层 `infer` 包目录结构发生导入遮蔽，无法直接收集 `infer/infer/tests`。需要后续整理测试启动方式或包结构后再补跑。

当前 T9_3 未挂载时只能进行独立 PICO 遥操测试。推理接管测试需要先确认 T9_3 已挂载、上述 ckpt 目录下存在 pretrained_model/model.safetensors 与 assets/norm_stats.json，并先启动服务端再启动客户端。
