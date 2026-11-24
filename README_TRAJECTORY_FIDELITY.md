# D1 轨迹记录：本地开发版

2026-11-18。原始 `../tk_data_Driven_snapshot/` 保持不变。本目录为独立开发副本；未部署上位机，未连接机械臂，未做真机测试。本地缺少 torch，完整推理进程尚未在本机启动验证。

## 已实现

在当前 async client / direct_robot 路径增加显式开关，默认关闭：

```bash
# 下面是未来将受审查文件部署到原上位机目录后的用法，不是本地启动命令。
cd /home/tk/data_Driven
bash recipes/pi05/run_current_tk.sh client --trajectory-fidelity
```

不加该参数则不创建记录器。未修改跟随模式、频率、轨迹、延迟补偿算法或状态查询负载。仅支持当前异步客户端的 direct_robot 模式，其他启动入口没有自动启用。

每次输出在原 run 目录的 `trajectory_fidelity/`：

- `commands.csv`：每臂 SDK 发送起止、返回码/异常、实际 SDK 角度和对应弧度参考；左右共享 command_id，可查双臂提交时差。
- `feedback.csv`：每臂关节查询起止、返回码/异常和转换后的弧度位置。时间轴采用查询结束时刻，非硬件精确采样时刻。
- `chunks.csv`：已安装动作块的完整后处理参考，chunk_id 与执行索引关联。参考偏移按原 15 Hz 定义，不能当作实际发送时间。
- `events.csv`：请求、worker 结束、安装、任务/组边界及可复用的外骨骼/设备事件。worker 结束不代表推理成功；未安装的失败/作废块不会伪造预测点。
- `gripper.csv`：成功调用发送后的 UDP 目标和执行步骤后的反馈快照；反馈快照不等于夹爪原始数据包时间序列。当前 local 夹爪模式未补齐独立发送测量。
- `run_meta.json`：配置快照、相关源码 SHA256、时钟原点、队列接收/写入/丢弃计数、错误和正常返回标记。设备版本尚需人工填写实验记录。

记录使用有界队列和后台写盘。目录/文件初始化失败在硬件启动前报错；运行期队列满或写入失败不改变机器人调用，质量状态写入元数据。异常终止未完成 close 时，初始元数据仍为 `closed=false`，不能将这类记录认作完整数据。断电可能丢失缓冲数据。

## 离线分析

有 numpy、matplotlib 的环境下：

```bash
python3 infer/tools/analyze_trajectory_fidelity.py /path/to/run/trajectory_fidelity
```

这里 `infer/tools/` 相对本开发目录。可加 `--no-plots` 仅计算指标；`--max-gap 0.3` 控制允许的参考年龄，是首版分析参数，不是硬件安全阈值。

生成 `analysis/summary.json`、双臂各 7 关节目标/反馈叠图、误差图、间隔图，以及供 PlotJuggler 导入的 `commands_left.csv`、`commands_right.csv`、`feedback_left.csv`、`feedback_right.csv`。导入时选择共同的 `time_sec` 时间列，叠加对应的 target 与 measured 曲线；四份文件共享同一运行时钟，无需分别归零。尚未在桌面 PlotJuggler 实测或保存布局模板。

分析用实际 SDK 实参换回的模型关节约定作参考，在主机查询结束时间上作零阶保持匹配。排除未来参考、超出最大间隔、失败/无 chunk 来源命令、查询横跨命令切换及已记录任务/接管边界的样本。成功 SDK 返回不等于控制器接收或机械到位。

首版只报告原时间轴误差和时序统计；没有实现物理滞后拟合、硬件采样时钟校准、UDP 速度序列、FK 末端误差或任务成功率分析。需要这些能力时继续按 D1 文档扩展。记录完整性不足会在报告中提示，不能仅看图判断优化有效。

## 本地验证

```bash
cd /home/zzx/VLA/infer/tk_data_Driven_dev/infer
PYTHONPATH="$PWD" python3 -m pytest --import-mode=importlib -q infer/tests/test_trajectory_fidelity.py
```

使用 importlib 模式以避免原工程内外两个 infer 路径的测试导入冲突。测试通过模拟 SDK 和抽取真实引擎方法执行，不启动完整 GPU/相机/GUI 推理。

合成示例在 `test_results/trajectory_fidelity_synthetic/`，明确标记为 SYNTHETIC：20 组双臂目标、预设反馈偏差 0.02 rad、左右发送时差 1 ms。示例不代表现场结果。

## 真机前仍需完成

1. 对比上位机最新源码与 11 月 12 日基线，审查 `trajectory_fidelity_changes.patch`，不要覆盖远端期间的新改动。
2. 在原上位机环境验证导入与启动参数，确认记录输出磁盘和设备版本。
3. 部署与真机操作另行确认后，先对照关闭/开启记录的基线，检查实际发送间隔和记录丢失，不改变 RM 参数。
4. 取得真实记录后再判断误差、估计滞后、补齐 PlotJuggler 布局和其他 D1 能力。D2/D3 尚未实施。
