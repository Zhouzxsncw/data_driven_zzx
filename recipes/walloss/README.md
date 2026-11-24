# WALL-OSS TongBot 一整套（本仓）

换机 **clone `data_Driven` 的 `walloss/infer-ik`（或以后合进 main 的同路径）** 即可拿到：

- 真机客户端（`infer/` 里的 adapter + YAML）
- 服务端 websocket + RM75 FK/IK
- 江算 / 华为 / gpu-node1 训练胶水

**不要**再从 `/home/xk/VLA/wall_oss_ppu_setup` 或 `walloss_staging` 拷。那两处不是 git 仓。

训练实现仍在单独 fork，再 clone 一次：

```bash
git clone -b walloss/infer-ik git@github.com:qiaow371/data_Driven.git
git clone -b tongbot git@github.com:qiaow371/wall-x.git
```

权重、数据集、`RM_API`、conda **不进 Git**。本机 gpu-master 布局：`/data/wall-oss/{conda,src,outputs,datasets}`，SDK：`/home/xk/RM_API`。

端到端张量与 IK 截断：[docs/WALLOSS_DATAFLOW.md](../../docs/WALLOSS_DATAFLOW.md)。

## 目录

| 路径 | 干什么 |
|------|--------|
| `serve_tong_walloss.py` `launch_walloss_{server,client}.sh` `env_infer.sh` | 真机服务 1239 |
| `walloss_policy.py` `ee_infer_pipeline.py` `convert_tongbot_joint_to_ee.py` | 策略 + FK + 连续 IK |
| [IK.md](IK.md) `ik_config.py` `ik.yaml` `ik_solver_*.yaml` `ik_traj_*.yaml` | IK mode + solver + traj_opt |
| `openloop_00086.py` `test_ee_infer_pipeline.py` `test_ik_config.py` `test_ik_modes.py` `test_ik_trajectories.py` `test_ik_solvers.py` | 开环 / IK 模块、多轨迹、多 solver |
| `ee_sft_pipeline.sh` | joint → EE → norm → SFT |
| `train/` | TongBot SFT YAML 与启动脚本 |
| `env/` `docker/` `download/` `bootstrap_jiangsuan*.sh` | 江算 16 / 4 环境 |
| `huawei/` + `scripts/bootstrap_huawei_walloss.sh` | 华为 A2/A3 |
| `../../infer/infer/policies/walloss_adapter.py` | 客户端（发 16D，不做 IK） |
| `../../infer/infer/config/lingbot/walloss_tongbot_ee_*.yaml` | 00036 / 00073 AH8 / 00086 / 00091 |

## 真机（gpu-master）

```bash
# 服务（等日志 listening）
WALLOSS_CKPT=/path/to/ckpt WALLOSS_N_EXEC=8 \
  bash recipes/walloss/launch_walloss_server.sh

# 客户端
bash recipes/common/launch_client.sh walloss_00086
# 或 walloss_00036 / walloss_00073 / walloss_00091
```

conda 必须 `source recipes/walloss/env_infer.sh`。菜单 action-step 常用 **8**，不要默认 32。

## IK 算法

RM75 是 7 关节、末端只有 6 自由度。同一位姿有一圈肘构型（RealMan 臂角 ψ）。解析 IK（默认解 + 当前 ψ 起扫整圈）经常**有解**，但离当前关节 45–67°，被单步 40° 门限丢掉。00073 卡在特定角度，是连续性策略把 chunk 砍成 1 步，不是模型停转。

五种 mode 共用解析求解器 `realman`，只在「近支过不了 40°」之后不同。另有数值 solver，以及整段 `traj_opt`（`psi_dp` / `smooth` / `gn` / `retract`）。YAML `policy.ik.mode` / `solver` / `traj_opt` 切换，不必重载权重。公式、参数、对照顺序见 [IK.md](IK.md)。

离线已测（RM_API，不碰真机）：

```bash
python recipes/walloss/test_ik_config.py
python recipes/walloss/test_ik_modes.py
python recipes/walloss/test_ik_trajectories.py   # 四套示教 + 00073 真机 eval_log 奇异附近
```

`test_ik_trajectories.py` 刚跑过 **PASS**。示教 00036/00073/00086/00091 共 40 个窗口（26 个 |J4|<5°）；真机 `action-step_10` eval_log 267 帧里 76 帧是 `n_planned=1` 且 |R4|<3°。全部 mode 关节差 ≤40°。卡死帧 t=94（R4≈0°）上 `abort_rest` 只出 1 步，另外四个 mode 出满 8 步。**真机还没对照过。**

**`abort_rest`（默认）**  
近支失败就在笛卡尔上二分插值，停在当前肘，然后**丢掉本 chunk 剩余路点**。安全、不翻肘；肘伸直时每轮只执行 1 步，看起来像卡住。

**`continue_chunk`**  
单步和上面完全一样，但缩短之后继续解后面的绝对 EE。肘附近每步只能挪几厘米时，用满 `n_exec` 才能爬出去。00073 首选对照。

**`psi_blend`**  
远支已经算出来时，不要关节硬跳。把当前 ψ 朝远支拧 `psi_step_deg`（默认 8°），再跑「固定臂角」解析 IK。走的是冗余角流形，中间构型末端仍尽量盯着目标。适合日志里反复 `rejected_deg` 40–70。

**`split_jump`**  
同样有远支，但在**关节空间直线**走向它，每步只走 `rejected / split_max_substeps`（默认摊 4 步）。中间末端会暂时离开模型 EE，相当于先翻肘再够。比 `psi_blend` 更「莽」。

**`dls`**  
不换肘。当前构型上用 FK 差分估雅可比，阻尼最小二乘 `dq = Jᵀ(JJᵀ+λ²I)⁻¹ e` 往目标推。奇异处 λ 托住最小奇异值，避免关节爆。笛卡尔插值也走不动、且没有远支可跳时用。它积不出另一支肘。

不要默认改 RealMan `movel` / `movep`：15 Hz VLA 一样吃「小关节增量」，还把 7DOF 选择权交给厂家。

## 训练

```bash
# 本机 FK + 写 YAML（要 RM_API）
bash recipes/walloss/ee_sft_pipeline.sh --src /path/to/lerobot_v21 --until norm

# 江算 16：两路 8 卡，不要 NPROC=16
bash recipes/walloss/bootstrap_jiangsuan16.sh
# 容器内: bash /workspace/code/train/train_00086_ee.sh
# 槽 B: GPUS=8,9,10,11,12,13,14,15 PORT=29619
```
