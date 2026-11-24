# WALL-OSS RM75 IK

模型出的是相对 EE，GPU 还原成绝对位姿后把 **EE20 回给客户端**，再解 **RM75 7DOF**。卡住几乎都是**连续性策略**，不是解析 IK 无解。

任务盒子太大时，这条 EE→IK 路线会「loss 能降、真机跟不住」——那是任务 workspace 超出了 40° 连续可解集，不是求解器坏了。经验：[docs/EE_IK_WORKSPACE.md](../../docs/EE_IK_WORKSPACE.md) / [docs/kb/ee_ik_workspace.yaml](../../docs/kb/ee_ik_workspace.yaml)。

默认 **`abort_rest` + `solver: realman` + 40°**，和改本功能之前的真机行为一致。对照改 `policy.ik.mode` 或 `policy.ik.solver`，不要改模型。开环脚本 `openloop_00086.py` 仍在本进程里解 IK。

怎么切 YAML / 环境变量、日志字段、对照顺序：文末「怎么切」。算法本身从下一节开始。

---

## 1. 要解什么

一次 infer：

```text
相对 EE (n, 20)  +  当前 FK 绝对 EE
        → 绝对 EE 路点 x_1 … x_n
        → 每个路点：左臂 7 关节 + 右臂 7 关节（度）
        → 逻辑 16D 下发
```

末端位姿是 **6 自由度**（位置 3 + 姿态 3）。RM75 是 **7 关节**，多出来的 1 维是肘绕肩–腕轴线的冗余角，RealMan 叫 **arm angle ψ**。同一组 (xyz, R) 对应一整圈肘构型，关节差可以到 50–100°。

真机 15 Hz、开环 chunk 8～10 步。相邻两步关节跳太大，下位机 `movej_canfd` 会拒、或看起来像飞肘。所以求解器找到解还不够，必须选**离当前关节最近的那一支**，并且单步不超过 `max_joint_step_deg`（默认 40°）。

00073 卡住的典型形态：右肘第 4 关节 |R4|≈0°（肘伸直附近），解析 IK 往往仍能给出另一支解，但相对当前构型差 45–67°，被 40° 门限丢掉；笛卡尔缩短只能走到 `cart_t≈0.4`，然后 **丢掉本 chunk 剩余路点** → 日志 `n_planned=1`。模型输出并没有冻住。

五种 mode **共用同一套解析求解器**，只在「近支不连续」之后分叉。

```text
目标 EE x* ，当前关节 q₀
        │
        ▼
解析 IK：默认解 + 以当前 ψ 为中心、步长 arm_angle_step_deg 扫整圈
        │
        ├─ 存在近支  ‖Δq‖∞ ≤ 40° 且 FK 对得上  → 直接用（五种 mode 相同）
        │
        └─ 近支没有
              ├─ abort_rest      笛卡尔缩短，然后丢掉剩余路点
              ├─ continue_chunk  同样缩短，但继续解下一个绝对 EE
              ├─ psi_blend       朝远支的 ψ 拧一小步，再解同一 EE
              ├─ split_jump      关节空间朝远支走一小步（摊开翻肘）
              └─ dls             同 continue，走不满再用 DLS 补剩余 EE
```

---

## 2. 共享前半段：解析 IK + 选支

实现：`RM75Kinematics.ik_physical`。

1. 把 rot6d 收成四元数，调 RealMan `rm_algo_inverse_kinematics`（带当前关节当 seed）。
2. 用 `rm_algo_calculate_arm_angle_from_config_rm75` 算当前 ψ₀，再调 `rm_algo_inverse_kinematics_rm75_for_arm_angle`，从 ψ₀ 起按 `arm_angle_step_deg`（默认 15°）左右交替扫满 ±180°。这不是「种子太少」——整圈都会扫到。
3. 每个候选必须 **FK 回对** 目标位姿（位置 5 mm、姿态 2°），重复解去重。
4. 相对 q₀ 单关节最大差 ≤ 20° 的立刻返回（近得足够，不必看更远的支）。
5. 否则取离 q₀ 最近的一支。若 `enforce_max_step` 且最大关节差 > `max_joint_step_deg`，返回「近支失败」，把这支记成 `rejected_deg`（远支确实存在，只是一步跳不过去）。

**奇异附近解析仍然可以出解。** 失败的是「连续地待在当前肘」。R4≈0 时雅可比接近奇异，笛卡尔方向上可走的关节增量很小，这是后面 DLS / 笛卡尔插值要处理的，不是 inverse Jacobian「算不出来」。

---

## 3. 五种算法

### 3.1 `abort_rest`（默认，基线）

近支失败后走 **笛卡尔二分插值**，不跟远支关节：

```text
t ∈ [0, 1]
x(t) = (1-t) x_now + t x*          位置线性
R(t) = slerp(R_now, R*, t)         姿态
在 x(t) 上再跑一遍解析 IK（enforce 40°）
能解的最大 t → 该路点的关节
```

`t=1` 就是完整目标；`t≈0.4` 表示只走到目标的四成。位移 < 4 mm 且关节几乎不动 → `held`。

然后 **丢掉本 chunk 里这条之后的全部路点**。下一轮 infer 用新观测重新出相对 EE。

这是改本功能之前的行为。安全：不会翻肘、不会硬插 60° 关节。代价：R4≈0 时每轮只剩 1 步，有效控制频率掉到 ~3 Hz，看起来像卡在某个角度。

### 3.2 `continue_chunk`

**单步求解和 `abort_rest` 完全一样**（同一套笛卡尔缩短）。差别只在 chunk 层：

| | `abort_rest` | `continue_chunk` |
|--|--|--|
| 这一步 `cart_t<1` | 后面的绝对 EE 全部丢弃 | 用缩短后的关节当 seed，继续解 x_{k+1} |
| 双臂都 `held` | 丢弃 | `abort_on_held: true` 时也丢弃（避免空转） |

直觉：肘伸直时每步只能挪几厘米，但模型一个 chunk 里后面的目标往往还在同一方向。丢掉它们等于每秒重新问一次模型、每次只执行 1 步。接着走，可以把这几厘米乘上 `n_exec`，同一轮里爬出奇异附近。

**不会翻肘。** 远支仍然不用。00073 卡住的首选对照就是它。

### 3.3 `psi_blend`

针对「解析已经给出远支，差的是肘翻转」：不在关节空间硬跳，**在冗余角上插值**。

```text
ψ₀ = 当前构型的臂角
ψ* = 远支解的臂角
Δψ = wrap(ψ* − ψ₀) ∈ (−180°, 180°]
ψ_next = ψ₀ + clip(Δψ, ± psi_step_deg)     默认 ±8°

q = IK(x*, ψ_next)                         固定臂角的解析解
若 40° 内解不出：对 x(0.45) 再解一次同一 ψ
仍没有 → 退回笛卡尔缩短（同 abort_rest 的单步）
```

7DOF 上从一支肘到另一支，最短路径往往是 ψ 连续拧过去，而不是某一关节突然 +60°。每步只拧 8°，FK 位姿仍盯着当前目标（或它的 45% 插值），单步关节差通常过得了 40° 门限。

适合日志里反复出现 `rejected_deg` 40–70、且 `rm_algo_inverse_kinematics_rm75_for_arm_angle` 有解。不适合远支根本不存在（纯工作空间 / 奇异、`rejected_deg=0`）。

### 3.4 `split_jump`

解析远支 q* 已经在手里，只是 ‖q* − q₀‖∞ = `rejected_deg` > 40°。**关节空间直线走向远支**，一次只走一段：

```text
Δq = wrap(q* − q₀)
step = min( max_joint_step_deg,  rejected_deg / split_max_substeps )
q = q₀ + (step / ‖Δq‖∞) · Δq
```

`split_max_substeps=4`、远支差 60° → 本路点最多走 15°。后面的路点若仍指向同一支，几步后落到远支上。chunk 不 abort。

和 `psi_blend` 的差别：`psi_blend` 走的是 **冗余角流形**（中间构型的 FK 仍尽量靠近目标 EE）；`split_jump` 走的是 **关节直线**（中间构型的末端会暂时离开目标，类似「先翻肘再够」）。翻肘过程末端可能划一条捷径外的弧，所以默认把大跳变摊成 4 步，不要 `split_max_substeps: 1` 配 40° 一次甩满。

适合确认远支是「对的那一支肘」、可以接受末端短暂离开模型 EE。不适合工作空间边界上没有远支。

### 3.5 `dls`（解析优先 + 阻尼最小二乘补）

**不换肘。** `mode: dls` 先走和 `continue_chunk` 一样的解析 + 笛卡尔缩短；`t<1` 时再用数值雅可比，从缩短后的关节把剩余 EE 补完。单步总预算仍 ≤40°。纯数值（跳过解析）写 `solver: dls`。

误差（位置米 + 姿态旋转矢量弧度）：

```text
e = [ p* − p(q) ,  log(R*^T R(q)) ]
```

雅可比用 FK 差分（每关节 +0.05°）：

```text
J_{:i} = ( FK(q + ε e_i) − FK(q) ) / ε
```

加权 DLS（位置 1/2cm，姿态 1 rad）：

```text
dq = (WJ)ᵀ ( WJ(WJ)ᵀ + λ² I )⁻¹ (W e)
```

`λ = dls_lambda`（默认 0.08）在奇异处把最小奇异值托起来，避免 dq 爆炸。然后三层裁剪：

1. 位置误差大于 `dls_pos_step_m`（1.2 cm）则按该长度缩小 dq，避免一步笛卡尔过大。
2. 本路点累计关节行程不超过 40°。
3. 最多 `dls_iters` 次（默认 12）。位置 < 2 mm 且姿态 < 1.5° 提前停。

推不动（位移 < 4 mm 且关节 < 0.5°）→ `held`。

这就是「奇异附近用数值解」。`mode: dls` **先走解析**，笛卡尔 `t<1` 才从缩短后的关节再 DLS 补剩余 EE；单步总预算仍 ≤40°。纯数值（跳过解析）写 `solver: dls`。它 **不能** 跳到另一支肘：雅可比积分沿着当前流形走。

---

## 4. 怎么选

一次只改一个旋钮。菜单 action-step 仍用 8 或 10。

| 现场现象 | 先试 |
|----------|------|
| `n_planned=1`、`cart_t≈0.4`、`rejected_deg≈0`、\|R4\| 很小 | `continue_chunk`（40° 不动） |
| 上面还卡 | `continue_chunk` + `max_joint_step_deg: 58`（不要一次 90） |
| 经常 `rejected_deg` 40–70 | `psi_blend`，其次 `split_jump`；或整段 `traj_opt: psi_dp` / `retract` |
| `rejected` 很少、笛卡尔也走不满（00073 cart_t≈0.34） | `dls`（解析优先再补） |
| 逐步 IK 总砍成 1 步 | `traj_opt: psi_dp`（前瞻换肘），其次 `gn` / `smooth` |
| 想回到改之前 | `abort_rest` 且 `traj_opt: none` |

不要默认改 RealMan 原生 EE 控制（`movel` / `movep_canfd` / `movev_canfd`）：时间尺度和「小关节增量」规则与这里同类，还丢掉 ψ 选择。

---

## 5. 怎么切

客户端 YAML（已写入 00036 / 00073 / 00086 / 00091）：

```yaml
policy:
  ik:
    mode: abort_rest          # abort_rest | continue_chunk | psi_blend | split_jump | dls
    solver: realman           # realman | dls | lm | svd | jacT | ccd | multiseed
    traj_opt: none            # none | psi_dp | smooth | gn | retract
    max_joint_step_deg: 40    # 对照可改 58
    arm_angle_step_deg: 15    # 解析扫 ψ 的步长
    abort_on_held: true       # 双臂都 held 时仍丢掉剩余路点
    psi_step_deg: 8           # psi_blend：每步拧多少 ψ
    split_max_substeps: 4     # split_jump：远支跳变摊几步
    dls_lambda: 0.08          # dls / svd 阻尼
    dls_iters: 12
    dls_pos_step_m: 0.012      # 00073 τ0-WM 客户端 YAML 用 8 / 0.024
```

`mode` 管 chunk 连续性（砍不砍剩余路点、翻肘怎么摊）。`solver` 管单步怎么解关节。`traj_opt` 管**整段** EE 路点怎么变成关节轨；默认 `none` 仍逐步贪心。其它 traj-opt 会覆盖 abort 砍短，尽量输出满 `n_exec` 步（每步仍 ≤40°）。

预设文件（服务端 `WALLOSS_IK_CONFIG` 直接指过去）：

| 文件 | 作用 |
|------|------|
| `recipes/walloss/ik.yaml` | realman + traj_opt none（默认） |
| `recipes/walloss/ik_solver_*.yaml` | 单步 solver |
| `recipes/walloss/ik_traj_psi_dp.yaml` | 整段 ψ DP |
| `recipes/walloss/ik_traj_smooth.yaml` | 贪心后再关节平滑 |
| `recipes/walloss/ik_traj_gn.yaml` | 沿轨迹 Gauss-Newton（别名 ilqr） |
| `recipes/walloss/ik_traj_retract.yaml` | 朝远支关节插值再投影 EE |

客户端 YAML `policy.ik` 只打在 websocket **`solve_ik`** 上，不进模型 infer。换 mode 下一包热切。

`WALLOSS_IK_MODE` / `TAU_IK_MODE` 仍能传给 serve 的 argparse，但 **不再求解**。真机对照只改 YAML。

真机对照只改客户端 YAML（或重启客户端）。`WALLOSS_IK_*` / `TAU_IK_*` 对 websocket 求解无效；开环 `openloop_00086.py` 仍读那些环境变量。

```yaml
policy:
  ik:
    mode: continue_chunk
    solver: realman
    traj_opt: none
    max_joint_step_deg: 40
```

mode 别名：`abort`/`baseline` → `abort_rest`；`continue`/`no_abort` → `continue_chunk`；`psi`/`arm_angle` → `psi_blend`；`split` → `split_jump`；`numerical`/`jacobian` → `dls`。

solver 别名：`analytical`/`rm`/`sdk` → `realman`；`levenberg` → `lm`；`pinv`/`pseudoinverse` → `svd`；`jt`/`transpose` → `jacT`；`multi` → `multiseed`。

traj_opt 别名：`off`/`greedy` → `none`；`dp` → `psi_dp`；`qp` → `smooth`；`ilqr`/`gauss_newton` → `gn`；`interp` → `retract`。

日志：

```text
ik_mode=continue_chunk n_planned=8 aborted_ik=False cart_lim=True cart_t=0.43 rejected_deg=0.0
```

| 字段 | 含义 |
|------|------|
| `n_planned` | 本轮实际下发路点数；`< n_exec` 且 `aborted_ik` 就是旧 abort |
| `cart_lim` / `cart_t` | 笛卡尔缩短；`t=1` 表示完整 EE |
| `rejected_deg` | 解析远支相对当前关节的最大差（被 40° 丢掉的那个） |
| `held` | 双臂都没动 |

```bash
python recipes/walloss/test_ik_config.py
python recipes/walloss/test_ik_modes.py
python recipes/walloss/test_ik_trajectories.py
python recipes/walloss/test_ik_solvers.py            # 多 solver：示教 EE+当前关节 / 真机卡住帧
python recipes/walloss/test_ik_traj.py               # 整段 EE 拟合：示教 action.ee + 00073 卡住插值
python recipes/walloss/test_ee_infer_pipeline.py
```

单臂 solver 对照在 `ik_solvers.py`，YAML `policy.ik.solver` 已接入。样本是示教 `action.ee` + 当前 7 关节，以及 00073 eval_log 里 R4≈0 的卡住帧。双臂都到位（<5 mm、<2°）且关节差 ≤40°：

| solver | 示教双臂到位 | 奇异附近 | 真机卡住帧 | 位置误差 p50 | 耗时 p50 |
|--------|--------------|----------|------------|--------------|----------|
| **realman** 解析 | 93% | 94% | 38% | 0.0 mm | 0.5 ms |
| **lm** 自适应阻尼 | 93% | 94% | 0%（位置很好，姿态常差一口） | 0.9 mm | 5.8 ms |
| **multiseed** 当前 DLS + 邻近 ψ | 93% | 94% | 38% | 0.0 mm | 11 ms |
| svd 阻尼伪逆 | 60% | 61% | 0% | 0.2 mm | 10 ms |
| dls | 33% | 39% | 0% | 0.5 mm | 9.6 ms |
| ccd | 10% | 11% | 0% | 1.1 mm | 49 ms |
| jacT | 0% | 0% | 0% | 49 mm | 23 ms |

示教上解析仍然最准最快。真机卡住帧目标是「朝恢复姿态走一半」，笛卡尔跨度大：解析常缩短（误差厘米级），**lm/dls 位置能跟到 1 mm 内**但不能两臂姿态都到 2°。jacT/ccd 不值得上真机。

`test_ik_trajectories.py`（刚跑过 **PASS 2/2**）覆盖：

| 来源 | 窗口 | 结果 |
|------|------|------|
| 示教 EE：00036 / 00073 / 00086 / 00091 | 40（26 个 \|J4\|<5° 奇异附近） | 全部 ≤40°；`continue_chunk` 路点数 ≥ `abort_rest`；00073 有 1 窗 abort 砍短、其它 mode 走满 8 |
| 真机 eval_log `action-step_10` 267 帧 | 102 次 n_planned=1，其中 76 次 \|R4\|<3° | 回放已执行 chunk 全部成功 |
| 从 R4≈0 卡住帧笛卡尔插值到后来恢复姿态 | 6 窗 | t=94（R4≈0°）`abort_rest` n=1、rejected 41°；另外四 mode n=8 |
| 把 abort 允许的那一小步 EE 方向拉成 8 路点 | 6 窗 | 同样 t=94 上 abort 出 6 步、其它出满 8 |

这是离线 FK/IK，**不是真机对照**。

| 文件 | 干什么 |
|------|--------|
| `ik_config.py` | `IkSettings` / `parse_ik_config`（`mode` + `solver` + `traj_opt`） |
| `ik_solvers.py` | realman / dls / lm / svd / jacT / ccd / multiseed |
| `ik_traj.py` | psi_dp / smooth / gn / retract |
| `ik.yaml` / `ik_solver_*.yaml` / `ik_traj_*.yaml` | 开环 / 本机预设；真机以客户端 YAML 为准 |
| `ee_infer_pipeline.py` | `_ik_one_arm` 分发；`ik_abs_chunk` 截断或整段 traj-opt |
| `walloss_policy.py` / `serve_tong_walloss.py` | GPU：FK + 绝对 EE20 |
| `infer/infer/policies/{walloss,tau0wm,client_ee_ik}.py` | 客户端 IK |
| `infer/infer/config/lingbot/walloss_tongbot_ee_*.yaml` | 客户端 `policy.ik` |

张量流：[docs/WALLOSS_DATAFLOW.md](../../docs/WALLOSS_DATAFLOW.md)。
