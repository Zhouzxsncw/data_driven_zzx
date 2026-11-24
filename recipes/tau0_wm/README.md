# τ0-WM（智元 · 罗剑岚团队）训练 recipe

- 论文：[arXiv:2606.01027](https://arxiv.org/abs/2606.01027) · 项目：[tau0-wm.github.io](https://tau0-wm.github.io/)
- 代码：`git@github.com:sii-research/tau-0-wm.git` → 本机 `/home/xk/VLA/tau-0-wm`（Apache-2.0）
- 权重：[hf sii-research/tau-0-wm](https://huggingface.co/sii-research/tau-0-wm) → 本机 `/data/pretrained/tau-0-wm`（22.93 G，不 gated）
  - VAM `diffusion_pytorch_model-0000{1,2}-of-00002.bin` = **11.02 G**（bf16 ≈ 5.5 B）
  - Simulator `simulator/diffusion_pytorch_model.bin` = **11.90 G**
- 定位：与我们 FastWAM / GigaWorld-Policy **同主干（Wan2.2-TI2V-5B）**，是本舰队换更大具身先验的首选，见 [docs/WAM_BACKBONES.md §5.1](../../docs/WAM_BACKBONES.md)

## 一句话结构

Wan2.2-TI2V-5B 视频 DiT（30 层 / dim 3072 / 48ch latent）+ 一个 hidden 1024 的 Action DiT
经 **cross-attn** 耦合（**不是** MoT 共享注意力），外加**第二个复用同一 Wan 变换器的
「动作条件视频模拟器」**用于给低分动作重生成（LAR）。实测 config：

| | VAM | Simulator |
|---|---|---|
| `action_in_dim` | **20**（EEF 相对：每臂 xyz+6D+爪 = 10×2） | 26 |
| `action_dim` / `action_ffn_dim` / heads | 1024 / 2048 / 16 | 同 |
| `dim` / `ffn_dim` / layers / heads | 3072 / 14336 / 30 / 24 | 同 |
| `in_dim`/`out_dim` / `patch_size` / `model_type` | 48 / [1,2,2] / ti2v | 同 |
| `max_seq_len` | 1024 | 1024 |
| 额外分支 | — | `act_cond_*`（1536，patch 16）、`reward_*`（`use_re: true`）、`use_ae: false` |

## 落档的坑（都已解决，别重踩）

1. **`HF_ENDPOINT=hf-mirror` + Xet 会 401**
   mirror 不代理 `cas-server.xethub.hf.co`，报 `File reconstruction error ... 401 Unauthorized`。
   必须 `HF_HUB_DISABLE_XET=1`。（`HF_HUB_ENABLE_HF_TRANSFER` 在新 huggingface_hub 里已弃用，无效。）
2. **00036 必须用 v3.0 那份**
   `/home/xk/dataset/00036_20260804_RM_01_yj_v21` 是 **v2.1**，lerobot≥0.4 只认 **v3.0**；
   同目录的 `/home/xk/dataset/00036_20260804_RM_01_yj` 才是 v3.0（151 ep / 50440 帧 / 16D / 3 cam）。
   相机键名也变了：v2.1 的 `front/left/right` → v3.0 的 **`cam_mid/cam_left/cam_right`**。
   我们自己的 EE 版是从 **v2.1** 转的，所以键仍是 `observation.images.{front,left,right}`
   —— 换数据集时 `valid_cam` 必须跟着换，别照抄 joint 版的 `cam_*`。
3. **`tasks.parquet` 索引退化 → caption 变成 "0"**（只发生在别人那棵旧树上）
   那次 v2.1→v3.0 转换把任务文本留在列 `__index_level_0__`、索引退化成整数，而 τ0-WM 用
   `meta.tasks.iloc[idx].name` 取 caption，于是语言条件被喂成字符串 `"0"`。
   `make_dataset_root.sh` 生成一棵只软链、不改动原数据集的修正树
   `/home/xk/dataset/00036_v30_for_tau0wm`。**用 lerobot 0.4.4 自带的
   `lerobot.datasets.v30.convert_dataset_v21_to_v30` 重新转换即可，产物索引就是真指令**
   （`convert_ee_on_h100.sh` 走的就是这条，已验证 caption 不再是数字）。
4. **上游 `runner/posttrain.py` 有个致命死导入**
   commit `a3e4c0f` 里 `from utils.model_utils import forward_pass` 引用的符号**全仓库没有定义**
   （训练循环是内联在 `Trainer.train()` 里的，这个 import 纯粹是遗留），
   结果 `from runner.posttrain import Trainer` 直接 ImportError —— 训练根本起不来。
   `patch_tau_repo.py` 把它改成可选导入（幂等，h100 侧由 `setup_h100.sh` 与
   `relay_watch.sh` 各打一次）。**如果哪天 upstream 修了，这个脚本会自己跳过。**
5. **rot6d 的行列约定会静默毁掉预训练先验**
   WALL-OSS `euler_zyx_to_rot6d` = ZYX 矩阵**前两行**；τ0-WM
   `matrix_to_rotation_6d` / `rotation_6d_to_matrix` = **前两列**。两者互为转置，
   喂错**不报错**，只是 `abs_eef_to_rela` 里的 `R_ref^T Δp`、`R_ref^T R_cur` 全变味
   （小角度 ≈ 旋转符号翻转）。`convert_00036_ee.py` 用 monkeypatch 换成立约定，
   并用 scipy `from_euler('ZYX')` 交叉验证（实测 `max|R_dec−R_ref|≈1e-7`）。

以下 6–9 是 2026-09-15 h100-103 正式开训时连环踩到的，全部已进 `patch_tau_repo.py`
或脚本修复，双端（gpu-master / h100）生效：

6. **uv venv 里没有 `torchrun` 入口脚本**
   torch 2.7.1 装好了但 console script 缺失（`command not found`，冒烟 rc=127）。
   `launch_train.sh` 现在优先用 PATH 上的 torchrun，找不到就回退等价的
   `python -m torch.distributed.run`；h100 venv 里也补了个 `bin/torchrun` shim。
7. **`WanModel` 缺 `_supports_gradient_checkpointing = True`**（补丁 2）
   `forward()` 里本来就实现了 checkpoint 分支，但 diffusers≥0.30 的
   `enable_gradient_checkpointing()` 先查这个类属性，没有直接
   `ValueError: WanModel does not support gradient checkpointing`（冒烟 rc=120）。
   补类属性即可让 `gradient_checkpointing: true` 真走省显存路径，而不是被迫关掉。
8. **args Namespace 由 YAML 顶层键生成 → 自写配置缺 `allow_tf32` 必崩**
   `runner/posttrain.py:504` 无条件读 `self.args.allow_tf32`；官方 yaml 有这键，
   我们自写的两份（joint/EE）漏了 → `AttributeError`（冒烟 exitcode 1）。
   已补 `allow_tf32: false`。**换官方新版本时先 set-diff 官方 yaml 的顶层键。**
9. **两个"训练已成功却非零退出"的上游 bug**（补丁 `epoch-stop` / `shutdown-tee`）
   - `train_steps` 早退只 `break` 内层 step 循环，外层 epoch 循环（train_epochs=10000）
     继续 → 到点后永不退出（实测冒烟 20 步刷到 134 it / epoch 115）。relay 的
     SMOKE_TIMEOUT 会把它判成超时 FATAL，正式跑过 train_steps 也只是空转。
   - `utils/__init__.py` 的 `Tee.write/flush` 在解释器 finalization 阶段对已析构的流
     抛异常 → unraisable → **exitcode 120**：冒烟 20/20、epoch 跳出、11 G 存档全部
     成功后死在退出路上，torchrun 报 main.py FAILED。`Tee` 已改成写失败静默丢弃。
   - 顺手一个 shell 坑：`LOG="...$( [[ ${SMOKE} -eq 1 ]] && echo '_smoke').log"` 在
     `set -e` 下，非冒烟分支替换串 rc=1 会把赋值变命令失败 → **launch_train.sh 正式
     路径静默 exit 1、零输出**（冒烟反而能过）。已改 if/else。

## 20D EE 动作契约（决定能不能继承预训练动作专家）

预训练的权威规格在 **`configs/deployment/tau_pretrain_rela_eef6d.yaml`**：
`action_dim/action_in_dim=20`、`action_type: relative`、`action_space: eef6d`。
（官方 post-train 示例 `taco_abs_joint.yaml` 是 `absolute/joint/7D`，那等于**自己把动作头随机初始化**，
别被它带跑。）

落盘的 `action.ee` / `observation.state.ee` 是 20 维**绝对** EE：

```
左 xyz(0:3) | 左 rot6d(3:9, 列约定) | 左爪(9) | 右 xyz(10:13) | 右 rot6d(13:19) | 右爪(19)
```

dataset 内部（`data/example_dataset.py:326-393`）**先**按 `action_index/state_index` 切 18 宽、
**再**做 `relative`（`abs_eef_to_rela` 断言 D∈{12,18}，索引 0:3/3:9/9:12/12:18）、
**最后**才 `dual_arm` 插爪重排成 20D。所以配置必须写：

```yaml
action_key: action.ee            state_key: observation.state.ee
action_type: relative            action_space: eef6d
action_index/state_index: [0..8, 10..18]        # 18 个非爪维
action_gripper_index/state_gripper_index: [9, 19]
```

`scripts/compute_stat.py --action_space eef6d --action_type relative` 走同一条重排路径，
生成的 mean/std 直接就是这 20 维顺序（实测 50440 帧 / 0 跳过 / 无零 std，
`idx 3/7/13/17`（rot6d 对角项）mean ≈ 0.94，与预训练 `statistics.json` 的 0.995 同结构 ——
这是「动作语义对上了」最硬的一条证据）。

**基座系朝向差异不用管**：body-frame 相对增量满足
`R_ref'^T(p'-p') = R_ref^T Δp`，两臂基座系之间任意固定旋转 Q 会被抵消；
爪的量程差异（我们 0–1、预训练 0–120）由 per-dataset mean/std 归一化吸收。


## 分辨率与 token 预算

3 个视图**各自独立过 VAE**，再在**宽度维拼接** latent
（`runner/posttrain.py`: `rearrange(video,'b c v t h w -> (b v) c t h w')` → `rearrange(latents,'b v c m h w -> b c m h (v w)')`）。
`sample_size` 必须被 **32** 整除（VAE 16× × patch 2）。

| sample_size（单视图 H×W） | latent | 拼接后 | patch 后 token/帧 | ×3 帧 |
|---|---|---|---|---|
| 192×256（= 480×640 等比，官方示例值） | 12×16 | 12×48 | 144 | **432** |
| 256×352（未采用） | 16×22 | 16×66 | 528 | 1584 |

`chunk 9 / action_chunk 33` 是硬约束：`video_temporal_stride=(33-1)//(9-1)=4`，
且代码 assert `(chunk-1)*stride == action_chunk-1`。改任一都要满足它。

## gpu-master 侧（备用环境 / 算统计用）

```bash
bash setup_env.sh            # conda env tau0wm（py3.11 + torch 2.7.1+cu128）
source env.sh                # 激活并设 TAU_ROOT / PYTHONPATH
bash download_pretrained.sh  # 权重 → /data/pretrained/tau-0-wm
bash link_pretrained.sh      # Wan2.2 三件（T5/tokenizer/VAE）软链，省 32 G
python compute_stat.py …     # 实际是 repo 里的 scripts/compute_stat.py，见下
python verify_ee_dataset.py --root-map /data1/tau0wm/datasets /data/wall-oss/datasets
```

`setup_env.sh` / `h100/setup_h100.sh` 与官方 `requirements.txt` 的差异都写在文件头注释里
（cu128、不装 xformers/flash-attn、不装 pinocchio 栈、不装 deepspeed、不装 `msgpack-python`、
lerobot 用 0.4.4 `--no-deps`）。

## h100-103 侧（正式承载机）+ GWP 门控

**硬要求：τ0-WM 必须等 GigaWorld-Policy 训练跑完才能开始**，所以 GPU 的占用完全交给
`h100/relay_watch.sh`，人工只做前三步的推送与启动：

```bash
# ① gpu-master：推物料（repo 19M + recipe + RM_API 13M；1.4G 数据集不推，到 h100 现算）
setsid nohup bash h100/push_when_up.sh > push_when_up.out 2>&1 &   # gateway 限流时会自动等
# ② h100：装环境 + 拉 22.9G 权重（与安装并行）+ 软链 Wan2.2 三件
setsid nohup bash /data1/tau0wm/local_scripts/h100/setup_h100.sh > /data1/tau0wm/setup_h100.out 2>&1 &
# ③ h100：本地做 joint→EE（FK + v2.1→v3.0 + 与 gpu-master 统计互校）
setsid nohup bash /data1/tau0wm/local_scripts/h100/convert_ee_on_h100.sh > /data1/tau0wm/convert_ee.out 2>&1 &
# ④ h100：挂接力，然后就可以不管了
setsid nohup bash /data1/tau0wm/local_scripts/h100/relay_watch.sh > /data1/tau0wm/relay.out 2>&1 &
tail -f /data1/tau0wm/relay.log          # 终态：RELAY_DONE / RELAY_FATAL
touch /data1/tau0wm/HOLD                 # 任意时刻让 watcher 停在「等待」阶段
```

`relay_watch.sh` 的四个阶段：

| 阶段 | 判据 |
|---|---|
| 0 前置自检 | `SETUP_DONE`、VAM/T5/VAE/tokenizer、数据集、统计、repo 都在 + 打 fork 补丁 —— **现在就查**，别等 GWP 跑完才发现缺东西 |
| 1 等 GWP | 「GWP 家族」进程全部消失 **且** GPU 2-7 峰值显存 < 3000 MiB，**连续 3 次**才放行（120 s/次，上限 48 h）。家族 = 训练本体 `scripts/train.py --config giga_world_policy*` + `run_task.py --config configs.giga_world_policy*` + **会把它重新拉起的看门狗** `resume_bump_bs_watch.sh` / `watch_then_launch_train.sh` / `wait_upload_then_prep.sh`（判定集中在 `h100/gwp_gate.sh`） |
| 2 冒烟 | `CUDA_VISIBLE_DEVICES=2` 单卡 bs1 / 20 step，要求退出码 0 + 日志出现 `loss=` + 无 Traceback/OOM；顺带记录显存峰值供调 bs |
| 3 正式 | 再确认卡还空 → `GPU 2-7` 六卡 torchrun，40 min 内看见 loss 就写 `RELAY_DONE` |

只吃 **GPU 2-7**（GWP 的六张），0/1 是别人的；不抢卡、不 kill 任何进程。

**为什么阶段 1 不能只 pgrep 训练进程**：GWP 自己有个 `resume_bump_bs_watch.sh`，
它在 bs4→bs8 切换时会 kill 旧六卡再拉起新六卡 —— 中间有几分钟「训练进程不在、显存也是空的」
的空窗。只看训练进程就会在这里误放行，τ0-WM 冲进去跟 GWP 的重启撞车。所以门控把
**看门狗本身也算成「GWP 还在」**（阶段 0 还会额外核对：显存占着却一个 GWP 进程都匹配不上
→ 直接 FATAL，而不是当作已结束放行）。

同一个 `gwp_gate.sh` 也被 `launch_train.sh` 用（放在脚本最前面，同步配置之前），
所以手滑直接 `bash launch_train.sh` 也会被拒：

```
[gate] REFUSED：GigaWorld-Policy 还没收口，τ0-WM 不许启动。命中进程： ...
exit 75
```
人工确认越权才加 `TAU_IGNORE_GWP_GATE=1`。

配置真源在本目录 `configs/`，`sync_configs.sh` 拷进 repo（统计文件以 recipe 为准）；
`-o key.path=value` 可临时覆盖，生成的 YAML 落 `${TAU_RUNS}/_generated/` 留档。
`launch_train.sh` 换机器只需 `source h100/env_h100.sh`（它负责 TAU_ROOT / TAU_PY / TAU_RUNS / CONFIG）。


### 坑 10（09-16，致命级）：steps_to_save 周期块是死代码，全程零落盘

上游 posttrain.py 训练循环里，`%steps_to_log / %steps_to_val / %steps_to_save / %gc`
四个周期块缩进在外层 epoch-stop `break` **之后**（epoch-stop 补丁插入后更确定不可达），
**永远不会执行**。铁证：第一个 30k 跑 events 文件只有 88B/263B、无 latest_accelerator_state、
无 step_5000..25000 中间目录、config 无 latest_global_step 回填。30 万步长跑不补这个
= 中途零 ckpt，崩了全损。

修复 = patch_tau_repo.py **补丁 5**（per-step-blocks）：四个块移回内层 step 循环
（accelerator.log 之后、sync_gradients 守住），死区副本删除。
**今后任何超长 τ0-WM 训练前必须确认补丁 5 已应用**（patch_tau_repo.py 幂等自带验证）。

### 30 万步续训（09-16 启动，进行中）

- 起点：上个 30k 收官权重 `..._bs1/step_30000`（optimizer state 不存在——见坑 10），
  配置 `configs/tau_model/posttrain_tongbot_00036_ee_300k.yaml`：
  `model_path` 指 step_30000 + `latest_global_step: 30000`/`latest_epoch: 0` 续编号，
  `train_steps: 300000`，`steps_to_save: 20000`（15 份 ckpt ≈165G + accelerator state）。
- GPU 3-7 五卡（GPU 0-2 被 zjw 的 GR00T-N1.7-3B 微调占了，2 号卡回不来；
  五卡 1.56 it/s ETA ~48h；2 空了可杀掉从 latest_accelerator_state resume 切回六卡）。
- 启动必须 `source h100/env_h100.sh` 先（否则 gwp_gate 按本地路径判「找不到仓库」）。
- 日志 `/data1/tau0wm/logs/train_300k_0916.log`；首个存档 step_40000。

## 真机推理（00036 EE，端口 1244）

训练还没出终档也可以先把代码接上。契约已经定死：20D 相对 EE、列约定 rot6d、三路 192×256、chunk 33。

```bash
# step_N 出来后只拉 transformer bin（约 11G），不要 latest_accelerator_state
bash recipes/tau0_wm/pull_tau_ckpt.sh

bash recipes/tau0_wm/launch_tau_server.sh --load-only
bash recipes/tau0_wm/launch_tau_server.sh
bash recipes/tau0_wm/launch_tau_client.sh
```

客户端 YAML：`infer/infer/config/lingbot/tau0wm_tongbot_ee_00036_local.yaml`。握手 `model=tau0wm`。菜单 action-step = 服务端 IK 路点数（默认 8，上限 33）。denoise / n_exec / 哪些菜单项无效：见 [docs/COSMOS_TAU_INFER_PARAMS.md](../../docs/COSMOS_TAU_INFER_PARAMS.md)。

无 GPU 契约测试：`python recipes/tau0_wm/test_tau_tongbot.py`。

**E2E 已实测通过（09-16）**：服务真起后（非 --load-only），用
`.tmp_tau_e2e_client.py`（00036 数据集真实帧 + qpos，不接真机）打 3 发合成请求：
元数据 `model=tau0wm` / horizon=33，返回 action=(8,16) 满路点 `aborted_ik=False`，
flow≈230ms（denoise=10，GPU0 与 walloss/pi0.5 服务共存）。注意客户端 numpy→msgpack
必须走 `__ndarray__` 二进制约定（`infer/infer/policies/_lingbot_msgpack_numpy.py`），
tolist() 会把 e2e 拖慢 1s+；随机噪声图会让 IK 无解走安全保持（aborted_ik=True），
属预期，不是 bug。

## 显存 / 磁盘预算（h100-103 单卡 80G 已实测；gpu-master 96G 口径沿用）

- `mixed_precision: bf16` 时代码**不**调 `cast_training_params`（只在 fp16 下调），
  所以参数/梯度/AdamW 状态全是 bf16：≈ 11 + 11 + 22 = **44 G 静态** + 激活。
- **h100 冒烟实测（09-15，单卡 bs=1、gc=true、192×256）**：峰值 allocated 64.70 /
  reserved 64.78 GiB → 每样本激活 ≈ 20.7 G，**80G 卡 bs≥2 必 OOM，正式只能 bs=1/卡**
  （6 卡全局 bs6）。EE 配置已据此把 `batch_size` 8→1、`sub_folder` 后缀改 `_bs1`。
  速度 1.6–2.0 it/s，30000 步 ETA ≈ 5 h。
- 本机常驻两个 WALL-OSS 推理服务占 ~18 G，故先开 `gradient_checkpointing: true`。
- ckpt：`latest_accelerator_state`（含优化器）+ 每 `steps_to_save` 一份 `step_N/*.bin` ≈ 11 G。
  官方 2000 步一存会把 `/data`（剩 ~155 G）写爆，这里放开到 **5000** 并需要定期清理。
- h100 侧 `output_dir=/data1/tau0wm/outputs`（`/data1` 3.5T，开训时剩 1.1T）；
  gpu-master 侧必须落 `/data`：`/`（含 `/home`）只剩 109 G。

## 待办 / 未验证

- [x] **h100 侧开荒已落地（09-15）**：①→④ 全部走完，EE 转换跨机统计 max|Δ|=0，
      relay 四阶段跑到 `RELAY_DONE`，六卡（GPU 2-7）正式训练 09-15 16:05 起出 loss。
      看进度：`tail /data1/tau0wm/relay.log`、`tr '\r' '\n' < /data1/tau0wm/logs/train_*.log | tail`。
- [x] h100 本地 FK 生成的 EE 数据与 gpu-master 版统计逐维 diff 校验（`convert_ee_on_h100.sh` 里已内置，
      09-15 实测 max|Δ|=0.000e+00）
- [x] 冒烟实测单卡能上到多大 `batch_size`：**80G 卡只能 bs=1**（峰值 64.78/80 G @bs1），
      EE 配置已定版 bs=1（见上面显存账）
- [ ] 官方测试时计算 **RCS**（重加噪-重去噪一致性打分）+ **LAR**（低分动作回炉模拟器重生成）
      推理侧脚本：`scripts/run_posttrain_infer_ttc.sh`、`web_infer_utils/*_withTTC`；尚未接我们的下位机
- [ ] Simulator 后训练（`runner/posttrain_sim.py` + `configs/tau_model/posttrain_sim_taco_play_abs.yaml`）未跑
- [x] 真机 websocket：`recipes/tau0_wm/serve_tong_tau.py`（端口 1244）。客户端仍发 16D 关节；服务端 FK（列约定 rot6d）→ `TauPolicy.play` → IK → 16D。**不要**把官方 14D quat 协议直接给真机客户端。RCS/LAR 仍未接。
