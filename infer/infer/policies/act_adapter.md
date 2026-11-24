# ACT 推理流程

对应代码：`act_adapter.py`  
示例配置：
- LeRobot ACT：`infer/config/lerobot/act_left_arm_direct.yaml`
- 原生 ALOHA ACT：`infer/config/aloha/act_aloha_*_direct.yaml`  
注册类型：`policy.type: act`

原生 ALOHA **DP / RDT** 是独立 adapter，见：
- [`aloha_dp_adapter.py`](aloha_dp_adapter.py) — `policy.type: aloha_dp`
- [`aloha_rdt_adapter.py`](aloha_rdt_adapter.py) — `policy.type: aloha_rdt`

## 原生 ALOHA ACT

`policy.act_backend: aloha` 绕过 LeRobot `from_pretrained`，经 `AlohaActRuntime` 加载
vendored `ACTPolicy` + flat `.ckpt` + `dataset_stats.pkl`。引擎 / 同步 / 录像链路不变。

`aloha_model_args` 须与训练一致。TongBot 固定 `arm_joint_state_dim: 8`（见 YAML）。
图像：RGB→BGR、JPEG round-trip、640×480 center crop、ImageNet norm；状态动作用 pickle 统计归一化。

**环境安装：** [../docs/ALOHA_ENV.md](../docs/ALOHA_ENV.md)

---

## 模型特点（LeRobot ACT）

- **Action Chunking Transformer**
- 一次推理预测 action chunk，内部 `_action_queue` 逐步消费
- 需要本地 **`dataset_root/meta/stats.json`**
