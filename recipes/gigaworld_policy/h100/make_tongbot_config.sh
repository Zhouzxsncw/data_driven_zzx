# -*- coding: utf-8 -*-
# 在 h100 上生成 TongBot 版 config(基于 agilex 版打三处补丁,幂等)
# 1) view_keys: agilex cam_* -> TongBot front/left/right
# 2) delta_mask_by_embodiment_id: 14D(6关节+夹爪) -> TongBot 16D(14 关节 delta + 2 夹爪绝对)
# 3) model_action_dim 32 -> 16:GWP-0.5 发布的 state/action encoder 权重实测
#    in_dim=16([2,16,128]),transform 必须输出 16D,否则 bmm 维度崩
#    (bs=1 也崩,与 batch 无关);in/out_action_channels=16 保持不动
set -e
cd /data1/gwp/giga-world-policy
SRC=configs/giga_world_policy_0_5_agilex_finetune.py
DST=configs/giga_world_policy_0_5_tongbot_finetune.py
cp "$SRC" "$DST"
python3 - "$DST" <<'PY'
import sys, re
p = sys.argv[1]
s = open(p).read()
old_views = '''view_keys = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]'''
new_views = '''view_keys = [
    "observation.images.front",
    "observation.images.left",
    "observation.images.right",
]'''
assert old_views in s, "view_keys 块不匹配,检查上游 config"
s = s.replace(old_views, new_views)
m = re.search(r"'0': \[[^\]]*\],", s)
assert m, "delta_mask 行不匹配"
s = s.replace(m.group(0), "'0': [True] * 14 + [False] * 2,")
assert "model_action_dim=32," in s, "model_action_dim 行不匹配"
s = s.replace("model_action_dim=32,", "model_action_dim=16,")
# 4) with_ema=True -> False:12B 主干在 zero2 下每卡权重+EMA 各 ~24G,bs8 必 OOM;
#    finetune 场景 EMA 属可牺牲项,关掉每卡省 ~24G
assert "with_ema=True" in s, "with_ema 行不匹配"
s = s.replace("with_ema=True", "with_ema=False")
open(p, "w").write(s)
print("[patch] tongbot config OK: view_keys + delta_mask(16) + model_action_dim(16) + no_ema")
PY
