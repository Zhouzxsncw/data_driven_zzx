# VLA 自动评估

根据真实评估表格结构 + 推理日志 + 可配置规则，自动填写评分表并生成统计报告。

## 安装依赖

```bash
pip install -r vla_eval/requirements.txt
```

## 批量推理集成

`run_batch_infer.py` 在每个 **action-step** 组结束后，自动在 `<action-step_xx>/eval_table/` 生成：

- `eval_result.et`
- `summary.md`

在 direct 配置的 `batch_infer.vla_auto_eval` 中开关：

```yaml
batch_infer:
  vla_auto_eval:
    enabled: true
    output_dirname: eval_table
    result_table_name: eval_result.et
    summary_name: summary.md
```

默认表格模板：`vla_eval/template/评测表.et`（分值版）  
默认规则配置：`vla_eval/config_scored.yaml`

## 命令行

```bash
cd /path/to/infer
export PYTHONPATH="$(pwd):$PYTHONPATH"

python3 vla_eval/main.py \
  --table ./input/real_eval_table.et \
  --logs /path/to/action-step_50 \
  --config vla_eval/config.yaml \
  --output ./output/eval_result.et \
  --summary ./output/summary.md
```

## 配置说明

所有评估项、字段映射、规则与成功判定均在 `vla_eval/config.yaml` 中配置，**不要改主程序硬编码**。

| 配置段 | 作用 |
|--------|------|
| `table` | 表格列名映射、test 列识别、元数据行 |
| `log` | 日志文件名、test 匹配、task 分段、video 路径 |
| `rules` | 每条规则对应表格中一项「具体表现」 |
| `success` | 成功/失败判定与特殊 test 处理 |
| `exclusivity` | 同阶段多项命中时只保留一项 |
| `output` | 备注与新增列行为 |

## 规则类型

- `field_stat`：时间窗口内字段统计（min/max/range/...）
- `consecutive`：连续 N 帧满足阈值
- `motion_range`：关节运动幅度
- `gripper_cycle`：夹爪闭合/张开模式
- `sequence`：多步骤顺序检查
- `all` / `any`：组合逻辑

## 日志字段

默认读取：

- `machine_flow.jsonl` → `action_step_sent`（密集状态，含 gripper / joints）
- `eval_log.jsonl` → `eval_inference_frame`（含末端位姿、相机、动作 chunk）
- `task_segments.jsonl` → 按 test 切分时间段

## 输出

- 评估结果表格：保留原格式，命中表现写 `1`（不写 `0`）
- `summary.md` / `summary.xlsx`：成功率、表现统计、失败原因、调整建议
