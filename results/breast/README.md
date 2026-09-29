# 乳腺 MRI 生成与 pCR 联合模型

本目录仅发布聚合信息，不含患者标识、真实病例清单、逐例预测、影像或权重。

## 已完成训练与核查

四阶段训练于 **2026-09-29 08:24:57（北京时间）**完成，representation、flow、readout、joint 分别完成 10,000 / 30,000 / 3,000 / 5,000 步。
完整阶段解释、准确率、过拟合诊断和分支对照见
[2026-09-29 训练诊断报告](training_review_20260929/README.md)，机器可读结果见
[summary.json](training_review_20260929/summary.json)，可视化见
[训练诊断图](training_review_20260929/training_diagnostics.png)。

| 检查点 | 开发验证 AUROC | AP | NLL | 准确率，阈值 0.5 |
| --- | ---: | ---: | ---: | ---: |
| readout/best | 0.7096 | 0.6230 | 0.5640 | 74.51% |
| joint/best | 0.7136 | 0.6067 | 0.5774 | 73.53% |
| joint/last | 0.6772 | 0.5219 | 0.7519 | 69.61% |

四阶段正常完成计算，但表征分类监督和联合阶段后期有过拟合证据，不支持原样增加训练步数。已有 T0-only 分支对照未显示生成 T3 提高 AUROC、AP 或 NLL。
当前是 **T0→T3 单区间任务的四阶段优化**，不是已经验证的多段纵向预测。以上均为开发验证结果，不是独立测试成绩。

## 历史训练快照

快照采集时间：**2026-09-28 22:15:16（北京时间）**，对应 UTC 14:15:16。
采集当时控制器和阶段日志显示 `flow` 正在训练，以下为保留的历史记录，不能作为当前进度。
原 [training_snapshot.json](training_snapshot.json) 保持不变，最终结果以上述完成报告为准。

原患者划分为 **764 人训练、102 人开发验证**，没有独立测试集。
其中有真实 T3 监督的患者分别为 524、86 人；其余患者保留 pCR 监督。
所有输入使用阶段索引，当前任务为 T0 到 T3 的单区间生成与 pCR 预测。

| 阶段 | 最新日志步数 / 预算 | 最优开发验证目标 | 最近开发验证目标 |
|---|---:|---:|---:|
| representation | 10,000 / 10,000 | 0.724066 | 3.965158 |
| flow | 27,990 / 30,000 | 1.372435 | 1.444326 |
| readout | 尚未开始 / 3,000 | 无 | 无 |
| joint | 尚未开始 / 5,000 | 无 | 无 |

**这些目标是各阶段的损失，不是 pCR AUROC，也不是生成 MRI 的质量评分。**
representation 最近目标高于历史最优，不能据此宣称表征已经稳定泛化。
此时尚无最终 pCR AUROC、AP 或独立测试结果。后续阶段从前一阶段
选中的检查点接续，具体选择逻辑见 `breast_world_joint/src/responsewm/training.py`。
机器可读记录见 [training_snapshot.json](training_snapshot.json)。

冻结 VQ 曾使用相同开发验证患者选择权重；上游定位器的患者重叠仍未核验。
这些限制意味着即使后续有较高开发验证指标，也不能重新命名为独立测试结果。

## 工程验证

[cuda_preflight.json](cuda_preflight.json) 是真实全尺寸 latent 上的四阶段
CUDA/BF16 单次更新检查。它验证损失、梯度与优化器可运行，不评价临床性能。
公开副本的测试结果见 [engineering_checks.json](engineering_checks.json)。

`joint.py smoke` 使用合成数据，属于独立的工程检查；旧合成性能报告没有作为
真实结果收录。复现安装、数据准备与训练命令见
[项目 README](../../breast_world_joint/README.md)。
