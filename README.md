# Medical World Models

胃癌多阶段 CT 世界模型与乳腺 MRI 生成及 pCR 预测的研究代码、数据适配器和汇总实验结果。代码及结果更新日期：2026-09-29；乳腺部分新增四阶段完整训练及结果复核，保留 2026-09-28 的历史快照。

## 项目与代码

| 项目 | 任务 | 入口 |
| --- | --- | --- |
| 胃癌 | 治疗情景条件化的 CT 状态预测、CT1 观察更新及多阶段结局预测 | [gastric_tcwm_upgrade](gastric_tcwm_upgrade/README.md) |
| 乳腺 | I-SPY2 的 T0 到 T3 MRI 潜在状态生成与 pCR 联合建模 | [breast_world_joint](breast_world_joint/README.md) |

两个项目分别安装、训练和测试。公开内容包括完整下游网络、训练和推理代码、适配器、配置、测试、聚合指标与图表。真实患者数据、特征缓存、逐例预测和训练权重保存在研究环境，不包含在本仓库中。

## 当前结果

| 项目与评价范围 | 结果 | 解释 |
| --- | --- | --- |
| 胃癌上一版模型，原留出测试 65 人 | 复发 AUROC **0.6240**，AP **0.3725** | 当前保留研究模型，主要收益来自临床基线 |
| 胃癌 2026-09-28 CT 优化，521 人嵌套交叉验证 | S1 AUROC **0.5989**；仅临床 **0.5892**；同结构对照 **0.5789** | 历史候选开发结果，增益区间跨零 |
| 胃癌 2026-09-28 候选全量重训，原验证 65 人 | S1 AUROC **0.5196** | 未替换上一版，原测试未再评分 |
| 胃癌 2026-09-29 G0-G3，521 人内部 OOF | G0 S1 AUROC **0.5964**；临床 **0.5892**；差值区间 **[-0.0025, 0.0166]** | 12 次训练完成，未证明稳定收益；本轮不重评原验证/测试 |
| 乳腺 readout 选中模型，102 人开发验证 | pCR AUROC **0.7096**，AP **0.6230**，NLL **0.5640** | 四阶段训练已完成；阈值 0.5 准确率 74.51%，敏感度 31.25% |
| 乳腺 joint 选中模型，同一开发验证 | pCR AUROC **0.7136**，AP **0.6067**，NLL **0.5774** | 相比 readout 未证明稳定收益；后期分类过拟合，无独立测试集 |

胃癌 2026-09-28 比较完成 6 个配置各 3 折的 18 次训练，以及一次原训练集 521 人的全量重训。2026-09-29 在原训练代码中加入阶段权重、零步临床候选、病例级固定 MC 和分支诊断，并完成 G0-G3 共 12 次训练、5400 个监督更新。降低世界模型学习率、观察重建和简化读出均未带来稳定额外收益；仍保留上一版研究用检查点。

本轮同时完成 B0-B4 固定特征诊断和历史 token_prior 的真实先验预测评价。CT1 有初步信号，但先验预测在固定 CT1 聚合空间中仍差于训练均值，生成分布很窄。源码、逐折指标与负结果见 [最新胃癌结果](results/gastric/next_round_20260929/README.md)、[完整报告](gastric_tcwm_upgrade/docs/NEXT_ROUND_RESULTS.md) 和 [复现协议](gastric_tcwm_upgrade/docs/NEXT_ROUND_PROTOCOL.md)。新旧训练日程和评价口径不同，不作未经控制的直接归因。

乳腺正式训练于 2026-09-29 08:24:57（UTC+8）结束，表征、生成、读出、联合阶段分别完成 10,000 / 30,000 / 3,000 / 5,000 步。四份分类检查点重新验证后，原记录的 AUROC、AP、Brier、NLL 与选模分数完全复现。表征分类监督和联合阶段后期存在过拟合，不支持原样增加步数；同模型 T0-only 分支的 AUROC、AP、NLL 略优于加入生成 T3，尚未证明生成未来有增益。完整指标、诊断图与限制见 [乳腺训练复核](results/breast/training_review_20260929/README.md)。

原 764 人训练、102 人开发验证划分保留；冻结 VQ 也曾使用该开发验证集选权重，不能将其解释为独立测试。2026-09-28 的 [训练快照](results/breast/training_snapshot.json) 仅保留采集时进度，不代表当前状态。

详细指标与限制见 [胃癌结果](results/gastric/README.md) 和 [乳腺结果](results/breast/README.md)。不同评价范围的分数不可直接当成同一测试集上的模型比较。

## 安装与验证

建议 Python 3.12，为两个子项目分别建立环境。以下命令从仓库根目录开始，先按实际硬件安装合适的 PyTorch。

```bash
python -m venv .venv-gastric
. .venv-gastric/bin/activate
python -m pip install -e './gastric_tcwm_upgrade[dev]'
(cd gastric_tcwm_upgrade && python -m pytest -q)
deactivate

python -m venv .venv-breast
. .venv-breast/bin/activate
python -m pip install -e './breast_world_joint[test]'
(cd breast_world_joint && python -m pytest -q)
```

合成数据运行示例、真实数据适配所需输入和训练命令见各子项目 README。没有真实数据时可以执行工程测试与合成示例；聚合结果不足以重新构建原始患者队列。

## 结果解释与来源

胃癌当前真实数据实验使用记录性二分类终点，不代表固定年限生存风险。回顾性治疗摘要按指定情景解释，不识别治疗因果效应。乳腺当前实验为 T0 到 T3 单区间，完整纵向接口不代表已完成多区间训练。工程测试或合成数据分数不作为临床性能证据。

保留原项目及第三方贡献的署名和许可证。胃癌来源见 [NOTICE](gastric_tcwm_upgrade/NOTICE.md)，乳腺见 [LICENSE](breast_world_joint/LICENSE) 及其 `licenses/`。本仓库不统一重新许可已有代码；部分组件有非商业条款，公开访问不等于授予无限制的商业使用权。

整理范围与数据边界见 [PUBLICATION.md](PUBLICATION.md)。
