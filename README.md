# Medical World Models

胃癌多阶段 CT 世界模型与乳腺 MRI 生成及 pCR 预测的研究代码、数据适配器和汇总实验结果。整理日期：2026-09-28。

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
| 胃癌本轮优化，521 人嵌套交叉验证 | S1 AUROC **0.5989**；仅临床 **0.5892**；同结构对照 **0.5789** | 候选开发结果，增益区间跨零 |
| 胃癌新候选全量重训，原验证 65 人 | S1 AUROC **0.5196** | 未替换上一版，本轮未再次评分原测试集 |
| 乳腺正式训练 | 表征阶段完成，生成阶段进行中 | 无独立测试集，尚无最终 pCR 性能结论 |

胃癌比较完成 6 个配置各 3 折的 18 次训练，以及一次原训练集 521 人的全量重训。新增了 prior CT 的直接预测监督，并比较直接 CT 与小型条件预测网络；尚未证明世界模型有稳定的额外收益。

乳腺状态来自 2026-09-28 14:15 UTC 的只读快照：表征阶段 10,000/10,000 步，生成阶段 27,990/30,000 步，读出及联合阶段尚未开始。这是采集时状态，不是实时监控。原 764 人训练、102 人开发验证划分保留；冻结 VQ 也曾使用该开发验证集选权重，不能将其解释为独立测试。

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
