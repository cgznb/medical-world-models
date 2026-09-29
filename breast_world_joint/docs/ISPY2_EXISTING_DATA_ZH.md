# 现有 I-SPY2 数据适配与运行

本轮使用既有匿名连续 VQ 缓存，保留原来的 764 名训练患者和 102 名开发验证患者。没有独立测试集。原始 3,039 个三相 latent 均经过形状与有限值检查，尺寸为 `[24,8,32,32]`，没有重新编码或重复标准化。

## 数据和建模范围

- `data/ispy2/direct_t0_t3.json`：正式首轮，一名患者一个 T0→T3 请求；524 名训练患者和 86 名验证患者有真实 T3 生成监督。其余患者保留最终 pCR 监督，缺失 MRI 为 `null`。
- `data/ispy2/longitudinal.json`：已准备的完整纵向接口，在可用 T0/T1/T2 时点使用当时已观察历史，预测后续全部阶段。该清单尚未进行正式多区间训练。
- 时间使用固定阶段索引，T0/T1/T2/T3 不是实际天数。没有读取未来实际扫描日期作为条件。
- 临床输入为筛查年龄、HR、HER2、MP 四个基线字段。源代码 `ClinicalTextPolicy` 将这些字段定义为 locked bundle 的静态基线信息，`baseline_conditions/connected_pairs` 校验它们跨访视不变；本次再次逐患者核对一致性。`known_at=0` 依据这个数据字典，而非独立逐字段时间戳审计。缺失年龄保留 `null`，不填成零岁。
- 治疗输入为零维。历史 treatment-arm 文本没有提供可靠的前瞻性分段计划，因此本轮不能解释为治疗方案比较或治疗因果模型。
- 每名患者各访视具有完全相同的 prepared ROI 网格、spacing、affine 和 T0 grid ID。该事实不等于专家确认解剖对齐；所有未来目标的 `anatomy_comparable=false`，并关闭 anatomy invariance loss。既有更严格空间质控子集没有被冒充为完整队列。
- 冻结 VQ 是原有单通道 DCE0 codec，分别应用于三相。它使用原训练患者训练、原验证患者选择 step 98,000。因此本轮验证属于现有开发协议，不能重新命名为独立测试。上游定位器的患者重叠仍未经核验。
- 新状态编码器、联合生成器和 pCR 头从头训练，没有迁移旧 V2 生成器或分类器权重。原包的完整 native 后端、生产网络宽度、K=2、20 步 Heun 和 BF16 保留。缺失 Pillar 教师没有被伪造，相关 loss 关闭。

## 运行配置

本轮正式训练使用 RTX 5090 32 GB、Python 3.12.3、PyTorch 2.12.1+cu130，已于 2026-09-29 08:24:57（UTC+8）完成。
以下命令从本项目目录执行，数据准备方式见 [README](../README.md)。

正式配置为 `configs/ispy2_t0_t3_5090.yaml`，物理 batch=4、累积=1，有效 batch=4。四阶段预算依次为 10,000 / 30,000 / 3,000 / 5,000 优化步；每 100 步保存恢复点，每 250 步在全部 102 名验证患者上评估。验证仅用于 checkpoint 选择。

```bash
python -u scripts/run_existing_ispy2.py \
  --config configs/ispy2_t0_t3_5090.yaml \
  --manifest data/ispy2/direct_t0_t3.json \
  --output runs/ispy2_t0_t3
```

入口持有运行锁，顺序运行四阶段；重启同一命令会从各阶段 `last.pt` 恢复。不要修改进行中实验的配置、manifest 或输入数组；框架会拒绝契约不一致的恢复。`controller_status.json` 为当前阶段，`console.log` 为总日志，各阶段包含 `training.jsonl`、`last.pt`，验证后产生 `best.pt`。

## 核验记录

`scripts/prepare_existing_ispy2.py` 可从现有 V2 清单重新生成两个适配清单，按文件名重绑定新的缓存根目录。`adaptation_report.json` 保存患者、标签、配对 MRI 和数组扫描数量。

`scripts/real_data_preflight.py` 使用真实完整尺寸 MRI latent，对四个生产阶段分别执行 loss、反向传播和 AdamW 更新，记录数值及 CUDA 峰值。预检使用临时模型，不改变正式训练模型，也不代表临床性能。公开聚合报告见 [cuda_preflight.json](../../results/breast/cuda_preflight.json)。

历史预检中，原包 CPU 测试与新增适配测试共 41 项通过；MONAI 后端测试因未安装可选 MONAI 跳过。另核验临床 4 维、治疗 0 维、未来 MRI 全缺失时表征/读出/联合梯度有限，以及 joint 中断恢复后模型张量和 RNG 完全一致。本次公开副本的工程核验入口见 [VALIDATION.md](VALIDATION.md)，历史通过数不代替本轮测试记录。

四阶段已分别完成 10,000 / 30,000 / 3,000 / 5,000 步；选中检查点位于 250 / 12,500 / 250 / 750 步。后续阶段从前阶段 `best.pt` 开始，并非从末步检查点开始。2026-09-28 的进度快照仅为历史记录。

## 完成后的开发验证

102 名验证患者中有 32 名 pCR 阳性和 70 名阴性。原选模协议为每人 4 条生成轨迹、每条 20 步 Heun，对轨迹概率取均值。固定 0.5 阈值下，readout/best 的准确率为 74.51%、敏感度为 31.25%、特异度为 94.29%，对应 AUC 0.7096、AP 0.6230、NLL 0.5640；joint/best 对应为 73.53%、46.88%、85.71%，AUC 0.7136、AP 0.6067、NLL 0.5774。没有一份检查点在所有指标上更好。

四份 readout/joint 的 best/last 权重均重新执行原验证函数，原记录五项标量完全复现。表征分类监督和联合后期存在过拟合；joint/last 的 AUC 下降至 0.6772、NLL 升至 0.7519，当前证据不支持原样增加训练步数。同模型的 T0-only 分支在 AUC、AP、NLL 上略优于加入生成未来的分支，尚未证明生成 T3 提高 pCR 预测性能。

完整指标、阶段分项和重训建议见 [训练复核](../../results/breast/training_review_20260929/README.md)，机器可读汇总见 [summary.json](../../results/breast/training_review_20260929/summary.json)。四阶段优化已完成，完整纵向实验、解码 MRI 质量和独立测试泛化性能仍未验证。
