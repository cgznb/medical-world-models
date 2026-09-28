# 工程核验与真实训练的区别

本公开版本保留当前源码和测试，未收录旧交付包中的合成数据报告。
实际患者训练的时间戳状态见 [乳腺结果](../../results/breast/README.md)。

## 自动化测试

在本项目目录运行：

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

测试覆盖概率边缘化、缺失标签、患者级 Energy Score、未来监督隔离、
padding、编码和流梯度、四阶段训练、VQ 编解码、严格权重迁移、
患者划分、train-only 统计、阶段坐标与断点恢复，以及现有数据适配。
公开导出后的重跑结果记录在 `../../results/breast/engineering_checks.json`。
MONAI 1.5.1 parity 测试需要可选依赖；跳过该项不能视为 MONAI 已通过验证。

## 合成 smoke

```bash
python joint.py smoke --output runs/synthetic_smoke
```

该命令使用合成患者和缩小网络检查四阶段与独立推理接口。
它产生的 AUROC、pCR 概率和 MRI 输出均不能作为真实患者效能证据。

## 真实数据 CUDA 预检

已有预检使用真实 `[24,8,32,32]` 三相 latent、batch=4、K=2、
20 步 Heun，执行四阶段 loss、反向传播和 AdamW 更新。
所有更新有限；joint 峰值 CUDA 分配约 1.92 GiB，单次更新约 5.45 秒。
聚合记录见 [cuda_preflight.json](../../results/breast/cuda_preflight.json)。
预检使用临时模型，不改变正式训练模型，也不评估临床性能。

## 证据边界

正式训练已经使用真实患者缓存和 CUDA/BF16；当前快照尚未完成所有阶段。
训练/开发验证分别为 764/102 名患者，没有独立测试集。
冻结 VQ 权重使用这 102 名患者进行过选择，因此不能将其称为新独立验证。
尚无生成 MRI 专家评分、跨医院外部测试或临床因果效应证据。
单区间显存预检不等价于完整三段纵向训练验证。
