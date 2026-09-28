# 公开代码来源审计

指定模块的源码审阅，不是整仓复现。blob SHA是单文件指纹，不是仓库commit；未猜测未返回的commit。

核对日期：2026-09-28。除原仓库外，实际查看8个公开仓库的下列模块。只有CLARITY注意力通过原项目继续改编；其余主要是经过源码核对后的机制重实现或明确不采用。不要将此表解释为“8套顶会模型全部复现完成”。

## CLARITY

论文：https://arxiv.org/abs/2512.08029v4

代码：https://github.com/DingTianxingjian/CLARITY/blob/HEAD/Predictor/models/survival_module.py

实际查看范围：lines 1–250；文件blob SHA：`21e984ee04745348b8fc5151e4650a63616a4567`。

定位：作者arXiv页面标注Accepted to ECCV 2026；v4为2026-09-19。

采用/借鉴：通过原项目保留pre-norm顺序双向attention；新增统一结局读取，不复制上游完整SurvivalModule。

未采用：策略优化、原训练数据/权重、将sigmoid自动解释为时间生存概率。

许可边界：核验MIT，声明已保留。

## DreamerV3

论文：https://arxiv.org/abs/2301.04104

代码：https://github.com/danijar/dreamerv3/blob/HEAD/dreamerv3/rssm.py

实际查看范围：lines 1–230；文件blob SHA：`44b8e9ca5f01be7d6abeb91a2dc57a8364552d1a`。

定位：DreamerV3作者论文/代码；本次Nature网页访问失败，不用失败页面支撑细节。

采用/借鉴：observe/imagine分离、平衡KL、free-nats的机制；本版独立实现Gaussian belief。

未采用：离散32×32 latent、原GRU核心、actor/critic/RL、原权重。

许可边界：不打包上游源码/权重；不替上游指派许可。

## Set Transformer

论文：https://proceedings.mlr.press/v97/lee19d.html

代码：https://github.com/juho-lee/set_transformer/blob/HEAD/modules.py

实际查看范围：完整modules.py；文件blob SHA：`3b7e698e14ef7b8e1ea0c419fe51d93e3ee04bf5`。

定位：ICML 2019，PMLR官方页面核验。

采用/借鉴：PMA learned seed pooling思路；用mask-safe MHA和FFN重新实现。

未采用：原模型整包、未经监督的医学语义slot命名。

许可边界：本次未逐字核验其许可；没有打包上游文件。

## DiT

论文：https://arxiv.org/abs/2212.09748

代码：https://github.com/facebookresearch/DiT/blob/HEAD/models.py

实际查看范围：lines 95–140；另检查LICENSE.txt开头；文件blob SHA：`c90eeba7b2eee18b40b2128045248795b4b38d91`。

定位：DiT作者代码；小潜变量速度场不等于图像DiT复现。

采用/借鉴：adaLN-Zero六路调制与残差门控的设计；独立小型PyTorch实现。

未采用：timm依赖、图像patchify/VAE、预训练DiT权重。

许可边界：核验上游CC-BY-NC 4.0；未打包其源码或权重。

## Meta Flow Matching

论文：https://arxiv.org/abs/2412.06264

代码：https://github.com/facebookresearch/flow_matching/blob/HEAD/flow_matching/path/affine.py

实际查看范围：lines 1–150；文件blob SHA：`81cb7ed31f2434d03424ea9a5571a36bfc9f2681`。

定位：作者官方Flow Matching库及指南。

采用/借鉴：仿射概率路径与速度监督；本版独立Heun采样器。

未采用：exact flow density、CNF散度计算、把KL(q||Gaussian)写成KL(q||flow)。

许可边界：文件核验CC-by-NC声明；未打包其源码或权重。

## Causal Transformer

论文：https://proceedings.mlr.press/v162/melnychuk22a.html

代码：https://github.com/Valentyn1997/CausalTransformer/blob/HEAD/src/models/ct.py

实际查看范围：lines 1–165；文件blob SHA：`d2a3dfefdec3c6ad8e3889ffcd6e5cef68648ccf`。

定位：ICML 2022，PMLR官方页面核验。

采用/借鉴：审查过去治疗/结局/状态与当前治疗分流、前缀mask边界；作为因果目标警示而非直接移植。

未采用：对抗治疗平衡、因果识别结论、动态治疗策略优化。

许可边界：不打包上游源码/权重；不替上游指派许可。

## pycox / PC-Hazard

论文：https://arxiv.org/abs/1910.06724

代码：https://github.com/havakv/pycox/blob/HEAD/pycox/models/loss.py

实际查看范围：lines 130–173与170–255；文件blob SHA：`5345c755506148cac85cf63e7d1eda275d571a02`。

定位：作者生存建模实现；PC-Hazard方法论文。

采用/借鉴：分段风险softplus与删失似然原理；独立实现连续暴露、多样本混合及竞争风险。

未采用：ranking loss、把排序分数当绝对风险、抄注释中sigmoid误写（代码实际softplus）。

许可边界：不打包上游源码/权重；不替上游指派许可。

## MeWM

论文：https://arxiv.org/abs/2506.02327

代码：https://github.com/scott-yjyang/MeWM/blob/HEAD/Survival/model/dim1/TransMIL.py

实际查看范围：完整TransMIL.py（请求1–190）；文件blob SHA：`1850fea8cf280ede210b5ddf2293658ddb5bbdb2`。

定位：医学世界模型作者论文/代码。

采用/借鉴：比较生成后结局读取与多实例聚合的选择；当前短CT网格保留精确attention。

未采用：Nyström/2D square padding/硬编码.cuda()，未把该文件复制进本包，未训练像素生成器。

许可边界：本次未独立核对许可全文；没有打包该源码/权重。

## 单文件指纹的复查

链接使用仓库HEAD便于导航；HEAD可能改变。要找审阅版本，可使用GitHub Git Blob API对照返回内容，或在仓库历史中按blob指纹确认。没有取得上游仓库commit时，没有虚构固定commit链接。原项目基线则明确固定到aec8cbe08157687fd447594c8a9406d6ee5a464d。

## 最新论文但没有代码接入

V-JEPA2.1与LeWorldModel只用于评估研究方向；当前包没有它们的encoder、教师、目标分布正则或权重，也没有声称其机制已经通过消融验证。

## 许可

CLARITY MIT声明见licenses/CLARITY_LICENSE.txt。DiT及Meta Flow Matching不是MIT，不能在打包完整上游源码时默认使用MIT。当前没有附带它们完整源码或预训练权重。原仓库研究代码的许可不被本扩展重新指定，见NOTICE.md。
