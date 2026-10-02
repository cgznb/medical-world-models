# V3 / V6 输入和监督合同

两个网络读取 `modality-event-v2` 的患者级缓存，不直接读取 DICOM，也不随仓库提供
CT 编码器权重或患者特征。严格验证实现见
[timeline_data.py](../../src/stageworld_tcwm/timeline_data.py) 和
[modality_schema.py](../../src/stageworld_tcwm/modality_schema.py)。

## 目录文件

固定队列目录包含 `cohort.pt`、`split.json`、`split_protocol.json`、`preparation.json`。
`cohort.pt` 包含 `schema`、`ids`、`tensors`、`metadata`、`encoders`；成员顺序、
来源 pool、预处理训练成员、划分指纹和准备文件哈希相互绑定。
正式研究人数固定为651，train456/validation65/test130，划分 seed17。

正式入口要求原划分文件 SHA256：

```text
2b6b17fe70291c94673e0fbf31ac482828a55b318acc7407e13f216151edad48
```

公开仓库不包含该文件的成员 ID；哈希仅用于复现实验时校验身份，不能据此重建数据。
从旧 pool 转换用 `prepare_modality_fixed.py`，需要明确的旧临床编码源码和既有事件缓存。
重放既有研究时传入原 `--split`，不要重新划分后冒充同一实验。

## 张量概览

N 为患者数，E=3 为新辅助/手术/术后治疗三个阶段槽，Q 为请求的阶段数。

| 字段 | 形状 | 用途 |
| --- | --- | --- |
| `ct0` | N×27×768 | 基线 CT 冻结特征，前向输入 |
| `clinical` | N×32 | sex、age、bmi、cT、cN、cM 六字段的基线编码 |
| `image_valid` | N×2 | 影像观测标志；基线读出只使用第0列 |
| `modality_value/known/applicable` | N×E×7 | 治疗是否存在、是否已知、是否适用，分别保存 |
| `event_mask` | N×E | 有效事件槽 |
| `phase/operation/role` | N×E | 阶段、操作类型、事实/假设角色 |
| `event_id/event_order` | N×E | 事件身份与阶段顺序 |
| `time_features` | N×E×6 | 时间表示；当前使用 ordinal，不推造日历日期 |
| `occurred_at/available_at` | N×E | 无可靠日期时为 NaN，与 ordinal 合同一致 |
| `query_order/query_mask` | N×Q | Timeline 模型的阶段请求 |
| `ct1` | N×27×768 | 未来 CT 辅助目标；本配方不可作为前向输入 |
| `scan_event_index` | N | CT1 目标阶段索引，当前为 S1 |
| `pcr/pcr_valid` | N | pCR 标签及有效性，仅适用事实 S1 监督 |
| `binary/binary_valid` | N | 记录性复发标签及有效性，仅完整事实 S3 监督 |

七手段的精确定义、合法 value/known/applicable 组合，以 `modality_schema.py` 为准。
未知、已知未实施、结构性不适用不能互换。V6 的 `terminal_eligible` 还要求原来的
三槽 operation/applicability 结构；低层可以推进潜状态，不等于任意情景可获可靠风险。

V3 增加 `s1_concepts: N×4` 与 `s1_concept_valid: N×4`。
未观测值必须为 NaN，不能用0代替；`metadata.s1_report_concepts` 记录字段名、变换、
S1目标/S2可用边界、弱标签属性和来源。由 `prepare_report_concepts.py` 校验自动抽取
报告与原证据后附加，不接受把模型自生成标签声明为医学测量。

## 统计和信息边界

标准化、PCA、临床参考和概念均值尺度只在训练成员上拟合。验证集负责选模；
测试集不能用于选权重或调参。V6 两臂协调器只物化训练和验证行，拒绝测试索引。
报告字段及未来 CT 不能进入状态前向计算。pCR 不监督 S0，复发不逐阶段复制监督。

当前没有真实、验证过的 S2/S3 状态测量，未实现严格概念瓶颈。医生标注计划不等于
标注已经进入训练。病理标本的指标不能改名为术后患者体内疾病负荷。
