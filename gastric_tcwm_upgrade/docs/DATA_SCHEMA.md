# TCWM v1 数据合同与推理请求

本合同明确保留现有 `[27,768]` 缓存路径。四个治疗token是描述组，不是四个临床时间点。所有张量按同一患者顺序排列。

## 原生队列文件

PyTorch可安全读取的字典，使用 `torch.save` 保存，只允许字典、列表、基本值和张量，不使用自定义对象pickle：

```python
payload = {
    "schema": "tcwm-cohort-v1",
    "ids": ["与本地原缓存一致的脱敏ID", ...],
    "tensors": {...},
    "metadata": {
        "treatment_semantics": "explicit_interval_scenario",
        "endpoint_definition": "由研究协议指定的结局定义",
    },
    "encoders": {},  # 原缓存转换器会填入原临床与药物训练折编码状态
}
```

| key | 形状 / dtype | 含义 |
|---|---|---|
| ct0 / ct1 | [N,27,D] float | 同一冻结编码器的基线和治疗后特征；缺失以0占位 |
| image_valid | [N,2] bool | 两次CT各自是否确实可用 |
| clinical | [N,32] float | 原临床编码器的ridge_features；尚未做本模型标准化 |
| treatment | [N,4,82] float | 原治疗描述组编码；不是将所有槽随意设one-hot |
| interval_days | [N] float | 指定/事实CT间隔，必须正数 |
| surgery | [N] int64 | 0 absent / 1 present / 2 unknown / 3 conflict |
| prefix_valid | [N,3] bool | 各信息前缀是否可用于该终点分析，由数据核验确定 |
| binary / binary_valid | [N] float / bool | 记录复发状态及有效性；不是长期未复发证明 |
| pcr / pcr_valid | [N] float / bool | pCR辅助监督；不会自动用作S2观测输入 |
| ct1_available_stage | [N] int64 | 1=S1可用，2=S2才可用，3=三个阶段均不可用 |

`binary`/`pcr`失效位置允许0占位，valid位置必须0/1。不同患者的前缀不能分到不同fold。

## 生存字段（必须成套提供）

| key | 形状 | 含义 |
|---|---|---|
| time | [N] float | 从声明原点至首次事件或删失的观察月数 |
| event | [N] int64 | 0右删失，1首次复发，2复发前竞争死亡 |
| entry | [N,3] float | 三个前缀的风险集进入月数；与time同一原点 |

metadata必须含 `time_origin` 和 `time_unit="months"`。合法前缀要求time>entry。已复发患者不再进入首次复发预测风险集。术前预测手术后结局时entry=0是定义终点起点，而非声称术前检查发生在手术当天；此时预测人群条件是已进入该手术队列。

时间由已核验日期计算，例如 `(event_date - surgery_date).days / 30.4375`。不要用CT0到CT1的interval_days当成复发时长。未复发病例必须有末次确认未复发的随访日期；空白结局不能自动当作长期阴性。当前实现不支持区间删失、重复复发、多个复发原因或超过两个竞争原因。

`followup_template.csv`要求全部唯一ID恰好覆盖cohort。`s0_valid`等只可0/1；其中反映事件是否尚未发生、临床定义和信息可用性。不能在没有审计时全部机械填1。

## 可选真实术后观察

`post: [N,L,D_post]`、`post_mask: [N,L] bool`，metadata必须有 `post_available_stage=2`。模型配置设置 `postoperative_dim=D_post`。L是每患者缓存的token数，padding位置mask=False。只在S2使用。不要把未知病理、未来治疗或结局标签编码进post。

原版未提供真实术后图像状态；因此默认不启用post模块。对病理图像编码器、病例/切片划分、基础权重来源及信息可用日期的审计属于新增数据处理任务，本包不假造现成病理特征。

## 患者划分

`split.json`为：

```json
{"train": ["实际脱敏ID"], "validation": ["另一实际脱敏ID"], "test": ["第三实际脱敏ID"]}
```

上例只表示结构，不是可训练的三人数据。真实文件必须完整覆盖你的队列且互不重叠。词汇支持、原临床编码器拟合、标准化和治疗支持统计均以train为准。改变fold后应重新从原始缓存转换，不允许拿另一个fold的已拟合编码器继续运行。

## 无结局的推理请求

```python
query = {
    "schema": "tcwm-query-v1",
    "plan_source": "hypothetical",  # 或documented_plan / retrospective_factual
    "interval_source": "specified_query",  # 或retrospective_actual
    "tensors": {
        "clinical": clinical,      # [B,32]
        "treatment": treatment,    # [B,4,82]
        "interval_days": interval, # [B]
        "surgery": surgery,        # [B] long
        "ct0": ct0,                # [B,27,D]
        "image_valid": valid,      # [B,2] bool
        # S1/S2存在并已可用的CT1：
        "ct1": ct1,
        "ct1_available_stage": availability,
        # 生存模式必须传entry [B]或[B,3]：
        "entry": entry,
    },
}
torch.save(query, "query.pt")
```

不传患者结局。binary、pcr、event、time等结局字段被拒绝，不默默忽略。S0前向忽略真实CT1和post；S1/S2才允许使用按availability核验可用的CT1。阶段可用性是整数合同，不是完整临床时间戳系统；构造请求前应完成真实日期审计。

`documented_plan`必须另外传 `[B] int64` 的 `plan_available_stage`，并满足≤请求stage。治疗前使用这种声明却输入事后才知道的实际CT间隔，会被拒绝；请改为明确的指定查询间隔。声明本身不是对原始病历真实性的自动核验。

转换自原版缓存的bundle保留原始编码器参数，可用：

```python
from stageworld_tcwm.legacy import encode_legacy_inputs
values, warnings = encode_legacy_inputs("inference.pt", clinical_rows, treatment_rows)
```

这一步需要原repo的 `stageworld` 安装可用。输出clinical/treatment尚未做本模型标准化；Predictor内部应用fitted buffers。原生数值缓存的bundle不包含原始药名到编码的映射，必须使用与训练一致的预处理；不要再次标准化输入或拿完整数据重拟合。

## 输出解释

二分类返回recorded_status_probability；生存返回survival、cumulative_incidence和recurrence_probability。两种模式都标明`causal_effects_identified=False`和`clinical_validation=False`。latent标准差不是临床置信区间，MC标准误不是疗效不确定区间。超出精确治疗组合支持域默认报错；覆盖率诊断仅是训练频数，不是条件positivity证明。
