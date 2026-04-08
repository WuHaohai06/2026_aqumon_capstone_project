# Prompts 说明

这个目录目前和 Form 4 相关的提示词里，最值得重点区分的是：

- `4_structured_input.txt`
- `4_golden_structured_input.txt`

这两个 prompt 都是给结构化的 SEC Form 4 数据做情绪判断，但它们的目标并不完全一样。前者更像“可直接用于批量跑分的生产型 prompt”，后者更像“为了构建高质量 golden dataset 而写的标注规范型 prompt”。

说明：原始 prompt 文本里有少量编码异常字符，例如 `2鈥?`、`鈫?`。下面的内容按语义做了干净的中文整理，不逐字符对照，但保留了原意。

## 1. `4_structured_input.txt` 中文整理

### 1.1 角色与任务

这个 prompt 把模型设定为“擅长分析 SEC Form 4 内部人交易情绪的金融 NLP 分析师”。

它要求模型：

- 一次处理多条结构化 Form 4 记录；
- 对每条 filing 独立输出情绪分数、置信度和解释；
- 从“市场参与者在披露当下看到这份文件时会如何解读”的角度来判断信号强弱。

### 1.2 核心约束

它强调以下规则：

- 每个 filing 必须独立判断，不能互相串信息；
- 不能使用未来股价、后续新闻、后续业绩或任何 hindsight；
- 只能使用 filing 结构化记录中明确给出的信息；
- 如果内容大多是行政性、机械性、信息量弱的交易，分数应靠近中性。

### 1.3 输入形式

输入是一个 `FILINGS` 数组，每个元素类似：

```json
{
  "id": "F1",
  "filing": {
    "form_type": "4",
    "issuer": {},
    "reporting_person": {}
  }
}
```

其中 `__FILINGS_JSON__` 会在运行时替换成真实的结构化数据。

### 1.4 情绪分数定义

情绪分数范围是 `-1.0` 到 `1.0`，步长必须是 `0.1`。

它的判分逻辑偏“市场解读”：

- 强负面：明确的、带主观判断的 insider selling，且卖出规模较大、角色重要、剩余持仓下降显著；
- 弱负面：存在卖出，但可能规模一般、角色一般，或动机不够明确；
- 中性：大概率是税务代扣、归属、自动转换、非自主转移等机械事件；
- 弱正面：存在买入，但规模或角色不足以支撑强烈正面；
- 强正面：明确的、较大规模的自主买入，且 insider 身份较强。

### 1.5 它关注哪些字段

这个 prompt 会引导模型综合看以下信息：

- 报告人身份：CEO、CFO、创始人、董事长的信号权重更高；
- 非衍生和衍生交易行；
- `transaction_code`、买卖方向、交易股数、价格、交易后持股；
- 直接持有还是间接持有；
- footnotes / remarks 里是否说明这是 10b5-1、税务、vesting、conversion、gift、trust 等偏机械事件；
- 交易完成后，insider 是否还保留大量头寸。

### 1.6 置信度定义

置信度并不只是“模型自信程度”，而是由数据清晰度决定，包括：

- 买卖方向是否明确；
- insider 身份是否明确；
- 是自主交易还是机械交易是否明确；
- 交易数量、价格、交易后持股是否完整；
- footnotes 是否足够清楚。

如果信息稀疏、原因模糊、经济意义不明确，就应降低置信度。

### 1.7 输出格式

输出必须是严格 JSON 数组，每个对象包含：

- `id`
- `sentiment_score`
- `confidence`
- `summary`
- `reasons`

其中 `reasons` 下包含：

- `positives`
- `negatives`
- `guidance_tone`
- `qa_tone`

这里的 `guidance_tone` 和 `qa_tone` 实际上是沿用旧 schema 的占位字段：

- `guidance_tone` 被重新解释为“内部人意图或方向性信号”；
- `qa_tone` 被重新解释为“交易更像自主、机械、行政还是模糊”。

### 1.8 额外特点

这个 prompt 还做了几件对“生产稳定性”很友好的事：

- 明确要求输出顺序必须和输入顺序一致；
- 明确要求不能漏掉任何 filing；
- 给了默认兜底值，比如不确定时打 `0.0`、`40%`；
- 提供了若干分数校准锚点，例如 CEO/CFO 的大额买入、卖出、税务代扣等；
- 对 summary 的写法也给了规范。

整体上，它是一个兼顾“业务判断”和“批处理稳定性”的通用评分 prompt。

## 2. `4_golden_structured_input.txt` 中文整理

### 2.1 角色与任务

这个 prompt 把模型设定为“擅长从结构化 SEC Form 4 中解读 insider trading signal 的金融 NLP 分析师”。

它的任务同样是：

- 一次处理多条结构化 Form 4；
- 输出情绪分数、置信度和结构化解释；
- 但它额外强调：目标是产出“高质量、可解释、可用于建模的 GOLDEN DATASET”。

这说明它不只是为了跑结果，更是在追求标签的一致性和可复核性。

### 2.2 核心约束

它也要求：

- 每个 filing 独立判断；
- 不能引入未来信息和外部背景；
- 只能用结构化 JSON；
- 信息弱或模糊时，要主动向中性偏。

相比 `4_structured_input.txt`，它的约束更像“标注员守则”。

### 2.3 情绪分数定义

分数范围同样是 `-1.0` 到 `1.0`，步长 `0.1`。

但它的定义更简洁，不做太多文字解释，而是把重点放在后面的规则化判断框架上。

### 2.4 更严格的结构化解释框架

这个 prompt 最关键的部分在于，它把判断过程拆得更细、更规范。

#### 2.4.1 交易行过滤

它先要求模型判断某一行是否是真正的“有效交易行”：

- `transaction_code` 不能为空；
- 且 `transaction_shares` 或 `transaction_price` 至少有一个不为空。

不满足时，该行应被视为“仅反映持仓状态的记录”，不应直接用于情绪判断。

这一点比 `4_structured_input.txt` 更严格，也更适合减少误标。

#### 2.4.2 聚合逻辑

如果一个 filing 里有多行交易，它要求把多行视为“一个组合信号”来判断，并综合看：

- 总交易股数；
- 买卖方向是否一致；
- 时间是否接近。

它明确规定：

- 多笔一致买入会增强正面；
- 多笔一致卖出会增强负面；
- 买卖混合会把分数往 0 拉。

#### 2.4.3 自主交易 vs 机械交易

它非常强调“机械性交易要降权”，例如：

- `price == 0`
- 某些表明转换或衍生处理的标识字段为真
- footnotes 指向 tax withholding、vesting、conversion、option exercise、10b5-1 plan

这些都应把分数推向中性。

#### 2.4.4 所有权权重

它把 ownership type 直接纳入强度判断：

- 直接持有最强；
- 配偶等间接持有次之；
- trust / entity 等间接持有更弱。

### 2.5 强制的信号拆解

这是它和 `4_structured_input.txt` 最大的不同之一。

它要求模型在内部显式分解以下维度：

1. `direction_signal`：买、卖、混合或不清楚；
2. `insider_strength`：按 CEO / Chairman、CFO、其他高管、董事分层；
3. `size_signal`：大、中、小、不明确；
4. `discretionary_score`：高、中、低；
5. `ownership_signal`：直接、配偶间接、实体间接；
6. `information_quality`：高、中、低。

也就是说，它不只是要一个结论，而是要求模型沿固定维度形成结论。

### 2.6 学术化原则与 few-shot 校准

这个 prompt 还显式写入了一些“经验法则”：

- insider buy 通常比 sell 更有信息量；
- 大多数 sell 都是弱信号，除非规模大且很自主；
- 机械性交易应尽量靠近中性；
- cluster buy 是强信号；
- 间接持有会削弱信号；
- 数据稀疏会降低分数幅度。

同时它还给了更像 few-shot 的校准锚点，例如：

- CEO 大额买入通常是 `+0.6` 到 `+0.9`；
- 10b5-1 下的 CFO 卖出大多接近中性或轻微负面；
- 董事小额买入通常只是轻微正面；
- vesting / 税务代扣通常为 `0.0`；
- cluster buy 是较强正面；
- 混合买卖应接近 `0`。

### 2.7 输出格式

输出仍然是 JSON 数组，但字段更丰富。除基础字段外，还新增：

- `signal_breakdown`
- `decision_factors`
- `risk_flags`

其中 `signal_breakdown` 明确要求输出：

- `direction`
- `insider_role_strength`
- `size_assessment`
- `discretionary_level`
- `ownership_type`
- `information_quality`

这让结果更像“可审计标签”而不是“普通模型回答”。

### 2.8 整体特点

这个 prompt 的整体风格更短、更硬、更规整。

它不像 `4_structured_input.txt` 那样花很多篇幅解释背景，而是把重点放在：

- 统一标准；
- 显式分解；
- 降低标注歧义；
- 让输出更适合做 golden dataset 和后续模型训练。

## 3. 两者的具体差异

### 3.1 目标定位不同

`4_structured_input.txt` 的目标更偏“让模型稳定完成批量情绪打分”。

`4_golden_structured_input.txt` 的目标更偏“让标签更可解释、更一致、更像人工标注规范”。它不是单纯求能跑，而是求标签质量。

### 3.2 决策方式不同

`4_structured_input.txt` 主要是“描述性引导”：

- 告诉模型应该关注哪些因素；
- 用自然语言解释什么是强正面、弱负面、中性；
- 给模型较大的综合判断空间。

`4_golden_structured_input.txt` 主要是“规则化约束”：

- 先过滤交易行；
- 再聚合多行；
- 再判断自主程度；
- 再考虑所有权；
- 再给出分解结果。

换句话说，前者更像“高级分析师提示词”，后者更像“标准化标注流程”。

### 3.3 对多行交易的处理不同

`4_structured_input.txt` 会提醒模型看 transaction rows，但没有把“多行如何合并”写成很强的操作规则。

`4_golden_structured_input.txt` 明确规定了 cluster logic，这对包含多笔买卖、衍生和非衍生混合记录的 filing 特别重要。它更能避免模型只盯住某一行、忽略整体结构。

### 3.4 对噪声行和机械交易的处理不同

`4_structured_input.txt` 也提到 option exercise、tax withholding、vesting 等，但更像提醒。

`4_golden_structured_input.txt` 则把这些内容写成“强降权规则”，甚至从交易行过滤阶段就开始控制噪声。这会显著降低把行政性 filing 误判成强情绪信号的风险。

### 3.5 输出结构不同

`4_structured_input.txt` 输出的核心是：

- 分数
- 置信度
- summary
- 正负面原因

这足够支持日常分析和结果落库。

`4_golden_structured_input.txt` 则额外要求：

- 信号分解
- 决策因素
- 风险标记

这使它更适合：

- 人工复核；
- 结果对比；
- 训练样本质检；
- 研究标签漂移来源。

### 3.6 对“卖出信号”的态度不同

`4_structured_input.txt` 会把卖出也作为常规负面信号处理，只是提醒要考虑动机、角色和剩余持仓。

`4_golden_structured_input.txt` 则更明确地表达了一个经验立场：

- buy 一般比 sell 更有信息量；
- 大多数 sell 默认不应打得太重，除非规模大、角色强、且明显是自主行为。

这会让 golden 版的标签整体更保守，尤其是在卖出样本上。

### 3.7 一致性与复现性不同

如果目标是“快速跑出一套业务可用结果”，`4_structured_input.txt` 已经足够强。

如果目标是“不同时间、不同批次、不同样本都尽量保持同一套打标标准”，`4_golden_structured_input.txt` 更占优，因为它把很多容易漂移的判断点前置并结构化了。

## 4. 哪个更适合什么场景

### 更适合用 `4_structured_input.txt` 的场景

- 你想快速批量跑 Form 4 情绪结果；
- 你更看重 summary 的自然语言可读性；
- 你希望 prompt 对复杂样本保留一定灵活判断空间；
- 你当前任务更偏“生成业务结果”而不是“构建标注基准”。

### 更适合用 `4_golden_structured_input.txt` 的场景

- 你在做 golden dataset；
- 你需要更稳定、更容易复核的标签；
- 你后面准备做 prompt 对比、模型蒸馏或监督训练；
- 你希望把“为什么打这个分”拆得更结构化；
- 你想降低机械交易、噪声交易、多行混合交易带来的误标风险。

## 5. 我的建议

如果把这两个 prompt 放在一个完整流程里，我会这样分工：

- `4_structured_input.txt`：适合作为日常批量生产版 prompt；
- `4_golden_structured_input.txt`：适合作为高质量评估集、对照集、训练集的标注版 prompt。

更具体一点：

- 如果你关心“结果是否更贴近市场直觉”，`4_structured_input.txt` 往往更自然；
- 如果你关心“标签是否更稳定、更可解释、更适合后续建模”，`4_golden_structured_input.txt` 更合适。

从方法论上看，两者不是谁替代谁，而是各自承担不同角色：

- 一个偏生产；
- 一个偏标注标准。

如果后面你要继续演进 prompt，我建议优先吸收 `4_golden_structured_input.txt` 里的三类优点，再回灌到生产版：

- 交易行过滤；
- cluster logic；
- `signal_breakdown` 这一层的显式推理结构。

这样通常能在不牺牲可用性的前提下，提高标签一致性。
