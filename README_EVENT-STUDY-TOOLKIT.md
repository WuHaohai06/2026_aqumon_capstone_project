# Event-Study-Toolkit 与 AQUMON Form 4 事件研究流程说明

这份文档用来说明两件事：

1. 本地 `event-study-toolkit` 源码的输入要求、指标计算方式和整体计算链路
2. 当前 AQUMON 仓库里的 Form 4 数据，是怎样一步步被处理成事件研究可用输入的

本地 toolkit 源码位置：

- `event-study-toolkit-main/event-study-toolkit-main/README.md`
- `event-study-toolkit-main/event-study-toolkit-main/src/event_study_toolkit/eventstudy.py`

## 1. `event-study-toolkit` 需要什么输入

这个 toolkit 需要两张长表。

### 1.1 收益率数据表 `data panel`

一行代表一只证券在一个交易日的数据，也就是 `security x trading_date`。

基础必需列：

- `date`：交易日，类型应为 `datetime64[ns]`
- `unique_id`：证券唯一标识列，类的默认列名是 `permno`，但可以通过 `unique_id=...` 自定义

不同模型需要的收益率/因子列：

- `market` 模型：`ret_dlst_adj`, `vwretd`
- `capm` 模型：`ret_dlst_adj`, `rf_h15`, `mktrf_h15`
- `famafrench` 模型：`ret_dlst_adj`, `mktrf_h15`, `smb`, `hml`
- 自定义 OLS 公式：公式中出现的列都必须存在于合并后的数据表里

### 1.2 事件表 `event panel`

一行代表一个事件。

必需列：

- `unique_id`：和收益率数据表使用同一个证券标识
- `EVT_DATE`：事件日期

可选列：

- 事件分组列，例如 `sentiment_bucket`

## 2. Toolkit 的核心指标是怎么算的

这个 toolkit 的基本思路是：

1. 在估计窗口里拟合“正常收益模型”
2. 在事件窗口里预测正常收益
3. 用“实际收益 - 预测收益”得到异常收益
4. 再把异常收益累加成 CAR，并做显著性检验

### 2.1 窗口定义

构造类时最重要的 4 个参数：

- `estperiod`：估计窗口长度
- `gap`：估计窗口结束到事件窗口开始之间的空档期
- `start`：事件窗口起始相对日
- `end`：事件窗口结束相对日

toolkit 会据此构造一个交易日映射表，包含：

- `EST_START`
- `EST_END`
- `EVT_START`
- `EVT_DATE`
- `EVT_END`

### 2.2 正常收益模型拟合

toolkit 会对每只证券，在估计窗口内单独跑一遍 OLS。

例如 `market` 模型：

```text
ret_dlst_adj ~ vwretd
```

例如 `capm` 模型：

```text
ret_dlst_adj ~ rf_h15 + mktrf_h15
```

例如 `famafrench` 模型：

```text
ret_dlst_adj ~ mktrf_h15 + smb + hml
```

### 2.3 预测收益 `predicted_return`

对事件窗口内每一天，预测正常收益：

```text
predicted_return_it = X_it * beta_i
```

其中 `beta_i` 是证券 `i` 在估计窗口里拟合出来的系数。

### 2.4 异常收益 `abret`

toolkit 的定义是：

```text
abret_it = actual_return_it - predicted_return_it
```

对于标准模型，`actual_return_it` 就是 `ret_dlst_adj`。

### 2.5 估计窗口残差与 `sigma`

在估计窗口内，残差定义为：

```text
e_it = actual_return_it - fitted_return_it
```

然后 toolkit 计算：

```text
sigma_i = sqrt( sum(e_it^2) / (n_i - k_i) )
```

这里：

- `n_i`：证券 `i` 在估计窗口中的样本数
- `k_i`：回归模型的参数个数

还会继续计算：

```text
sar_it = e_it / sigma_i
```

以及估计窗口内“正残差占比”：

```text
e_est_pos_i = mean( 1[e_it > 0] )
```

### 2.6 累积异常收益 `CAR`

若事件窗口长度为：

```text
N = end - start + 1
```

则：

```text
CAR_i = sum_t abret_it
```

也就是把一个事件窗口内所有异常收益加总。

### 2.7 标准化累积异常收益 `SCAR`

toolkit 用下面这个公式标准化 CAR：

```text
SCAR_i = sqrt(1 / N) * CAR_i / sigma_i
```

### 2.8 `scar_bmp`

源码里还会计算一个额外指标：

```text
scar_bmp_i = (CAR_i / sigma_i) / std(SCAR)
```

这个指标在 toolkit 里主要是后续 `GRANK` 排名统计时使用。源码注释对它的统计意义解释得不完整，所以在 AQUMON 里更适合把它理解为“toolkit 内部使用的一种标准化 CAR 变体”，而不是单独对外解释的主指标。

### 2.9 正 CAR 指示变量 `poscar`

源码里还会记录：

```text
poscar_i = 1 if CAR_i > 0 else 0
```

## 3. Toolkit 输出哪些显著性统计量

### 3.1 全样本统计

`getFullSampleTestStatistic(modelType)` 会返回一行全样本汇总结果。

主要字段：

- `car_mean`：CAR 均值
- `scar_mean`：SCAR 均值
- `poscar_mean`：正 CAR 事件占比
- `poscar_cnt`：正 CAR 事件数
- `evt_count`：事件数
- `e_est_pos`：估计窗口正残差比例的均值
- `car_t`：CAR 关于 0 的单样本 t 统计量
- `scar_t`：SCAR 关于 0 的单样本 t 统计量
- `tsign`：Sign Test 统计量
- `tpatell`：Patell 统计量
- `gen_z`：Generalized Sign Test 统计量

源码中的公式：

```text
tsign = (poscar_mean - 0.5) / sqrt(0.25 / evt_count)
tpatell = scar_mean * sqrt(evt_count)
gen_z = (poscar_cnt - evt_count * e_est_pos) / sqrt(evt_count * e_est_pos * (1 - e_est_pos))
```

### 3.2 分组统计

`getGroupLevelTestStatistics(modelType, GRP)` 会按分组列计算同一套统计量。

相比全样本，还会多出：

- `car_median`
- `scar_median`

在我们当前项目里，最自然的分组方式就是 `sentiment_bucket`。

### 3.3 Generalized Rank Z

`getGRANK(modelType, GRP)` 会把：

- 估计窗口里的 `sar`
- 事件窗口里的 `scar_bmp`

拼起来做组内排序，再计算 Generalized Rank Z 统计量。

### 3.4 Wilcoxon Rank-Sum

`getWilcoxon(modelType, GRP)` 会对 `scar` 做两组 Wilcoxon rank-sum 检验。

所以这个函数最适合处理二元分组，例如：

- `positive`
- `negative`

## 4. Toolkit 的完整计算链路

`eventstudy.py` 里的主链路可以概括成：

1. `getTradingCalendar()` / `generateTradingCalendar()`：构造交易日历和事件窗口映射
2. `getDataFrame()`：把 `data + events + calendar` 合并成一张大表
3. `createEstChunk()`：切出每只证券的估计窗口样本
4. `createEvtChunk()`：切出每只证券的事件窗口样本
5. `fitModel()` / `runModel()`：逐证券拟合 OLS
6. `getPredictedReturns()`：预测事件窗口内的正常收益
7. `getAbnormalReturns()`：计算异常收益
8. `getCARS()`：计算 CAR、SCAR、`poscar` 等指标
9. 再根据需要调用：
   - `getFullSampleTestStatistic()`
   - `getGroupLevelTestStatistics()`
   - `getGRANK()`
   - `getWilcoxon()`

## 5. 当前 AQUMON 的 Form 4 数据是怎么走到事件研究这一步的

目前 AQUMON 的链路还不是“直接把原始数据喂给 toolkit”，而是先通过一套脚本把 Form 4 做成事件研究可用的中间数据，再由 notebook 做事件研究和 toolkit 兼容导出。

## 5.1 `download_filing_metadata.py`：收集 filing 元数据

生产脚本：

- `scripts/download/download_filing_metadata.py`

主要输入：

- `us_symbol_list.xlsx`
- SEC submissions JSON 接口

主要输出：

- `data/raw_data/filing_metadata.csv`

输出格式：

- CSV
- 一行对应一条 filing

字段：

- `cik`
- `ticker`
- `form`
- `filingDate`
- `acceptanceDateTime`
- `accessionNumber`

字段含义里最关键的两点：

- `accessionNumber`：后面会作为情绪结果的主连接键
- `acceptanceDateTime`：比 `filingDate` 更精确，是事件时间对齐最重要的字段

## 5.2 `form4_structured_extraction.py`：把 Form 4 文本抽成结构化 JSON

生产脚本：

- `scripts/extraction/form4_structured_extraction.py`

主要输入：

- 原始 Form 4 文本目录：
  - `data/golden_dataset_engine/extracted_raw_data/4/<ticker>/<accession>/<accession>.txt`
- prompt：
  - `prompts/4_extraction_v1.txt`

主要输出：

- `data/results/4/<prompt_name>_<model_name>/structured_json/<ticker>/<accession>/<accession>.json`
- `structured_success.jsonl`
- `structured_failed.jsonl`
- `raw_responses/...`
- `completion_markers/...`

结构化 JSON 的格式：

- 一个 filing 对应一个 JSON 文件
- 是嵌套对象，不是扁平表

常见顶层字段：

- `form_type`
- `is_amendment`
- `filing_date`
- `date_of_earliest_transaction`
- `issuer`
- `reporting_person`
- `filing_characteristics`
- `table_i_non_derivative`
- `table_ii_derivative`
- `footnotes`
- `remarks`

其中最重要的是：

- `table_i_non_derivative`
- `table_ii_derivative`

这两个字段是交易记录数组，后面的情绪分析就是基于这些结构化交易信息来做的。

## 5.3 `sentiment_structured.py`：对结构化 Form 4 做情绪/信号打分

生产脚本：

- `scripts/analysis/sentiment_structured.py`

主要输入：

- 结构化 JSON 目录，例如：
  - `data/results/4/4_extraction_v1_gemini/structured_json`
- prompt：
  - `prompts/4_golden_structured_input.txt`

当前仓库里常见的输出目录：

- `data/results/4/4_golden_structured_input_gemi_batch/`

主要输出：

- `sentiment_results.jsonl`
- `sentiment_results.parquet`
- `failed_transcripts.jsonl`
- `completion_markers/...`
- `sentiment_pipeline.log`

其中 JSONL 是更接近“原始标准输出”的格式。

单条 JSONL 记录的典型结构：

- `id`
- `sentiment_score`
- `confidence`
- `summary`
- `signal_breakdown`
- `decision_factors`
- `risk_flags`
- `reasons`

对应的扁平化 Parquet 字段：

- `id`
- `sentiment_score`
- `confidence`
- `summary`
- `signal_direction`
- `signal_insider_role_strength`
- `signal_size_assessment`
- `signal_discretionary_level`
- `signal_ownership_type`
- `signal_information_quality`
- `decision_factors`
- `risk_flags`
- `reasons_positives`
- `reasons_negatives`
- `reasons_guidance_tone`
- `reasons_qa_tone`

这里最关键的是：

- 一行对应一个 filing
- `id` 就是 accession number

也就是说，情绪打分和 `filing_metadata.csv` 的连接键已经天然对齐了。

## 5.4 `build_sentiment_timeseries.py`：把 filing 级信号对齐到交易日

生产脚本：

- `scripts/analysis/build_sentiment_timeseries.py`

主要输入：

- sentiment 结果目录或文件：
  - `data/results/4/...`
- filing 元数据：
  - `data/raw_data/filing_metadata.csv`
- 复权收盘价矩阵：
  - `data/close_adj.csv`

输入格式说明：

- `close_adj.csv` 是宽表
- 第一列是日期
- 后面每一列是一只股票，例如 `A.US`、`AAPL.US`

脚本内部的映射逻辑：

1. `sentiment_results.id` 对齐 `filing_metadata.accessionNumber`
2. `filing_metadata.ticker` 转成 `price_ticker = ticker + ".US"`
3. 事件时间默认使用 `acceptanceDateTime`
4. 再把事件时间映射到一个交易日

当前默认对齐规则：

- 时间锚点：`acceptanceDateTime`
- 生效规则：`next_trading_day`

也就是：

- filing 如果在某个自然日披露
- 信号默认落到下一个交易日

这样做是为了避免未来函数。

主要输出：

- `event_level_sentiment.parquet`
- `daily_sentiment_events_long.parquet`
- `sentiment_score_mean_aligned_to_close_adj.parquet`
- `sentiment_score_sum_aligned_to_close_adj.parquet`
- `sentiment_score_last_aligned_to_close_adj.parquet`
- `confidence_pct_mean_aligned_to_close_adj.parquet`
- `filing_count_aligned_to_close_adj.parquet`
- `alignment_config.json`

### 5.4.1 `event_level_sentiment.parquet`

格式：

- 一行对应一个 filing 事件
- 是 sentiment 结果与 metadata 合并后的事件级表

当前样本里观察到的字段：

- `id`
- `sentiment_score`
- `confidence`
- `summary`
- `signal_breakdown`
- `decision_factors`
- `risk_flags`
- `reasons`
- `sentiment_reasoning`
- `sentiment_factor`
- `sentiment_tone`
- `confidence_pct`
- `source_path`
- `ticker`
- `price_ticker`
- `form`
- `filingDate`
- `filing_date`
- `acceptanceDateTime`
- `acceptance_datetime_utc`
- `acceptance_datetime_local`
- `effective_trade_date`
- `close_adj_on_effective_date`
- `has_metadata_match`
- `has_close_adj_match`

### 5.4.2 `daily_sentiment_events_long.parquet`

格式：

- 稀疏长表
- 一行代表一个 `date x ticker`
- 只保留当日确实有 filing 事件的股票

字段：

- `date`
- `price_ticker`
- `ticker`
- `sentiment_score_mean`
- `sentiment_score_sum`
- `sentiment_score_last`
- `confidence_pct_mean`
- `filing_count`

如果同一只股票在同一天有多份 filing，聚合规则是：

- `sentiment_score_mean`：平均值
- `sentiment_score_sum`：求和
- `sentiment_score_last`：按 `acceptance_datetime_utc` 排序后取最后一条
- `confidence_pct_mean`：平均置信度
- `filing_count`：文件数

如果某只股票某天没有 filing：

- 在这个稀疏长表里，不会出现这一行
- 在宽表输出里，默认会填 `0.0`
- 如果运行时传 `--fill-value nan`，则会保留为空值

### 5.4.3 宽表对齐输出

这些文件的格式都是：

- 行索引与 `close_adj.csv` 完全一致
- 列与 `close_adj.csv` 完全一致
- 本质上是日频因子矩阵

对应文件：

- `sentiment_score_mean_aligned_to_close_adj.parquet`
- `sentiment_score_sum_aligned_to_close_adj.parquet`
- `sentiment_score_last_aligned_to_close_adj.parquet`
- `confidence_pct_mean_aligned_to_close_adj.parquet`
- `filing_count_aligned_to_close_adj.parquet`

这些宽表很适合做日频回测，但它们还不是 `event-study-toolkit` 直接需要的“长表事件研究输入”。

## 5.5 `form4_event_study_draft.ipynb`：当前事件研究草稿与 toolkit 桥接

当前 notebook：

- `form4_event_study_draft.ipynb`

当前输出目录：

- `data/results/4/event_study_draft/`

主要输入：

- `daily_sentiment_events_long.parquet`
- `data/close_adj.csv`

当前 notebook 的处理逻辑：

1. 用 `close_adj.csv` 计算日收益率 `pct_change()`
2. 把收益率裁剪到 `[-0.8, 0.8]`
3. 用横截面等权平均收益构造一个临时市场收益 `market_ret`
4. 仅保留能映射到收益率数据的事件
5. 按阈值过滤弱信号或中性信号
6. 去掉同一只股票上彼此重叠的事件窗
7. 在本地手工跑一个 market model：
   - estimation window = `120`
   - gap = `5`
   - event window = `[-1, 5]`

主要输出：

- `toolkit_data.parquet`
- `toolkit_events.parquet`
- `abnormal_returns.parquet`
- `event_level_car.parquet`
- `aar_caar_summary.csv`
- `aar_caar_by_bucket.csv`

### 5.5.1 `toolkit_data.parquet`

格式：

- 长表
- 一行代表一个 `date x price_ticker`

当前样本字段：

- `date`
- `price_ticker`
- `ret_dlst_adj`
- `vwretd`

字段解释：

- `ret_dlst_adj`：由 `close_adj.csv` 算出的个股收益率
- `vwretd`：当前 notebook 里构造的等权市场代理收益，不是真正的 CRSP value-weighted market return

### 5.5.2 `toolkit_events.parquet`

格式：

- 一行代表一个事件

当前样本字段：

- `price_ticker`
- `EVT_DATE`
- `ticker`
- `sentiment_bucket`
- `sentiment_score_mean`
- `filing_count`

这是当前仓库里最接近 `event-study-toolkit` 原生事件输入格式的一张表。

### 5.5.3 `abnormal_returns.parquet`

格式：

- 一行代表一个 `event x relative_day`

当前样本字段：

- `date`
- `ret`
- `market_ret`
- `relative_day`
- `expected_ret`
- `abnormal_ret`
- `event_id`
- `event_date`
- `price_ticker`
- `ticker`
- `sentiment_score_mean`
- `filing_count`
- `sentiment_bucket`
- `alpha`
- `beta`

### 5.5.4 `event_level_car.parquet`

格式：

- 一行代表一个事件

当前样本字段：

- `event_id`
- `event_date`
- `price_ticker`
- `ticker`
- `sentiment_bucket`
- `sentiment_score_mean`
- `filing_count`
- `car`

### 5.5.5 汇总 CSV

`aar_caar_summary.csv` 字段：

- `relative_day`
- `aar`
- `median`
- `aar_std`
- `count`
- `t_stat`
- `p_value`
- `caar`

`aar_caar_by_bucket.csv` 字段：

- `sentiment_bucket`
- `relative_day`
- `aar`
- `aar_std`
- `count`
- `caar`

## 6. 现在的数据能不能直接喂给 `event-study-toolkit`

还不能完全“开箱即用”，但已经非常接近。

目前有 4 个关键兼容性问题。

### 6.1 toolkit 内部交易日历截止到 2023-07-10

本地源码 `getTradingCalendar()` 里把交易日历结束日期硬编码成了：

```text
2023-07-10
```

但我们当前导出的事件样本日期范围已经是：

- `toolkit_events.parquet` 最早：`2025-01-03`
- `toolkit_events.parquet` 最晚：`2026-01-29`

所以如果不改源码，当前 AQUMON 事件样本无法直接被 toolkit 正确处理。

### 6.2 toolkit 对字符串证券 ID 支持不好

在 `getResiduals()` 里，源码会把证券分组键强制写成：

```text
int(x)
```

这意味着像 `A.US`、`AAPL.US` 这种字符串 ID 很可能直接报错。

可选处理方案：

1. 直接 patch toolkit，去掉 `int(x)`
2. 先在外部构造数值型 `sec_id`，再把它作为 `unique_id`

### 6.3 `getGRANK()` 里还写死了 `CCAR`

虽然函数签名是 `getGRANK(modelType, GRP)`，但源码内部有一部分分组逻辑仍然硬编码成 `CCAR`。

这意味着：

- 如果你要直接跑 `GRANK`
- 你要么把分组列改名成 `CCAR`
- 要么先 patch 源码，把内部 `CCAR` 替换成传入的 `GRP`

### 6.4 `groups` 最好不要留空成 `None`

后续多个方法默认会使用 `self.groups`。

实际调用时更稳妥的做法是：

- 不分组时传 `groups=[]`
- 按情绪分组时传 `groups=["sentiment_bucket"]`

## 7. 当前 AQUMON 最推荐的接法

如果要把当前仓库真正接到 `event-study-toolkit`，我建议走下面这条路径：

1. 用 `sentiment_structured.py` 生成 filing 级情绪结果
2. 用 `build_sentiment_timeseries.py` 把 filing 信号对齐到交易日
3. 用 `form4_event_study_draft.ipynb` 导出：
   - `toolkit_data.parquet`
   - `toolkit_events.parquet`
4. 在正式调用 toolkit 前，先处理兼容性：
   - 扩展交易日历到当前样本年份
   - 把证券 ID 转成数值型，或 patch `int(x)`
   - 如果要用 `GRANK`，修掉 `CCAR` 写死的问题
5. 再实例化 toolkit

适配后的最小输入结构建议是：

- 收益率数据表：
  - `date`
  - `sec_id`
  - `ret_dlst_adj`
  - `vwretd`
- 事件表：
  - `sec_id`
  - `EVT_DATE`
  - `sentiment_bucket`

示例代码：

```python
import pandas as pd
from event_study_toolkit.eventstudy import eventstudy

data = pd.read_parquet("data/results/4/event_study_draft/toolkit_data.parquet")
events = pd.read_parquet("data/results/4/event_study_draft/toolkit_events.parquet")

id_map = {
    ticker: i
    for i, ticker in enumerate(sorted(data["price_ticker"].unique()), start=1)
}

data["sec_id"] = data["price_ticker"].map(id_map)
events["sec_id"] = events["price_ticker"].map(id_map)

es = eventstudy(
    estperiod=120,
    gap=5,
    start=-1,
    end=5,
    data=data[["date", "sec_id", "ret_dlst_adj", "vwretd"]],
    events=events[["sec_id", "EVT_DATE", "sentiment_bucket"]],
    unique_id="sec_id",
    calType="NYSE",
    groups=["sentiment_bucket"],
)
```

## 8. 一句话总结

- `event-study-toolkit` 本质上是一个“经典事件研究统计引擎”
- AQUMON 目前已经能把 Form 4 情绪结果加工到非常接近 toolkit 输入的状态
- 当前最接近 toolkit 输入的两张表是：
  - `data/results/4/event_study_draft/toolkit_data.parquet`
  - `data/results/4/event_study_draft/toolkit_events.parquet`
- 真正直接接 toolkit 之前，还需要补 3 个兼容点：
  - 交易日历年份范围
  - 字符串证券 ID
  - `GRANK` 里的 `CCAR` 硬编码
