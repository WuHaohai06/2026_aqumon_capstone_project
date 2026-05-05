# Event Study Toolkit Formula and Data Guide

本文档按 `src/event_study_toolkit/eventstudy.py` 的实际实现整理，说明事件研究流程、公式、输入输出数据格式，以及数据处理方法。

## 1. 核心对象

主要入口是：

```python
from event_study_toolkit import eventstudy as es

study = es.eventstudy(
    estperiod=50,
    gap=5,
    start=-1,
    end=1,
    data=data,
    events=events,
    unique_id="subject_id",
    calType="NYSE",
    groups=["CCAR", "GSIB"],
)
```

参数含义：

| 参数 | 类型 | 含义 |
| --- | --- | --- |
| `estperiod` | int | 估计期长度，必须为正数 |
| `gap` | int | 估计期结束和事件窗口开始之间空出的交易日数量 |
| `start` | int | 事件窗口起点，相对事件日的交易日偏移，例如 `-1` 表示事件日前一交易日 |
| `end` | int | 事件窗口终点，相对事件日的交易日偏移，例如 `1` 表示事件后一交易日 |
| `data` | pandas.DataFrame | 证券日度收益和因子数据 |
| `events` | pandas.DataFrame | 事件日期和事件分组数据 |
| `unique_id` | str | 证券唯一标识列名，默认是 `permno` |
| `calType` | str | `pandas_market_calendars` 支持的交易日历名称，默认 `NYSE` |
| `groups` | list[str] | 分组列名列表，例如 `["CCAR", "GSIB"]` |

注意：当前实现中，很多统计函数会使用 `self.groups`。实际使用时建议总是显式传入 `groups`，即使只需要一个分组。

## 2. 输入数据格式

### 2.1 `data` 收益面板

`data` 至少需要包含：

| 列名 | 必需性 | 含义 |
| --- | --- | --- |
| `date` | 必需 | 交易日，建议为 `datetime64[ns]` |
| `unique_id` 对应列 | 必需 | 证券 ID，例如 `permno` 或 `subject_id` |
| 模型所需的收益列和因子列 | 必需 | 取决于 `modelType` |

内置示例 `data_panel.csv` 的列包括：

```text
date, subject_id, ret_dlst_adj, rf_h15, vwretd, smb, hml, def, term, mktrf_h15
```

标准模型需要的列：

| `modelType` | 需要的列 |
| --- | --- |
| `market` | `ret_dlst_adj`, `vwretd` |
| `famafrench` | `ret_dlst_adj`, `mktrf_h15`, `smb`, `hml` |
| `capm` | `ret_dlst_adj`, `mktrf_h15`, `rf_h15` |
| 自定义公式 | 公式左侧响应变量和右侧解释变量都必须存在于 `dataFrame` 中 |

### 2.2 `events` 事件表

`events` 至少需要包含：

| 列名 | 必需性 | 含义 |
| --- | --- | --- |
| `unique_id` 对应列 | 必需 | 证券 ID，必须能和 `data` 合并 |
| `EVT_DATE` | 必需 | 事件日，会被转换为 `datetime64[ns]` |
| 分组列 | 可选但建议 | 例如 `CCAR`, `GSIB` |

内置示例 `event_panel.csv` 的列包括：

```text
subject_id, EVT_DATE, CCAR, GSIB
```

重要限制：

1. `EVT_DATE` 必须能匹配交易日历中的交易日。若事件日是周末、假日，当前实现不会自动顺延或回溯，而是在合并交易日历时丢失该事件。
2. 当前交易日历在代码中固定生成到 `2023-07-10`。晚于该日期的事件不会被匹配到。
3. 如果同一个证券有多个事件，估计模型按 `unique_id` 分组，不按单个事件分组。因此同一证券的多个事件会共用一个按该证券聚合后的估计模型。

## 3. 交易日窗口生成

代码先用 `pandas_market_calendars` 生成交易日序列，然后为每个可能的 `EVT_DATE` 生成如下窗口：

| 字段 | 含义 |
| --- | --- |
| `EST_START` | 估计期开始日 |
| `EST_END` | 估计期结束日 |
| `EVT_START` | 事件窗口开始日 |
| `EVT_DATE` | 事件日 |
| `EVT_END` | 事件窗口结束日 |

若事件日为 $\tau$，所有偏移都按交易日计算，则：

$$
EVT\_START = \tau + start
$$

$$
EVT\_END = \tau + end
$$

$$
EST\_END = \tau + start - gap - 1
$$

$$
EST\_START = \tau + start - gap - estperiod
$$

估计期包含 `EST_START` 到 `EST_END`，事件期包含 `EVT_START` 到 `EVT_END`，两端都是闭区间。

事件窗口长度在后续标准化时使用：

$$
N = end - start + 1
$$

## 4. 数据合并和分块

### 4.1 合并过程

`getDataFrame()` 执行两步合并：

1. `data` 和 `events` 按 `unique_id` 合并。
2. 上一步结果和交易日历按 `EVT_DATE` 合并，补充 `EST_START`, `EST_END`, `EVT_START`, `EVT_END`。

输出 `dataFrame` 包含原始日度数据、事件分组列和窗口边界列。

### 4.2 估计期分块

`createEstChunk()` 筛选：

$$
EST\_START \le date \le EST\_END
$$

然后按 `unique_id` 分组。

### 4.3 事件期分块

`createEvtChunk()` 筛选：

$$
EVT\_START \le date \le EVT\_END
$$

然后按 `unique_id` 分组。

## 5. 正常收益模型

模型使用 `statsmodels.formula.api.ols` 在估计期数据上按证券分别拟合。

### 5.1 Market Model

调用：

```python
study.fitModel("market")
study.runModel("market")
```

源码公式：

```python
ret_dlst_adj ~ vwretd
```

数学形式：

$$
R_{i,t} = \alpha_i + \beta_i R_{m,t} + \epsilon_{i,t}
$$

其中：

| 符号 | 源码列 |
| --- | --- |
| $R_{i,t}$ | `ret_dlst_adj` |
| $R_{m,t}$ | `vwretd` |

### 5.2 Fama-French 三因子模型

调用：

```python
study.fitModel("famafrench")
study.runModel("famafrench")
```

源码公式：

```python
ret_dlst_adj ~ mktrf_h15 + smb + hml
```

数学形式：

$$
R_{i,t} = \alpha_i + \beta_{mkt,i}MKT\_RF_t + \beta_{smb,i}SMB_t + \beta_{hml,i}HML_t + \epsilon_{i,t}
$$

其中：

| 符号 | 源码列 |
| --- | --- |
| $R_{i,t}$ | `ret_dlst_adj` |
| $MKT\_RF_t$ | `mktrf_h15` |
| $SMB_t$ | `smb` |
| $HML_t$ | `hml` |

### 5.3 CAPM

调用：

```python
study.fitModel("capm")
study.runModel("capm")
```

源码公式：

```python
ret_dlst_adj ~ rf_h15 + mktrf_h15
```

数学形式：

$$
R_{i,t} = \alpha_i + \beta_{rf,i}RF_t + \beta_{mkt,i}MKT\_RF_t + \epsilon_{i,t}
$$

其中：

| 符号 | 源码列 |
| --- | --- |
| $R_{i,t}$ | `ret_dlst_adj` |
| $RF_t$ | `rf_h15` |
| $MKT\_RF_t$ | `mktrf_h15` |

注意：这里的 CAPM 是按源码实现描述的。它没有先把个股收益转换成超额收益，而是把 `rf_h15` 和 `mktrf_h15` 都作为解释变量。

### 5.4 自定义模型

也可以直接传入 statsmodels 公式字符串：

```python
study.fitModel("ret_dlst_adj ~ vwretd + smb + hml")
```

代码会解析 `~` 左右两侧的列名，并检查这些列是否存在。自定义公式左侧变量会被后续异常收益计算作为实际收益列。

## 6. 预测收益和异常收益

### 6.1 预测正常收益

`getPredictedReturns(modelType)` 对事件期数据调用已拟合模型：

$$
\widehat{R}_{i,t} = \widehat{\alpha}_i + \sum_j \widehat{\beta}_{i,j}X_{j,t}
$$

输出会在事件期面板中新增：

| 列名 | 含义 |
| --- | --- |
| `predicted_return` | 模型预测的正常收益 |

### 6.2 异常收益 AR

`getAbnormalReturns(modelType)` 计算：

$$
AR_{i,t} = R_{i,t} - \widehat{R}_{i,t}
$$

标准模型中：

$$
R_{i,t} = ret\_dlst\_adj
$$

自定义模型中：

$$
R_{i,t} = \text{公式左侧的响应变量}
$$

输出新增：

| 列名 | 含义 |
| --- | --- |
| `abret` | 异常收益 |

## 7. 模型残差和标准化

`getModelErrors(modelType)` 先在估计期内取得每只证券的 OLS 残差：

$$
e_{i,t} = R_{i,t} - \widehat{R}_{i,t}
$$

然后计算残差符号：

$$
e\_sign_{i,t} =
\begin{cases}
1, & e_{i,t} > 0 \\
0, & e_{i,t} \le 0
\end{cases}
$$

每只证券的残差标准差按 RMSE 形式计算：

$$
\sigma_i = \sqrt{\frac{\sum_{t \in EST_i} e_{i,t}^2}{n_i - k_i}}
$$

其中：

| 符号 | 含义 |
| --- | --- |
| $n_i$ | 估计期残差数量 |
| $k_i$ | OLS 参数数量，包含截距 |

估计期标准化异常收益：

$$
SAR_{i,t} = \frac{e_{i,t}}{\sigma_i}
$$

估计期正残差比例：

$$
\bar{p}_i = \frac{1}{n_i}\sum_{t \in EST_i} e\_sign_{i,t}
$$

源码输出中该值命名为 `e_est_pos`。

## 8. 累计异常收益 CAR

`getCARS(modelType)` 按证券和事件日汇总事件期异常收益：

$$
CAR_i = \sum_{t \in EVT_i} AR_{i,t}
$$

输出列 `car` 即为 $CAR_i$。

### 8.1 标准化累计异常收益 SCAR

源码计算：

$$
SCAR_i = \sqrt{\frac{1}{N}} \cdot \frac{CAR_i}{\sigma_i}
$$

等价于：

$$
SCAR_i = \frac{CAR_i}{\sigma_i\sqrt{N}}
$$

其中：

$$
N = end - start + 1
$$

输出列名为 `scar`。

### 8.2 `scar_bmp`

源码计算：

$$
scar\_bmp_i = \frac{CAR_i / \sigma_i}{std(SCAR)}
$$

其中 `std(SCAR)` 是当前 `cars["scar"].std()`，即 pandas 默认的样本标准差。

输出列名为 `scar_bmp`。

### 8.3 正 CAR 指示变量

源码计算：

$$
poscar_i =
\begin{cases}
1, & CAR_i > 0 \\
0, & CAR_i \le 0
\end{cases}
$$

输出列名为 `poscar`。

## 9. 全样本统计量

`getFullSampleTestStatistic(modelType)` 对所有事件样本计算统计量。

设事件数量为 $M$。

### 9.1 均值和计数

$$
\overline{CAR} = \frac{1}{M}\sum_i CAR_i
$$

$$
\overline{SCAR} = \frac{1}{M}\sum_i SCAR_i
$$

$$
\overline{poscar} = \frac{1}{M}\sum_i poscar_i
$$

$$
poscar\_cnt = \sum_i poscar_i
$$

$$
\overline{p} = \frac{1}{M}\sum_i \bar{p}_i
$$

其中 $\overline{p}$ 输出列名为 `e_est_pos`。

### 9.2 t 检验

源码用 `scipy.stats.ttest_1samp(series, 0).statistic`。

对 CAR：

$$
t_{CAR} = \frac{\overline{CAR}}{s(CAR)/\sqrt{M}}
$$

对 SCAR：

$$
t_{SCAR} = \frac{\overline{SCAR}}{s(SCAR)/\sqrt{M}}
$$

输出列：

| 列名 | 含义 |
| --- | --- |
| `car_t` | $t_{CAR}$ |
| `scar_t` | $t_{SCAR}$ |

### 9.3 Sign Test

源码计算：

$$
tsign = \frac{\overline{poscar} - 0.5}{\sqrt{0.25/M}}
$$

输出列名为 `tsign`。

### 9.4 Patell Test

源码计算：

$$
tpatell = \overline{SCAR}\sqrt{M}
$$

输出列名为 `tpatell`。

### 9.5 Generalized Sign Test

源码计算：

$$
gen\_z =
\frac{poscar\_cnt - M\overline{p}}
{\sqrt{M\overline{p}(1-\overline{p})}}
$$

输出列名为 `gen_z`。

### 9.6 全样本输出格式

`getFullSampleTestStatistic(modelType)` 输出一行 DataFrame：

| 列名 | 含义 |
| --- | --- |
| `model` | 模型名称 |
| `car_mean` | CAR 均值 |
| `scar_mean` | SCAR 均值 |
| `poscar_mean` | 正 CAR 比例 |
| `poscar_cnt` | 正 CAR 个数 |
| `evt_count` | 事件样本数 |
| `e_est_pos` | 估计期正残差比例均值 |
| `car_t` | CAR 单样本 t 统计量 |
| `scar_t` | SCAR 单样本 t 统计量 |
| `tsign` | Sign Test 统计量 |
| `tpatell` | Patell Test 统计量 |
| `gen_z` | Generalized Sign Test 统计量 |

## 10. 分组统计量

`getGroupLevelTestStatistics(modelType, GRP)` 按指定分组列分别计算统计量。

对每个分组 $g$，设该组事件数量为 $M_g$。

分组内计算：

$$
\overline{CAR}_g,\ median(CAR)_g,\ \overline{SCAR}_g,\ median(SCAR)_g
$$

$$
\overline{poscar}_g,\ poscar\_cnt_g,\ \overline{p}_g
$$

t 检验：

$$
t_{CAR,g} = \frac{\overline{CAR}_g}{s(CAR_g)/\sqrt{M_g}}
$$

$$
t_{SCAR,g} = \frac{\overline{SCAR}_g}{s(SCAR_g)/\sqrt{M_g}}
$$

Sign Test：

$$
tsign_g = \frac{\overline{poscar}_g - 0.5}{\sqrt{0.25/M_g}}
$$

Patell Test：

$$
tpatell_g = \overline{SCAR}_g\sqrt{M_g}
$$

Generalized Sign Test：

$$
gen\_z_g =
\frac{poscar\_cnt_g - M_g\overline{p}_g}
{\sqrt{M_g\overline{p}_g(1-\overline{p}_g)}}
$$

输出列：

| 列名 | 含义 |
| --- | --- |
| `GRP` 对应列 | 分组值，例如 `CCAR` |
| `car_mean` | 分组 CAR 均值 |
| `car_median` | 分组 CAR 中位数 |
| `scar_mean` | 分组 SCAR 均值 |
| `scar_median` | 分组 SCAR 中位数 |
| `poscar_mean` | 分组正 CAR 比例 |
| `poscar_cnt` | 分组正 CAR 个数 |
| `evt_count` | 分组事件样本数 |
| `e_est_pos` | 分组估计期正残差比例均值 |
| `car_t` | 分组 CAR t 统计量 |
| `scar_t` | 分组 SCAR t 统计量 |
| `model` | 模型名称 |
| `tsign` | 分组 Sign Test 统计量 |
| `tpatell` | 分组 Patell Test 统计量 |
| `gen_z` | 分组 Generalized Sign Test 统计量 |

## 11. Generalized Rank Z Test

`getGRANK(modelType, GRP)` 实现 Generalized Rank Z Test 的近似流程。

### 11.1 构造用于排序的序列

估计期使用：

$$
GSAR_{i,t} = SAR_{i,t} = \frac{e_{i,t}}{\sigma_i}
$$

事件期使用：

$$
GSAR_{i,event} = scar\_bmp_i
$$

事件期观测被赋予哨兵日期：

```text
9999-12-31
```

### 11.2 排名和 U 值

对每个证券 $i$，将估计期 $GSAR_{i,t}$ 和事件期 $GSAR_{i,event}$ 合并排序。

令：

| 符号 | 含义 |
| --- | --- |
| $r_{i,t}$ | 排名，从 1 开始 |
| $T_i$ | 该证券参与排名的总观测数 |

源码计算：

$$
U_{i,t} = \frac{r_{i,t}}{T_i + 1} - 0.5
$$

标准差因子：

$$
sd\_factor_i = \frac{T_i - 1}{T_i + 1}
$$

### 11.3 分组 GRANK Z

对分组 $g$：

$$
\bar{U}_g = \frac{1}{M_g}\sum_{i \in g} U_{i,event}
$$

源码中的标准差为：

$$
U\_sd_g =
\sqrt{
\frac{1}{12M_g^2}
\sum_{i \in g}\frac{T_i - 1}{T_i + 1}
}
$$

最终：

$$
GRANK\_Z_g = \frac{\bar{U}_g}{U\_sd_g}
$$

输出列：

| 列名 | 含义 |
| --- | --- |
| `model` | 模型名称 |
| `GRP` 对应列 | 分组值 |
| `GRANK_Z` | Generalized Rank Z 统计量 |

重要限制：当前源码虽然函数参数叫 `GRP`，但内部多处硬编码使用 `CCAR` 分组。因此 `getGRANK(modelType, "CCAR")` 是最符合当前实现的用法，传入其他分组列可能报错或得到不符合预期的结果。

## 12. Wilcoxon Rank-Sum Test

`getWilcoxon(modelType, GRP)` 对 `getCARS(modelType)` 输出中的 `scar` 做 rank-sum 检验。

### 12.1 排名

先过滤 `scar` 的缺失值和无穷值，再对全部样本按 `scar` 排名：

$$
rank_i = rank(SCAR_i)
$$

ties 使用 pandas 的 `average` 方法处理。

### 12.2 分组秩和

对每个分组计算：

$$
R_g = \sum_{i \in g} rank_i
$$

$$
N_g = \#\{i \in g\}
$$

源码会按 `N_g` 从小到大排序，然后使用前两个分组：

| 符号 | 含义 |
| --- | --- |
| $N_1$ | 样本数较小组的数量 |
| $N_2$ | 第二组数量 |
| $R_1$ | 样本数较小组的秩和 |

### 12.3 Z 统计量

源码计算：

$$
Z =
\frac{
R_1 - \frac{N_1(N_1+N_2+1)}{2} + 0.5
}{
\sqrt{\frac{N_1N_2(N_1+N_2+1)}{12}}
}
$$

单尾 p 值：

$$
p_{1tail} = \Phi(Z)
$$

双尾 p 值：

$$
p_{2tail} = 2 \cdot sf(|Z|)
$$

其中 $\Phi$ 和 `sf` 来自 `scipy.stats.norm`。

`getWilcoxon()` 最终只返回：

| 列名 | 含义 |
| --- | --- |
| `model` | 模型名称 |
| `wilcox` | Z 统计量，保留两位小数 |
| `wilcox_p` | 双尾 p 值，保留八位小数 |

重要限制：

1. 该实现更接近两组 rank-sum 检验。若 `GRP` 有超过两个分组，源码只会使用按样本数排序后的前两个分组。
2. 返回结果不是每个分组一行，而是整个比较一行。

## 13. 主要方法输出汇总

### 13.1 `runModel(modelType)`

返回每个 `unique_id` 的 OLS 参数估计。标准模型的参数列通常包括 `Intercept` 和对应因子系数。

### 13.2 `fitModel(modelType)`

返回每个 `unique_id` 的 statsmodels OLS fit object。后续预测、残差和统计量都依赖该方法。

### 13.3 `getPredictedReturns(modelType)`

返回事件期面板，并新增：

| 列名 | 含义 |
| --- | --- |
| `predicted_return` | 事件期正常收益预测值 |

### 13.4 `getAbnormalReturns(modelType)`

在 `getPredictedReturns()` 的基础上新增：

| 列名 | 含义 |
| --- | --- |
| `abret` | 事件期异常收益 |

### 13.5 `getModelErrors(modelType)`

返回二元组：

```python
sigmas, model_errors = study.getModelErrors(modelType)
```

`model_errors` 主要列：

| 列名 | 含义 |
| --- | --- |
| `unique_id` 对应列 | 证券 ID |
| `date` | 估计期交易日 |
| `n` | 估计期残差数量 |
| `k` | OLS 参数数量 |
| `e` | 估计期残差 |
| `model` | 模型名称 |
| `e_sign` | 残差是否大于 0 |
| `sigma` | 证券层面的残差 RMSE |
| `sar` | 估计期标准化残差 |
| 分组列 | 例如 `CCAR`, `GSIB` |

`sigmas` 主要列：

| 列名 | 含义 |
| --- | --- |
| `unique_id` 对应列 | 证券 ID |
| `sigma` | 残差 RMSE |
| 分组列 | 例如 `CCAR`, `GSIB` |
| `sar` | 标准化残差 |
| `e_est_pos` | 正残差比例相关中间值 |

在 `getCARS()` 中，`e_est_pos` 会进一步按证券取均值。

### 13.6 `getCARS(modelType)`

返回事件级 CAR 数据：

| 列名 | 含义 |
| --- | --- |
| `unique_id` 对应列 | 证券 ID |
| `EVT_DATE` | 事件日 |
| `car` | 累计异常收益 |
| `model` | 模型名称 |
| `sigma` | 估计期残差 RMSE |
| 分组列 | 例如 `CCAR`, `GSIB` |
| `e_est_pos` | 估计期正残差比例 |
| `scar` | 标准化累计异常收益 |
| `scar_bmp` | BMP 风格标准化变量 |
| `poscar` | CAR 是否大于 0 |

### 13.7 `getFullSampleTestStatistic(modelType)`

返回全样本统计量，一行。

### 13.8 `getGroupLevelTestStatistics(modelType, GRP)`

返回指定分组列的分组统计量，每个分组值一行。

### 13.9 `getGRANK(modelType, GRP)`

返回每个分组的 `GRANK_Z`。

### 13.10 `getWilcoxon(modelType, GRP)`

返回 rank-sum 检验的整体 Z 值和双尾 p 值。

## 14. 推荐使用流程

```python
from event_study_toolkit import eventstudy as es

data = es.open_example_data()
events = es.open_example_events()

study = es.eventstudy(
    estperiod=50,
    gap=5,
    start=-1,
    end=1,
    data=data,
    events=events,
    unique_id="subject_id",
    calType="NYSE",
    groups=["CCAR", "GSIB"],
)

params = study.runModel("market")
pred = study.getPredictedReturns("market")
abret = study.getAbnormalReturns("market")
cars = study.getCARS("market")
full_stats = study.getFullSampleTestStatistic("market")
group_stats = study.getGroupLevelTestStatistics("market", "CCAR")
grank = study.getGRANK("market", "CCAR")
wilcox = study.getWilcoxon("market", "CCAR")
```

## 15. 实现注意事项

1. `groups` 建议必填。若不传，部分方法访问 `self.groups` 时会失败。
2. `data["date"]` 建议在传入前转换为 pandas datetime 类型。代码只主动转换了 `events["EVT_DATE"]`。
3. `EVT_DATE` 必须是交易日，代码不会自动做最近交易日匹配。
4. 当前交易日历固定到 `2023-07-10`。
5. 标准模型使用固定列名。如果列名不同，应改列名或使用自定义公式。
6. `getGRANK()` 内部硬编码 `CCAR`，非 `CCAR` 分组需要先修改源码。
7. `getWilcoxon()` 只真正使用两个分组。多于两个分组时，其他分组不会进入 Z 统计量。
8. 当前仓库测试文件为空，使用前建议用自己的数据做独立校验。

