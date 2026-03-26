# sentiment.py 使用说明

## 作用

`sentiment.py` 用于对 `data/raw_data/4` 下的 Form 4 文本文件执行批量情绪分析，并将结果保存为：

- `sentiment_results.jsonl`：逐条成功结果，便于断点续跑和增量追踪
- `failed_transcripts.jsonl`：失败批次及原因
- `sentiment_results.parquet`：面向分析的表格化结果
- `sentiment_pipeline.log`：运行日志

脚本的核心特性：

- 支持从磁盘流式读取文本，避免一次性加载全部文件
- 支持断点续跑，默认跳过已经处理过的 `id`
- 支持并发请求 Dify Workflow
- 每次运行结束后会基于历史结果重新生成 Parquet

## 输入数据约定

默认输入目录为：

```text
data/raw_data/4
```

脚本假定目录结构如下：

```text
data/raw_data/4/
  <SYMBOL>/
    <ACCESSION_OR_ID>/
      xxx.txt
```

例如：

```text
data/raw_data/4/AAPL/0000320193-24-000123/0000320193-24-000123.txt
```

其中：

- 股票代码目录和中间 `id` 目录仅用于遍历
- 实际写入结果的 `id` 来自 `.txt` 文件名去掉扩展名后的值

## Prompt 文件

默认 prompt 文件为：

```text
prompts/4_v2.txt
```

运行时会把当前批次的 transcript JSON 替换到 prompt 中的 `__TRANSCRIPTS_JSON__` 占位符里，再发送给 Dify。

如果 prompt 文件不存在或为空，脚本会尝试回退到内置模板；但当前代码里的默认模板变量为 `None`，因此实际使用时应保证 prompt 文件存在且内容有效。

## 输出目录约定

默认输出根目录为：

```text
data/results/4
```

如果不显式传入 `--output-dir`，最终输出目录为：

```text
data/results/4/<prompt_file_stem>
```

例如使用 `prompts/4_v2.txt` 时，输出目录默认是：

```text
data/results/4/4_v2
```

目录内会生成以下文件：

```text
sentiment_results.jsonl
failed_transcripts.jsonl
sentiment_results.parquet
sentiment_pipeline.log
```

## 运行前准备

### 1. 安装依赖

脚本依赖至少包括：

```bash
pip install aiohttp pandas pyarrow tqdm
```

说明：

- `pyarrow` 通常是写入 Parquet 所需依赖
- `tqdm` 可选，不安装也能运行，只是没有进度条

### 2. 检查 Dify 配置

脚本顶部写死了以下配置：

- `DIFY_API_URL`
- `DIFY_API_KEY`

需要确保它们在当前环境可用，否则请求会失败。

## 默认运行方式

在仓库根目录执行：

```bash
python sentiment.py
```

默认行为：

- 从 `data/raw_data/4` 读取文本
- 使用 `prompts/4_v2.txt`
- 输出到 `data/results/4/4_v2`
- 只处理前 `50` 个待处理文件
- 最大并发数为 `8`
- 每个批次最多重试 `1` 次
- 默认跳过已经存在于历史结果中的 `id`

## 命令行参数

### `--input-dir`

指定输入目录。

```bash
python sentiment.py --input-dir data/raw_data/4
```

### `--prompt-file`

指定 prompt 文件。

```bash
python sentiment.py --prompt-file prompts/4_extraction_v2.txt
```

### `--output-dir`

指定精确输出目录，而不是使用默认的 `data/results/4/<prompt_stem>`。

```bash
python sentiment.py --output-dir data/results/4/custom_run
```

### `--first-n-files`

只处理前 N 个待处理文件。

```bash
python sentiment.py --first-n-files 100
```

传入 `0` 或负数表示不限制。

```bash
python sentiment.py --first-n-files 0
```

### `--max-concurrency`

控制最大并发请求数。

```bash
python sentiment.py --max-concurrency 4
```

### `--max-retries`

控制每个批次的最大重试次数。

```bash
python sentiment.py --max-retries 3
```

### `--reprocess-existing`

重新处理已存在结果中的 `id`，不再跳过历史结果。

```bash
python sentiment.py --reprocess-existing
```

## 推荐运行示例

### 小规模测试

```bash
python sentiment.py --first-n-files 20 --max-concurrency 4
```

### 全量跑数

```bash
python sentiment.py --first-n-files 0 --max-concurrency 8 --max-retries 3
```

### 更换 prompt 并写入独立目录

```bash
python sentiment.py ^
  --prompt-file prompts/4_extraction_v2.txt ^
  --output-dir data/results/4/4_extraction_v2 ^
  --first-n-files 0
```

## 处理流程说明

脚本主流程如下：

1. 解析命令行参数
2. 根据 prompt 文件名确定输出目录
3. 读取历史 `jsonl/parquet`，收集已处理 `id`
4. 统计本轮待处理 transcript 数量
5. 流式遍历文本文件
6. 每 `5` 条 transcript 组成一个批次
7. 以有界并发方式请求 Dify
8. 成功结果实时追加到 `sentiment_results.jsonl`
9. 失败批次追加到 `failed_transcripts.jsonl`
10. 运行结束后重新生成 `sentiment_results.parquet`

## 结果格式

脚本期望模型返回一个 JSON 数组，数组中每个对象至少包含以下字段：

```json
[
  {
    "id": "sample_id",
    "sentiment_score": 0.4,
    "confidence": "84%",
    "summary": "Brief summary",
    "reasons": {
      "positives": ["..."],
      "negatives": ["..."],
      "guidance_tone": "...",
      "qa_tone": "..."
    }
  }
]
```

转换为 Parquet 后，主要字段为：

- `id`
- `sentiment_score`
- `confidence`
- `summary`
- `reasons_positives`
- `reasons_negatives`
- `reasons_guidance_tone`
- `reasons_qa_tone`

其中 `reasons_positives` 和 `reasons_negatives` 会以 JSON 字符串形式写入表格字段。

## 断点续跑逻辑

默认情况下，脚本会同时读取以下两个文件中的 `id`：

- `sentiment_results.jsonl`
- `sentiment_results.parquet`

只要某个 `id` 已经存在，就会被跳过。

这意味着：

- 中途中断后可直接重跑
- 新增数据时可继续补跑
- 如果要强制覆盖历史结果，需要显式传入 `--reprocess-existing`

另外，Parquet 生成时会把历史 Parquet 和 JSONL 的结果合并，并按 `id` 去重，保留最后一次出现的记录。

## 失败处理逻辑

单个批次失败后，脚本会记录：

- 失败的 `batch_ids`
- 失败原因
- 重试次数
- 时间戳

这些信息会追加到：

```text
failed_transcripts.jsonl
```

注意：失败记录是按批次写的，不是逐条 transcript 写的。

## 日志与进度

运行时日志同时输出到控制台和日志文件。日志文件默认位于：

```text
data/results/4/<prompt_stem>/sentiment_pipeline.log
```

如果安装了 `tqdm`，还会显示进度条。

## 已知注意事项

### 1. 默认只跑 50 个文件

当前代码中：

```python
TEST_FIRST_N_FILES = 50
```

因此直接执行 `python sentiment.py` 不是全量处理，而是测试模式。要跑全量请显式传入：

```bash
python sentiment.py --first-n-files 0
```

### 2. 重试次数默认只有 1

当前默认配置：

```python
MAX_RETRIES = 1
```

如果接口偶发超时或返回格式不稳定，建议在正式跑数时增大该值。

### 3. Dify 返回格式必须匹配当前解析逻辑

代码当前假定：

- HTTP 返回体是 JSON
- 可从 `data.outputs.text` 取到模型输出
- 输出文本中能正则提取出一个 JSON 数组

如果 Dify Workflow 的返回结构变化，需要同步修改解析代码。

### 4. Prompt 文件不能为空

虽然代码写了“缺失时回退到默认模板”，但默认模板目前是 `None`。实际运行时应始终保证 prompt 文件有效。

### 5. API Key 目前硬编码在脚本里

这适合临时开发环境，不适合长期维护或共享仓库。更稳妥的方式是改成环境变量或配置文件。

## 常见排查

### 没有新数据被处理

可能原因：

- 历史结果里已经有这些 `id`
- 输入目录结构不符合预期
- `.txt` 文件不在 `symbol/id/*.txt` 这一层级

可以尝试：

```bash
python sentiment.py --reprocess-existing --first-n-files 20
```

### 只看到 JSONL，没有 Parquet

通常是 Parquet 依赖缺失，先安装：

```bash
pip install pyarrow
```

### 批次频繁失败

优先检查：

- `DIFY_API_URL` 是否可访问
- `DIFY_API_KEY` 是否有效
- prompt 是否过长
- 接口返回是否还是 `data.outputs.text`

## 建议后续优化

如果这个脚本会长期使用，建议优先做以下改造：

1. 将 `DIFY_API_URL` 和 `DIFY_API_KEY` 改为环境变量
2. 把默认 prompt 模板补成真正可用的兜底模板
3. 将批大小 `5` 也做成命令行参数
4. 在失败记录中保留更多上下文，例如响应码和截断后的返回体
5. 增加一个单独的“仅从 JSONL 重建 Parquet”命令模式