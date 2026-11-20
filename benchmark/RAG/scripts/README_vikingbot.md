# VikingBot 单题 / 全量轨迹测试

入口脚本位于 `src/run_vikingbot.sh`，可以从任意目录执行。
默认使用仓库 `.venv/bin/python`，可用 `VIKINGBOT_PYTHON` 指定解释器。
下面的命令从仓库根目录执行。

## 本地 SyllabusQA

当前环境默认使用 `.local-data/syllabusqa/ov.conf` 和
`.local-data/syllabusqa/syllabusqa-test-import.json`，连接已经运行的服务。
如果没有这两个文件，需要提供 `--ov-conf` 和 `--source-map`。
脚本不会启动服务或导入数据。

单个题目 ID，保存轨迹：

```bash
benchmark/RAG/src/run_vikingbot.sh \
  --dataset syllabusqa \
  --id G4zZKP3yeU0rVsqy \
  --ark-key-file .local-data/syllabusqa/ark-api-key
```

全量 test 集，4 个独立进程：

```bash
benchmark/RAG/src/run_vikingbot.sh \
  --dataset syllabusqa --all --workers 4 \
  --ark-key-file .local-data/syllabusqa/ark-api-key
```

`ARK_API_KEY` 已设置时不需要 `--ark-key-file`。还可指定：

- `--data benchmark/RAG/datasets/SyllabusQA/val.csv`：替换数据文件；
- `--source-doc-id "BIOCHEM 320 Syllabus SP23 2 Feb 2023"`：只选某个文档的题；
- `--resource-uri viking://resources/syllabusqa-smoke-bioch-320`：明确指定该文档的资源；
- `--preview-chars 300 --max-iterations 12 --timeout 240`：预览长度、Agent 迭代上限和每题超时；
- `--limit 4`：只跑选中题目中的前 4 道，可搭配 `--all --workers 4`；
- `--no-judge`：跳过默认启用的 LLM judge；
- `--judge-model MODEL --judge-base-url URL --judge-timeout 60`：指定 judge 模型、兼容 OpenAI 的接口与请求超时；默认复用 `ov.conf` 中的 VLM 模型、接口和 API key；
- `--output-dir /path/to/output`：输出父目录；
- `--dry-run`：只保存选择计划与完整 chunk 快照，不调用模型；
- `--list-cases`：查看原始题目 ID 和 canonical ID，不访问服务或模型。

## 其他数据集

MuSiQue 的完整共享语料准备、导入、单题及 120 题测试见
[MuSiQue 接入说明](README_musique.md)。

支持 `financebench`、`qasper`、`locomo`；默认文件分别为
`financebench_open_source.jsonl`、`qasper-test-v0.3.json`、`locomo10.json`。
用 `--data` 可换成别的 split。ID 对应 adapter 的 `financebench_id`、
`question_id`、`original_id`。没有原始 ID 时使用 `source_doc_id:qa_INDEX`，
INDEX 是 adapter 中从 0 开始的题序号；原始 ID 重复时需额外指定文档 ID。

```bash
benchmark/RAG/src/run_vikingbot.sh --dataset locomo --list-cases
benchmark/RAG/src/run_vikingbot.sh \
  --dataset locomo --all --workers 4 \
  --ov-conf /path/to/ov.conf --source-map /path/to/source-map.json
```

资源映射 JSON 使用 adapter 的文档 ID 作为键，值为已经导入的资源根 URI：

```json
{
  "document-id-from-adapter": "viking://resources/loaded-document"
}
```

也支持现有 SyllabusQA 导入报告中的 `resources` 数组。
文档 ID 与 URI 必须准确对应；缺失映射会在调用模型前报错。
所有选中文档的完整 L2 `.md` chunk 会在 Agent 运行前读取并保存；
以 `.` 开头的摘要文件不计入 L2 快照。

## 输出

默认每次新建 `Output/vikingbot/<dataset>/<时间戳_随机后缀>/`：

```text
plan.json                    # 选择的题目、资源、运行参数
index.json                   # 题目 ID、完整问题文本、目录和结果路径
index.tsv                    # 每题一行，支持 grep 按 ID / query 查找路径
summary.json                 # 逐题状态、指标、judge 平均分、结果路径
snapshots/*.json             # 每个文档一份完整 chunk 快照
cases/G4zZKP3yeU0rVsqy__how-can-i-get-extra-credit__<hash>/
  result.json                # 回答、gold、工具事件、指标、轨迹路径、llm_evaluation
  benchmark.log              # 运行日志
  steps.readable.log         # 每步/每 chunk 的简明命中日志
  <uuid>.manifest.json       # gold 与快照的匹配清单
  <uuid>.steps.jsonl         # 完整工具证据轨迹
  <uuid>.steps.readable.json # 展开的步骤、chunk 预览与命中标记
```

样本目录名包含 `case_id`、问题文本预览和基于题目身份的短哈希，命名与执行顺序
及 worker 完成顺序无关。预览会过滤路径字符、限制字节长度；
同一原始 ID 在不同文档中出现时，哈希后缀可区分目录。
SyllabusQA 目录预览会省略 adapter 添加的课程名称前缀；
完整问题仍保存在索引、`plan.json`、`summary.json` 和每题的 `result.json` 中。
索引在 worker 启动前生成，因此也可以查看尚未完成的题目对应的预定路径。

在仓库根目录用 macOS 自带的 `grep` 按题目 ID 或 query 关键词查找：

```bash
grep -F 'G4zZKP3yeU0rVsqy' benchmark/RAG/Output/vikingbot/syllabusqa/*/index.tsv
grep -iF 'extra credit' benchmark/RAG/Output/vikingbot/syllabusqa/*/index.tsv
```

每个匹配行会同时显示题目 ID、问题和结果路径。
新命名用于之后的运行，已有输出目录保持原样。
如果旧运行没有 `index.tsv`，可直接搜索已有的结果文件：

```bash
find benchmark/RAG/Output -name result.json -exec grep -ilF 'extra credit' {} +
```

`completed` 表示 Agent 执行完成并生成了工具证据轨迹，不代表答案正确；
gold 匹配结果在 `evidence_summary` 中。没有工具轨迹或模型调用失败时标为
`failed`，保存已有结果/部分轨迹；有失败题时脚本返回非零退出码。
`--all` 执行 adapter 返回的全部题目，遵循 adapter 本身的过滤规则
（例如 SyllabusQA 会过滤 `no answer` 题）。

每题完成后默认执行 LLM judge，沿用 `core/judge_util.py` 的数据集评分规则：
SyllabusQA、FinanceBench、Qasper 为 0–4 分，Locomo 为 0 或 4 分。
`result.json.llm_evaluation` 保存 `status`、`score`、`normalized_score`
（分数除以 4，范围 0–1）、`reasoning`、`prompt_type`、模型、耗时、原始输出
和 judge token 用量。Agent token 用量与 judge 用量分别记录。
`summary.json.llm_judge` 汇总评分完成/失败/跳过数量及平均分。
judge 调用或 JSON 解析失败时 `score` 为 `null`，不纳入平均分；
Agent 回答和轨迹仍保留，运行总状态为 `failed`，退出码为非零。
Agent 失败、没有 gold 或指定 `--no-judge` 时跳过评分。
`--dry-run` 和 `--list-cases` 不调用 judge。启用 judge 会额外产生模型调用费用。

各进程共享服务和只读资源，每道题使用独立 workspace、session 和用户身份。
服务容量及模型限流会影响适合的 worker 数；可先用 2–4 个。
原始 `ov.conf` 不改动；临时运行配置移除当前分支不支持的 `bot.rag_checkpoints`。
评估的 gold 内容保留在 benchmark 侧，Agent 输入仅包含问题、资源范围和工具约束。
