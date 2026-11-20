# MuSiQue 共享语料导入和 VikingBot 测试

本地两份 JSONL 已检查：语料池有 **26,326** 个唯一 chunk，题目文件有
**120** 道可回答题；支持 chunk 数为 2/3/4 的题目分别有 61/40/19 道，
共引用 279 个不同 chunk，引用全部存在。

`MuSiQueAdapter` 返回一个共享 corpus sample，包含所有题目；
每题保留原始 qid，gold answers 包含去重后的答案与别名。
每个原始 chunk 单独转换为 Markdown，正文包含标题和 text。
导入整个语料池，包括干扰项。不能只导入 279 个 gold 支持 chunk，
否则检索难度与结果会受到 gold 信息影响。

以下命令均从 OpenViking 仓库根目录执行。

## 1. 准备语料

```bash
.venv/bin/python benchmark/RAG/scripts/prepare_musique.py \
  --questions /Users/bytedance/Developer/data/musique/musique_full_v1.0_dev_answerable_subset_120_seed42.jsonl \
  --chunks /Users/bytedance/Developer/data/musique/musique_full_v1.0_dev_chunks.jsonl \
  --output-dir .local-data/musique/dev \
  --resource-uri viking://resources/musique-full-v1-dev
```

此步骤不访问服务或模型，生成 `chunks/*.md`、`manifest.json`、`stats.json`。
文件名为原始 ID 的稳定短哈希，适配 Markdown 解析器的文件名长度限制。
manifest 保存原始 chunk ID、标题和文件名；问题、答案、支持关系不写入语料。

## 2. 导入已运行的 OpenViking 服务

```bash
.venv/bin/python benchmark/RAG/scripts/prepare_musique.py \
  --questions /Users/bytedance/Developer/data/musique/musique_full_v1.0_dev_answerable_subset_120_seed42.jsonl \
  --chunks /Users/bytedance/Developer/data/musique/musique_full_v1.0_dev_chunks.jsonl \
  --output-dir .local-data/musique/dev \
  --resource-uri viking://resources/musique-full-v1-dev \
  --ov-conf .local-data/syllabusqa/ov.conf --ingest
```

脚本通过 SDK 上传 `chunks/` 目录，使用 `no_split`、`preserve_structure=True`，
保留输入 chunk 粒度，并等待索引完成。默认 `vectors_only` 建立向量索引；
需要语义摘要时可加 `--processing-mode semantic_and_vectors`。
服务端 embedding 凭据必须可用；语义摘要模式还需要服务端 VLM 凭据。
客户端的 `--ark-key-file` 用于后续 worker，不能修复已经运行的服务端环境变量。
26,326 个文件的上传、解析与 embedding 会产生耗时和模型费用。

导入后核对实际递归 listing，缺失、重复、拆分或额外文件均报错。
只有核对成功才写出 `source-map.json`：
共享 corpus ID 指向资源根，每个原始 chunk ID 指向实际 L2 URI。
`import-report.json` 保存导入和验证状态。该验证核对文件身份与数量，
不逐文件比较远端正文，也不独立证明 embedding 索引质量。

已有目标会拒绝再次导入。若先前导入成功但映射验证中断，
用同一命令将 `--ingest` 换成 `--verify-only` 即可重建映射；
目标语料不同则使用新 URI 和新 output-dir。

## 3. 查看题目及单题测试

```bash
benchmark/RAG/src/run_vikingbot.sh \
  --dataset musique \
  --data /Users/bytedance/Developer/data/musique/musique_full_v1.0_dev_answerable_subset_120_seed42.jsonl \
  --chunks /Users/bytedance/Developer/data/musique/musique_full_v1.0_dev_chunks.jsonl \
  --list-cases
```

```bash
benchmark/RAG/src/run_vikingbot.sh \
  --dataset musique \
  --data /Users/bytedance/Developer/data/musique/musique_full_v1.0_dev_answerable_subset_120_seed42.jsonl \
  --chunks /Users/bytedance/Developer/data/musique/musique_full_v1.0_dev_chunks.jsonl \
  --source-map .local-data/musique/dev/source-map.json \
  --ov-conf .local-data/syllabusqa/ov.conf \
  --ark-key-file .local-data/syllabusqa/ark-api-key \
  --id 2hop__31585_61048 --max-iterations 12 --timeout 240
```

将最后一行换为 `--all --limit 4 --workers 4 --max-iterations 12 --timeout 240`
先测 4 道；去掉 `--limit 4` 后测试全部 120 道。
MuSiQue 必须提供完整 chunk 映射，不能仅用 `--resource-uri`。
支持证据只传给 benchmark 匹配器，Agent 输入仅有问题和共享 corpus 范围。

每次运行先对全语料读取完整 L2 快照：约 26,326 次正文请求，当前顺序读取；
即使单题也需要完整语料快照。一次 batch 共用同一份快照，
每个 worker 为当前题加载该快照。`--dry-run` 也会读取快照，但不调用 Agent/judge。

输出在 `benchmark/RAG/Output/vikingbot/musique/<run>/`，qid/query 查找、
独立 worker、日志和失败状态沿用 [通用说明](README_vikingbot.md)。
`evidence_summary` 按原始支持 chunk ID 计算 located/returned/delivered recall；
gold 文本为整个支持段落，部分正文预览不等于完整支持段落命中。
答案默认使用现有通用 LLM judge 的 0–4 分与别名匹配规则；
这些分数不是 MuSiQue 官方 answer EM/F1 或官方 support F1。

此接入面向 `run_vikingbot.sh` 工具轨迹评测；
旧 `run.py` 的按单文档组织的 pipeline 尚未适配共享 corpus。
