# 已导入文件的 grep 漏检对照扫描

从 OV 读取完整正文，提取每份文件中 7–24 个字母的英文单词；分别搜索整词、
删除末尾一个字符和两个字符的前缀。对每个查询，将原文中实际包含该模式的行
与 OV `/api/v1/search/grep` 返回的行对比，同时测试单文件和资源根目录范围。
使用当前服务的检索配置，不导入数据、不调用 LLM、不修改服务配置。

在仓库根目录执行：

```bash
.venv/bin/python benchmark/RAG/scripts/scan_grep_recall.py \
  --cli-config .local-data/syllabusqa/ovcli.conf \
  --uri viking://resources/syllabusqa-test \
  --uri viking://resources/syllabusqa-smoke-bioch-320 \
  --uri viking://resources/syllabusqa-chem111-fa22 \
  --max-words 100000 --workers 8
```

`--max-words` 是单词选择上限，100000 高于当前 syllabus 中符合条件的单词数量，
因此本次涵盖全部符合条件的单词。可用 `--max-words 300` 快速抽样，
`--seed 1` 更换抽样顺序。`--uri` 支持重复指定，也可以指定单个已导入 `.md` 文件。
`--scope file` / `root` / `both` 控制查询范围，默认为 both；
`--node-limit` 默认为 256；`--output-dir` 控制输出父目录。
读取目录时逐层遍历，避免 recursive listing 的默认深度遗漏正文。

每次运行生成 `benchmark/RAG/Output/grep-scan/<时间戳>/`：

- `report.md`：结果摘要与异常案例的 CLI 复现命令；
- `summary.json`：服务版本、范围、查询数量和状态计数；
- `results.jsonl`：全部查询、预期行、实际返回和复现命令；
- `failures.json`：漏检、截断、报错或原文变化案例；
- `snapshot.json`：测试使用的完整正文快照。

`miss` 表示原文存在但返回缺少预期行，且服务未报告截断；
再次执行仍漏检会记录 `stable_miss: true`。发现 miss 时会重读相关行，
原文变化单独记为 `source_changed`。报告候选或结果截断时记为 `capped`，
接口失败时记为 `error`，不将这两类当成已证明漏检。
HTTP 连接按线程复用，避免大规模扫描耗尽本机临时端口。

本次本地验证：183 个正文文件、39961 个查询全部通过，没有漏检、截断或接口错误。
覆盖字面量单词/前缀，区分大小写；未覆盖任意正则、MCP 多 pattern 的包装行为
或不同服务环境。当前 grep 默认 `engine=auto`，远程索引切换阈值为 10000；
大系统可能走不同路径。可把同一脚本的 `--cli-config` 指向旧系统，在那里的已导入
URI 上执行对照；实际检索引擎仍应从该服务的配置和日志确认。
发现 miss 或 error 时退出码为 1，否则为 0；有 capped 时应查看报告再决定缩小范围或增大上限。
