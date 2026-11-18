"""Export a completed SyllabusQA VikingBot run as searchable, copyable transcripts."""

import argparse
import json
from collections import defaultdict
from itertools import groupby
from pathlib import Path


def rows(path):
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source]


def transcript(path, selected_answer=None):
    events = rows(path)
    request = events[0]
    if request["event"] != "request":
        raise ValueError(f"Missing request in {path}")
    response = next((e["data"]["value"] for e in events if e["event"] == "response"), None)
    if selected_answer is not None and (not response or response["content"].strip() != selected_answer):
        raise ValueError(f"Answer mismatch in {path}")
    parts = [f"原始文件  {path.name}", "请求\n" + json.dumps(request["data"], ensure_ascii=False, indent=2)]
    for kind, group in groupby(events[1:], key=lambda event: event["event"]):
        batch = list(group)
        if kind == "content_delta":
            content = "".join(e["data"]["value"] for e in batch)
            if response and content == response.get("content") and batch[-1] is events[-2]:
                continue  # The final response below contains the same text verbatim.
            parts.append(f"[+{batch[0]['elapsed_seconds']}s] 输出片段（已合并）\n{content}")
            continue
        for event in batch:
            value = event["data"]
            if isinstance(value, dict) and "value" in value:
                value = value["value"]
            if kind == "response" and "content" in value:
                label = f"最终回答  {value.get('response_id', '')}"
                value = value["content"]
            else:
                label = {"iteration": "轮次", "tool_call": "工具调用", "tool_result": "工具返回",
                         "response": "服务响应", "error": "错误", "request_failed": "请求失败"}.get(kind, kind)
            if not isinstance(value, str):
                value = json.dumps(value, ensure_ascii=False, indent=2)
            parts.append(f"[+{event['elapsed_seconds']}s] {label}\n{value}")
    if events[-1]["event"] not in {"response", "request_failed", "error", "interrupt"}:
        parts.append("原始记录在此结束，未记录最终回答。")
    return "\n\n".join(parts)


PAGE = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>SyllabusQA 详细轨迹</title>
<style>
*{box-sizing:border-box}body{max-width:1080px;margin:auto;padding:24px;color:#1d2926;background:#fafbf9;
font:16px/1.5 "Source Sans 3",system-ui,sans-serif}h1{font-size:1.5rem}label{display:block;margin:10px 0 4px}
input,select,button{font:inherit;padding:7px;border:1px solid #849890;border-radius:5px;background:white}
input,select{width:100%}button{cursor:pointer;margin:14px 10px 0 0}pre{background:white;border-top:2px solid #38735c;
padding:16px;white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.6 "IBM Plex Mono",monospace}
</style><main><h1>SyllabusQA 详细轨迹</h1>
<label for="query">搜索样本 ID、课程或题目</label><input type="search" id="query">
<label for="samples" id="count"></label><select id="samples"></select>
<button id="copy" type="button">复制整题轨迹</button><span id="status" role="status"></span><pre id="trace"></pre></main>
<script type="application/json" id="data">__DATA__</script><script>
const records = JSON.parse(document.getElementById('data').textContent);
const byId = new Map(records.map(r => [r.id,r]));
const get = id => document.getElementById(id);
const query = get('query'), samples = get('samples'), trace = get('trace');
function show() {
  const record = byId.get(samples.value);
  trace.textContent = record?.text || '';
  get('copy').disabled = !record;
  get('status').textContent = '';
}
function filter(preferred) {
  const term = query.value.trim().toLowerCase();
  const hits = records.filter(r => [r.id,r.syllabus,r.question].some(v => v.toLowerCase().includes(term)));
  samples.replaceChildren(...hits.map(r => new Option(r.id+' · '+r.syllabus+' · '+r.question,r.id)));
  const match = hits.find(r => r.id.toLowerCase() === term) || hits.find(r => r.id === preferred);
  if (match) samples.value = match.id;
  get('count').textContent = '匹配样本 '+hits.length+' / '+records.length;
  show();
}
query.oninput = () => filter(samples.value);
samples.onchange = show;
get('copy').onclick = async () => {
  try {
    await navigator.clipboard.writeText(trace.textContent);
    get('status').textContent = '已复制';
  } catch {
    const range = document.createRange(); range.selectNodeContents(trace);
    const selection = window.getSelection(); selection.removeAllRanges(); selection.addRange(range);
    get('status').textContent = document.execCommand('copy') ? '已复制' : '已选中整题，请按 ⌘C 或 Ctrl+C';
  }
};
filter(location.hash.slice(1));
</script></html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    root = parser.parse_args().run
    results = rows(root / "results.jsonl")
    brief = {r["id"]: r for r in rows(root / "traces_brief.jsonl")}
    attempts = defaultdict(list)
    for path in sorted((root / "traces_detailed").glob("*.jsonl")):
        attempts[path.name.split(".", 1)[0]].append(path)
    ids = {r["id"] for r in results}
    if len(ids) != len(results) or ids != set(brief) or ids != set(attempts):
        raise ValueError("IDs differ between results, brief traces and raw traces")

    records = []
    for r in results:
        key = r["id"]
        chosen = root / brief[key]["trace"]
        if chosen not in attempts[key]:
            raise ValueError(f"Missing selected attempt for {key}")
        judge = r["scores"]["judge"]
        sections = [f"样本 ID  {key}", f"课程  {r['syllabus']}\n题型  {r['type']}",
                    f"问题\n{r['question']}", f"参考答案\n{r['gold']}",
                    f"LLM 评分  {judge['score']}/4     F1  {r['scores']['f1']}\n{judge['reasoning']}",
                    "用于评分的轨迹\n" + transcript(chosen, r["answer"])]
        for path in attempts[key]:
            if path != chosen:
                sections.append("其他尝试（未用于评分）\n" + transcript(path))
        records.append({k: r[k] for k in ("id", "syllabus", "question")} | {"text": "\n\n".join(sections)})

    divider = "\n\n" + "=" * 80 + "\n\n"
    (root / "traces_detailed.txt").write_text(divider.join(r["text"] for r in records) + "\n", encoding="utf-8")
    data = json.dumps(records, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    (root / "traces_detailed.html").write_text(PAGE.replace("__DATA__", data), encoding="utf-8")
    print(f"Exported {len(records)} samples with {sum(map(len, attempts.values()))} attempts")


if __name__ == "__main__":
    main()
