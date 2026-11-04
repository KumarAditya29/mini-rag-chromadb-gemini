"""Summarize Promptfoo results into a short, readable report.

Usage: python eval/summarize.py [eval/results/results.json]
Prints the summary and writes eval/results/summary.md next to the JSON.
"""

import json
import sys
from pathlib import Path
from statistics import mean

path = Path(sys.argv[1] if len(sys.argv) > 1 else "eval/results/results.json")
data = json.loads(path.read_text())
rows = data["results"]["results"]

METRICS = ["Faithfulness", "AnswerRelevance", "KeyFact", "Refusal"]


def components(row):
    return (row.get("gradingResult") or {}).get("componentResults") or []


def metric_scores(metric):
    return [
        c.get("score", 0)
        for r in rows
        for c in components(r)
        if (c.get("assertion") or {}).get("metric") == metric
    ]


def metric_passes(metric):
    results = [
        c.get("pass", False)
        for r in rows
        for c in components(r)
        if (c.get("assertion") or {}).get("metric") == metric
    ]
    return sum(results), len(results)


def status(row):
    if row.get("success"):
        return "PASS"
    # failureReason 2 = provider/API error (no answer was graded)
    return "ERROR" if row.get("failureReason") == 2 else "FAIL"


def name(row):
    return (row.get("testCase") or {}).get("description") or row["vars"]["query"]


lines = ["# Mini RAG Evaluation Summary", ""]
counts = {k: sum(1 for r in rows if status(r) == k) for k in ("PASS", "FAIL", "ERROR")}
gen_models = sorted({
    str(((r.get("response") or {}).get("metadata") or {}).get("rag_model"))
    for r in rows
    if ((r.get("response") or {}).get("metadata") or {}).get("rag_model")
})
judge = (((data.get("config") or {}).get("defaultTest") or {}).get("options") or {}).get("provider") or {}
lines += [
    f"- Generation model: **{', '.join(gen_models) or 'unknown'}**",
    f"- Judge model: **{(judge.get('text') or {}).get('id', 'unknown')}**, embeddings: **{(judge.get('embedding') or {}).get('id', 'unknown')}**",
    f"- Total test cases: **{len(rows)}**",
    f"- Passed: **{counts['PASS']}**, failed: **{counts['FAIL']}**, errors (not graded): **{counts['ERROR']}**",
    "",
]

lines += ["## Metrics", "", "| Metric | Avg score | Assertions passed |", "|---|---|---|"]
for m in METRICS:
    scores = metric_scores(m)
    if scores:
        ok, total = metric_passes(m)
        lines.append(f"| {m} | {mean(scores):.2f} | {ok}/{total} |")
lines.append("")

lines += ["## Refusal tests", ""]
for r in rows:
    if r["vars"].get("category") == "unsupported":
        answer = str((r.get("response") or {}).get("output", "")).strip().replace("\n", " ")
        lines.append(f"- **{status(r)}** {name(r)}  \n  Answer: _{answer[:200]}_")
lines.append("")

lines += ["## All test cases", "", "| Result | Test | Faith. | Relev. | KeyFact | Refusal |", "|---|---|---|---|---|---|"]
for r in rows:
    named = r.get("namedScores") or {}
    cells = [f"{named[m]:.2f}" if m in named else "–" for m in METRICS]
    lines.append(f"| {status(r)} | {name(r)} | " + " | ".join(cells) + " |")
lines.append("")

failed = [r for r in rows if not r.get("success")]
lines += ["## Failed / errored test cases", ""]
if not failed:
    lines.append("None.")
for r in failed:
    lines.append(f"### {name(r)}")
    if r.get("error"):
        lines.append(f"- Error: {r['error']}")
    for c in components(r):
        if not c.get("pass"):
            a = c.get("assertion") or {}
            lines.append(f"- `{a.get('type')}` ({a.get('metric')}): {c.get('reason')}")
    answer = str((r.get("response") or {}).get("output", "")).strip()
    lines.append(f"- Answer: {answer[:400]}")
    lines.append("")

report = "\n".join(lines)
print(report)
(path.parent / "summary.md").write_text(report + "\n")
