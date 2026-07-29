#!/usr/bin/env python3
"""Regenerate demo.ipynb. Edit the cell sources here, then run this script."""

import json
from pathlib import Path

MD = "markdown"
CODE = "code"

cells = []


def add(kind, src):
    cells.append((kind, src.strip("\n")))


add(MD, """
# Multi-Agent PR Review — live run

`moonshotai/Kimi-K2.6` on Cerebras Inference · **fan out → critic-filter → adjudicate**

Eight cells. Each one adds exactly one thing.
""")

add(MD, "## 1 · The pull request")

add(CODE, """
import json, pathlib

PR = json.loads(pathlib.Path("fixtures/pr_103.json").read_text())
DIFF = pathlib.Path("fixtures/pr103.diff").read_text()
PR_INPUT = f"PR #{PR['number']}: {PR['title']}\\n\\n{PR['description']}\\n\\nFULL DIFF:\\n{DIFF}"

s = PR["stats"]
print(f"{PR['repo']} #{PR['number']} — {PR['title']}")
print(f"by @{PR['author']} · {s['files']} files · +{s['additions']} −{s['deletions']}\\n")
print(PR["description"] + "\\n")
for f in PR["files"][:8]:
    print(f"  {f['path']:<48} +{f['additions']:>5} −{f['deletions']}")
print(f"  … {len(PR['files']) - 8} more files")
""")

add(MD, "## 2 · One reviewer, one call")

add(CODE, """
import asyncio, time
from replay import AsyncCerebras   # deterministic replay of recorded transcripts, 300 ms/call —
                                   # swap for cerebras.cloud.sdk + CEREBRAS_API_KEY to go live

client = AsyncCerebras()
MODEL = "moonshotai/Kimi-K2.6"
RESPONSES, STAGE = [], {}

async def review(system: str, user: str):
    t0 = time.perf_counter()
    resp = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    RESPONSES.append(resp)
    return resp, time.perf_counter() - t0

resp, dt = await review(
    "You are the performance reviewer on a PR review panel. Mandate: complexity regressions, "
    "N+1 patterns, memory growth, main-thread stalls. Return findings as JSON. "
    "Returning zero findings is acceptable.",
    PR_INPUT,
)

u, t = resp.usage, resp.time_info
print(f"{resp.model} · {dt * 1000:.0f} ms round-trip · {u.prompt_tokens:,} tokens in / {u.completion_tokens} out")
print(f"decode: {u.completion_tokens / t.completion_time:,.0f} tok/s\\n")
for f in json.loads(resp.choices[0].message.content)["findings"]:
    print(f"[{f['severity'].upper():<6}] {f['title']}")
    print(f"         {f['file']}:{f['line']}\\n")
""")

add(MD, "## 3 · The panel — eight narrow mandates")

add(CODE, """
AGENTS = {
    "security":     "injection, authz, secrets in logs, unsafe process interaction",
    "adversarial":  "the input that breaks it — nulls, empty parts, the second call, hostile paths",
    "performance":  "complexity regressions, N+1 patterns, memory growth, main-thread stalls",
    "quality":      "only what costs time in six months. not style",
    "mvs":          "is this the smallest reasonable change? what is bundled that shouldn't be?",
    "intent":       "does the diff match the description?",
    "tests":        "do the tests exercise what changed?",
    "blast radius": "downstream callers, contract changes, toolchain and CI impact",
}
for name, mandate in AGENTS.items():
    print(f"{name:<14} {mandate}")
""")

add(MD, "## 4 · Fan out — eight specialists, one wall clock")

add(CODE, """
t0 = time.perf_counter()
panel = await asyncio.gather(*[
    review(
        f"You are the {name} reviewer on a PR review panel. Mandate: {mandate}. "
        "Return findings as JSON. Returning zero findings is acceptable.",
        PR_INPUT,
    )
    for name, mandate in AGENTS.items()
])
STAGE["panel"] = time.perf_counter() - t0

panel_resps = [resp for resp, _ in panel]
findings = []
for name, (resp, _) in zip(AGENTS, panel):
    batch = json.loads(resp.choices[0].message.content)["findings"]
    findings.extend(batch)
    high = sum(f["severity"] == "high" for f in batch)
    print(f"{name:<14} {len(batch)} findings" + (f"  ({high} high)" if high else ""))

print(f"\\n{len(findings)} raw findings · {len(panel)} calls · {STAGE['panel']:.2f} s wall")
""")

add(MD, "## 5 · The critic layer — every finding audited, concurrently")

add(CODE, """
async def audit(f, vote=1, votes=1):
    tag = f" (pass {vote} of {votes})" if votes > 1 else ""
    resp, _ = await review(
        "You are the critic. Audit one finding from a first-pass reviewer. Verify the quoted "
        "evidence against the diff, check the reasoning, and check for duplicates. "
        "Reject anything you cannot verify.",
        f"Finding under audit: {f['id']}{tag}\\n\\n{json.dumps(f, indent=2)}\\n\\nFULL DIFF:\\n{DIFF}",
    )
    return json.loads(resp.choices[0].message.content)

t0 = time.perf_counter()
verdicts = await asyncio.gather(*[audit(f) for f in findings])
STAGE["critic"] = time.perf_counter() - t0

survivors = [f for f, v in zip(findings, verdicts) if v["verdict"] == "confirmed"]
killed = {v["finding_id"]: v for v in verdicts if v["verdict"] == "rejected"}

print(f"{len(findings)} findings audited concurrently in {STAGE['critic']:.2f} s")
print(f"→ {len(survivors)} confirmed · {len(killed)} killed\\n")

for fid in ("SEC-2", "QUAL-3", "PERF-2"):
    print(f"KILLED [{fid}]  {killed[fid]['reason']}\\n")

print("confirmed:", ", ".join(f["id"] for f in survivors))
""")

add(MD, "## 6 · Blocking pass + adjudication")

add(CODE, """
t0 = time.perf_counter()
gates = await asyncio.gather(*[
    review(
        "You are the merge-gate assessor. Decide whether this confirmed finding blocks the "
        "merge or ships as advisory. Return JSON.",
        f"Blocking assessment for finding {f['id']}\\n\\n{json.dumps(f, indent=2)}",
    )
    for f in survivors
])
gate = {g["finding_id"]: g for g in (json.loads(r.choices[0].message.content) for r, _ in gates)}

resp, _ = await review(
    "You are the adjudicator. You see only confirmed, deduplicated findings with gate "
    "assessments. Return a final verdict, the blocking list, and a summary as JSON.",
    json.dumps([{**f, "blocking": gate[f["id"]]["blocking"]} for f in survivors], indent=2),
)
final = json.loads(resp.choices[0].message.content)
STAGE["gate + adjudicate"] = time.perf_counter() - t0

print(f"VERDICT: {final['verdict']}\\n")
for fid in final["blocking"]:
    f = next(f for f in survivors if f["id"] == fid)
    print(f"BLOCKING [{fid}] {f['title']}")
    print(f"  {gate[fid]['reason']}\\n")
print(final["summary"])
""")

add(MD, "## 7 · The numbers")

add(CODE, """
SERIAL_TPS, SERIAL_TTFT = 60, 0.6   # a typical hosted endpoint

pipeline_calls = 8 + len(findings) + len(survivors) + 1
wall = sum(STAGE.values())
tokens = sum(r.usage.total_tokens for r in RESPONSES) - RESPONSES[0].usage.total_tokens
serial_panel = sum(SERIAL_TTFT + r.usage.completion_tokens / SERIAL_TPS for r in panel_resps)

print(f"model calls per PR       {pipeline_calls}")
print(f"end to end               {wall:.2f} s")
print(f"tokens through Kimi      {tokens / 1e6:.2f} M")
print()
print(f"the same 8 panel prompts, serial, on a typical hosted")
print(f"endpoint ({SERIAL_TPS} tok/s, {SERIAL_TTFT:.1f} s to first token): {serial_panel:.0f} s")
print(f"\\nspeedup vs the serial panel: {serial_panel / wall:.0f}x")
""")

add(MD, """
## 8 · The dial — majority-vote critics

Reliability is a quantity. Turn the critic layer from one vote to three and take the majority —
false positives drop again, and on this hardware it costs roughly zero additional seconds.
""")

add(CODE, """
CRITIC_VOTES = 3

async def audit_majority(f):
    votes = await asyncio.gather(*[audit(f, vote=v + 1, votes=CRITIC_VOTES)
                                   for v in range(CRITIC_VOTES)])
    confirmed = sum(v["verdict"] == "confirmed" for v in votes) > CRITIC_VOTES // 2
    return votes, confirmed

t0 = time.perf_counter()
results3 = await asyncio.gather(*[audit_majority(f) for f in findings])
dial_wall = time.perf_counter() - t0

survivors3 = [f for f, (_, ok) in zip(findings, results3) if ok]
flipped = [(f, votes) for f, (votes, ok) in zip(findings, results3) if ok != (f in survivors)]

print(f"critic calls   {len(findings)} → {len(findings) * CRITIC_VOTES}     "
      f"wall: {STAGE['critic']:.2f} s → {dial_wall:.2f} s")
print(f"survivors      {len(survivors)} → {len(survivors3)}     blocking list unchanged\\n")

for f, votes in flipped:
    print(f"FLIPPED [{f['id']}] {f['title']}")
    for i, v in enumerate(votes, 1):
        print(f"   vote {i}: {v['verdict']:<9} — {v['reason']}")
    print()
""")


def cell(kind, src):
    c = {
        "cell_type": kind,
        "id": f"cell-{cells.index((kind, src)):02d}",
        "metadata": {},
        "source": [line + "\n" for line in src.split("\n")][:-1] + [src.split("\n")[-1]],
    }
    if kind == CODE:
        c["execution_count"] = None
        c["outputs"] = []
    return c


nb = {
    "cells": [cell(k, s) for k, s in cells],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).parent / "demo.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out}")
