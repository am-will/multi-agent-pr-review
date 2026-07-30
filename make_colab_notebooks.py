#!/usr/bin/env python3
"""Make replay and live-fire Colab notebooks from the checked-in demo."""

import copy
import json
from pathlib import Path


ROOT = Path(__file__).parent
REPO_URL = "https://github.com/am-will/multi-agent-pr-review.git"


def source(text):
    lines = text.strip("\n").splitlines()
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def code_cell(cell_id, text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source(text),
    }


def set_colab_metadata(notebook, name):
    notebook.setdefault("metadata", {})["colab"] = {
        "name": name,
        "provenance": [],
    }


replay = json.loads((ROOT / "demo.ipynb").read_text(encoding="utf-8"))
set_colab_metadata(replay, "Multi-Agent PR Review — Replay")

bootstrap = code_cell(
    "colab-bootstrap",
    f"""
# Colab bootstrap: fetch the fixture files when the notebook is opened from GitHub.
import os, pathlib, subprocess

if not pathlib.Path("fixtures/pr_103.json").exists():
    subprocess.run(["git", "clone", "--depth", "1", "{REPO_URL}"], check=True)
    os.chdir("multi-agent-pr-review")

print("Replay mode ready — no API key required.")
""",
)

if not any(cell.get("id") == "colab-bootstrap" for cell in replay["cells"]):
    replay["cells"].insert(1, bootstrap)

(ROOT / "demo.ipynb").write_text(
    json.dumps(replay, indent=1, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

live = copy.deepcopy(replay)
set_colab_metadata(live, "Multi-Agent PR Review — Live Fire")
live["cells"][0]["source"] = source(
    """
# Multi-Agent PR Review — live fire

**Gemma 4 31B on Cerebras Inference** · fan out → critic-filter → adjudicate

This notebook makes real API calls. The replay notebook remains the safe,
deterministic presentation fallback.
"""
)

live["cells"][1] = code_cell(
    "live-colab-bootstrap",
    f"""
# Colab bootstrap: fetch fixtures and install the official Cerebras SDK.
import os, pathlib, subprocess, sys

if not pathlib.Path("fixtures/pr_103.json").exists():
    subprocess.run(["git", "clone", "--depth", "1", "{REPO_URL}"], check=True)
    os.chdir("multi-agent-pr-review")

try:
    import cerebras.cloud.sdk
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "--upgrade", "cerebras_cloud_sdk"]
    )

print("Live runtime ready. Add CEREBRAS_API_KEY in Colab Secrets before continuing.")
""",
)

for cell in live["cells"]:
    text = "".join(cell.get("source", []))
    if "## 8 · The dial" in text:
        cell["source"] = source(
            text.replace(
                "## 8 · The dial — majority-vote critics",
                "## 8 · Optional reliability dial — disabled by default",
            )
            + "\n\nThe standard live run skips this expensive stage. Set "
            "`RUN_MAJORITY_VOTE = True` below only when you want three independent "
            "critic votes per finding."
        )
    if "CRITIC_VOTES = 3" in text:
        indented = "\n".join(
            ("    " + line if line else "")
            for line in text.splitlines()
        )
        cell["source"] = source(
            "RUN_MAJORITY_VOTE = False\n\n"
            "if RUN_MAJORITY_VOTE:\n"
            + indented
            + "\nelse:\n"
            '    print("Majority-vote critics skipped. Set RUN_MAJORITY_VOTE = True to run them.")'
        )

for cell in live["cells"]:
    text = "".join(cell.get("source", []))
    if 'DIFF = pathlib.Path("fixtures/pr103.diff").read_text()' not in text:
        continue
    text = text.replace(
        'DIFF = pathlib.Path("fixtures/pr103.diff").read_text()',
        'FULL_DIFF = pathlib.Path("fixtures/pr103.diff").read_text()\n'
        "MAX_DIFF_CHARS = 175_000  # leaves headroom inside Gemma 4 31B's 65,536-token context\n"
        'DIFF = FULL_DIFF[:MAX_DIFF_CHARS]',
    )
    text += (
        '\n\nif len(FULL_DIFF) > len(DIFF):\n'
        '    print(f"Live context budget: reviewing {len(DIFF):,} of {len(FULL_DIFF):,} diff characters.")'
    )
    cell["source"] = source(text)
    break

for cell in live["cells"]:
    text = "".join(cell.get("source", []))
    replay_only = '''for fid in ("SEC-2", "QUAL-3", "PERF-2"):
    print(f"KILLED [{fid}]  {killed[fid]['reason']}\\n")
'''
    if replay_only not in text:
        continue
    live_summary = '''if killed:
    for fid, result in list(killed.items())[:3]:
        print(f"KILLED [{fid}]  {result['reason']}\\n")
else:
    print("No findings were rejected in this live run.\\n")
'''
    cell["source"] = source(text.replace(replay_only, live_summary))
    break

for cell in live["cells"]:
    text = "".join(cell.get("source", []))
    marker = '    batch = json.loads(resp.choices[0].message.content)["findings"]\n'
    if marker not in text:
        continue
    unique_ids = marker + '''    for index, finding in enumerate(batch, 1):
        finding["id"] = f"{name.upper().replace(' ', '_')}-{index}"
'''
    cell["source"] = source(text.replace(marker, unique_ids))
    break

for cell in live["cells"]:
    text = "".join(cell.get("source", []))
    if "from replay import AsyncCerebras" not in text:
        continue
    cell["source"] = source(
        """
import asyncio, json, os, random, time
import cerebras.cloud.sdk as cerebras_sdk
from cerebras.cloud.sdk import AsyncCerebras

# In Colab: click the key icon in the left sidebar and add CEREBRAS_API_KEY.
# If your dedicated endpoint ID does not contain "gemma", also add CEREBRAS_MODEL.
try:
    from google.colab import userdata
except ImportError:
    userdata = None

def secret(name):
    if userdata is not None:
        try:
            return userdata.get(name)
        except userdata.SecretNotFoundError:
            pass
    return os.environ.get(name)

API_KEY = secret("CEREBRAS_API_KEY")
MODEL_OVERRIDE = secret("CEREBRAS_MODEL")

if not API_KEY:
    raise RuntimeError("Add CEREBRAS_API_KEY to Colab Secrets, then rerun this cell.")

client = AsyncCerebras(api_key=API_KEY, max_retries=2)
catalog = await client.models.list()
AVAILABLE_MODELS = [model.id for model in catalog.data]

gemma_models = [
    model_id for model_id in AVAILABLE_MODELS
    if "gemma" in model_id.lower() and ("31" in model_id or "4" in model_id)
]
MODEL = MODEL_OVERRIDE or (
    "gemma-4-31b" if "gemma-4-31b" in AVAILABLE_MODELS
    else (gemma_models[0] if gemma_models else None)
)

if not MODEL:
    raise RuntimeError(
        "This key has no visible Gemma 4 31B endpoint. Gemma 4 is a Cerebras "
        "Dedicated Endpoint model. Ask Cerebras for access, or add the provisioned "
        "endpoint ID as a CEREBRAS_MODEL Colab Secret. Available models: "
        + ", ".join(AVAILABLE_MODELS)
    )

if "gemma" not in MODEL.lower() and not MODEL_OVERRIDE:
    raise RuntimeError(f"Refusing to substitute a non-Gemma model: {MODEL}")

RESPONSES, STAGE = [], {}

def json_contract(system):
    if "adjudicator" in system:
        return (
            'Return exactly one JSON object with this shape: '
            '{"verdict":"APPROVE or REQUEST_CHANGES","blocking":["FINDING-ID"],'
            '"summary":"concise final assessment"}.'
        )
    if "merge-gate" in system:
        return (
            'Return exactly one JSON object with this shape: '
            '{"finding_id":"the supplied finding ID","blocking":true,'
            '"reason":"concise gate rationale"}.'
        )
    if "critic" in system:
        return (
            'Return exactly one JSON object with this shape: '
            '{"finding_id":"the supplied finding ID",'
            '"verdict":"confirmed or rejected","reason":"concise audit rationale"}.'
        )
    return (
        'Return exactly one JSON object with this shape: '
        '{"findings":[{"id":"CATEGORY-1","severity":"high or medium or low",'
        '"title":"concise title","file":"path","line":1,'
        '"evidence":"verifiable evidence","reasoning":"why it matters"}]}. '
        'If there are no findings, return {"findings":[]}.'
    )

async def review(system: str, user: str):
    attempts = 8
    for attempt in range(attempts):
        t0 = time.perf_counter()
        try:
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system + "\\n\\n" + json_contract(system)},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_completion_tokens=1200,
            )
            RESPONSES.append(resp)
            return resp, time.perf_counter() - t0
        except (
            cerebras_sdk.RateLimitError,
            cerebras_sdk.APIConnectionError,
            cerebras_sdk.InternalServerError,
        ):
            if attempt == attempts - 1:
                raise
            delay = min(30, 2 ** attempt) + random.random()
            print(f"Cerebras busy; retrying in {delay:.1f}s ({attempt + 1}/{attempts - 1})")
            await asyncio.sleep(delay)

print("Live model:", MODEL)

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
"""
    )
    break

for cell in live["cells"]:
    text = "".join(cell.get("source", []))
    if "tokens through Kimi" in text:
        text = text.replace("tokens through Kimi", "tokens through Gemma")
        text = text.replace(
            "serial_panel / wall",
            "serial_panel / STAGE['panel']",
        )
        cell["source"] = source(text)

(ROOT / "demo_live.ipynb").write_text(
    json.dumps(live, indent=1, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

print("wrote demo.ipynb and demo_live.ipynb")
