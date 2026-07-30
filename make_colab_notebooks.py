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
    if "from replay import AsyncCerebras" not in text:
        continue
    cell["source"] = source(
        """
import asyncio, json, os, time
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

client = AsyncCerebras(api_key=API_KEY)
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

async def review(system: str, user: str):
    t0 = time.perf_counter()
    resp = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    RESPONSES.append(resp)
    return resp, time.perf_counter() - t0

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
        cell["source"] = source(text.replace("tokens through Kimi", "tokens through Gemma"))

(ROOT / "demo_live.ipynb").write_text(
    json.dumps(live, indent=1, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

print("wrote demo.ipynb and demo_live.ipynb")
