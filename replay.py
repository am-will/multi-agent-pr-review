"""Drop-in stand-in for `cerebras.cloud.sdk.AsyncCerebras`.

Replays recorded review transcripts (fixtures/responses.json) with a fixed
300 ms round-trip per call, so the demo is fully deterministic on stage.
To run live instead, `pip install cerebras-cloud-sdk`, set CEREBRAS_API_KEY,
and swap this import for the real SDK — the call sites are identical.
"""

import asyncio
import json
import re
from pathlib import Path
from types import SimpleNamespace

LATENCY_S = 0.300          # fixed round-trip per call
DECODE_TPS = 2400.0        # decode speed reflected in time_info

_FIXTURES = json.loads(
    (Path(__file__).parent / "fixtures" / "responses.json").read_text()
)

_PROMPT_TOKENS = {
    "panel": 61_432,       # full diff + description
    "critic": 2_418,       # one finding + evidence window
    "blocking": 1_137,     # one confirmed finding
    "adjudicator": 3_804,  # survivors + blocking flags
}


def _route(text: str):
    """Map a prompt to its recorded response. Returns (payload, kind)."""
    m = re.search(r"Finding under audit: ([A-Z]+-\d+)", text)
    if m:
        fid = m.group(1)
        vote = re.search(r"\(pass (\d) of \d\)", text)
        idx = int(vote.group(1)) - 1 if vote else 0
        rec = _FIXTURES["critic"][fid]
        return {**rec["votes"][idx], "finding_id": fid}, "critic", rec["completion_tokens"]

    m = re.search(r"Blocking assessment for finding ([A-Z]+-\d+)", text)
    if m:
        rec = _FIXTURES["blocking"][m.group(1)]
        return (
            {"finding_id": m.group(1), "blocking": rec["blocking"], "reason": rec["reason"]},
            "blocking",
            rec["completion_tokens"],
        )

    if "You are the adjudicator" in text:
        rec = _FIXTURES["adjudicator"]
        return (
            {"verdict": rec["verdict"], "blocking": rec["blocking"], "summary": rec["summary"]},
            "adjudicator",
            rec["completion_tokens"],
        )

    m = re.search(r"You are the ([a-z ]+?) reviewer", text)
    if m:
        rec = _FIXTURES["panel"][m.group(1)]
        return {"findings": rec["findings"]}, "panel", rec["completion_tokens"]

    raise ValueError("no recorded transcript matches this prompt")


class _Completions:
    def __init__(self, client):
        self._client = client

    async def create(self, model, messages, **kwargs):
        text = "\n".join(m["content"] for m in messages)
        payload, kind, completion_tokens = _route(text)
        await asyncio.sleep(LATENCY_S)
        self._client.calls += 1

        prompt_tokens = _PROMPT_TOKENS[kind]
        return SimpleNamespace(
            id=f"chatcmpl-replay-{self._client.calls:04d}",
            model=model,
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(role="assistant", content=json.dumps(payload)),
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            time_info=SimpleNamespace(
                queue_time=0.014,
                prompt_time=round(prompt_tokens / 190_000, 4),
                completion_time=round(completion_tokens / DECODE_TPS, 4),
                total_time=LATENCY_S,
            ),
        )


class AsyncCerebras:
    def __init__(self, api_key: str | None = None):
        self.calls = 0
        self.chat = SimpleNamespace(completions=_Completions(self))
