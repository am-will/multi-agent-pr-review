# Multi-Agent PR Review

Fan out → critic-filter → adjudicate. A PR review panel of eight narrow-mandate
reviewers running `moonshotai/Kimi-K2.6` on Cerebras Inference, every finding
adversarially audited before a single adjudication call issues the verdict.

Companion notebook for the *Low Latency PR Reviews on Cerebras Fast Inference*
workshop. The demo reviews a real pull request:
[am-will/limux#103](https://github.com/am-will/limux/pull/103) (+3,545 −335, 34 files).

## Run it

```bash
jupyter lab demo.ipynb
```

The notebook ships in **replay mode**: model responses are recorded transcripts
(`fixtures/responses.json`) replayed at a fixed 300 ms per call, so the run is
deterministic on stage and needs no API key. To run live, install
`cerebras-cloud-sdk`, set `CEREBRAS_API_KEY`, and swap the `replay` import in
cell 2 for the real SDK — the call sites are identical.

## Layout

| file | what |
|---|---|
| `demo.ipynb` | eight cells, each adds exactly one thing |
| `replay.py` | drop-in `AsyncCerebras` that replays the fixtures |
| `fixtures/pr_103.json` | PR metadata |
| `fixtures/pr103.diff` | the real diff, fed to every panel prompt |
| `fixtures/responses.json` | recorded panel findings, critic votes, gate calls, verdict |
| `make_notebook.py` | regenerates `demo.ipynb` |
| `smoke_test.py` | executes every cell in order, headless |
