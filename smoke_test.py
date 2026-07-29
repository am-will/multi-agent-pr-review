#!/usr/bin/env python3
"""Run every code cell of demo.ipynb in order, exactly as Jupyter would."""

import asyncio
import json
import textwrap
from pathlib import Path

nb = json.loads((Path(__file__).parent / "demo.ipynb").read_text())
code = "\n\n".join(
    "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"
)

wrapped = "async def _main():\n" + textwrap.indent(code, "    ")
ns = {}
exec(compile(wrapped, "demo.ipynb", "exec"), ns)
asyncio.run(ns["_main"]())
