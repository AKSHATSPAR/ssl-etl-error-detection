"""Inline the exported JSON into the demo page, producing one standalone file.

    PYTHONPATH=src python scripts/build_web_demo.py

The output has no external requests of any kind, so it opens by double-clicking
on a machine with no internet.
"""

from __future__ import annotations

import json
from pathlib import Path

WEB = Path("artifacts/web")
tpl = (WEB / "demo_template.html").read_text()
data = json.loads((WEB / "demo_data.json").read_text())
out = WEB / "pipeline_demo.html"
out.write_text(tpl.replace("__DEMO_DATA__", json.dumps(data, separators=(",", ":"))))
size_kb = out.stat().st_size / 1024
assert "__DEMO_DATA__" not in out.read_text(), "data placeholder was not replaced"
# Look for things the browser would actually fetch. The SVG and XML namespace
# URLs are identifiers rather than requests, so they are excluded by name.
html = out.read_text()
for ns in ("http://www.w3.org/2000/svg", "http://www.w3.org/1999/xhtml"):
    html = html.replace(ns, "")
for bad in ("http://", "https://", "//cdn", "@import", "<link"):
    assert bad not in html, f"external reference {bad!r} would break offline use"
print(f"wrote {out} ({size_kb:.0f} KB, {len(data['runs'])} runs, no external requests)")
