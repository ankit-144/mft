#!/usr/bin/env python3
"""Publish the rendered diagrams: size them for viewing, and build an index.

Run by `make diagrams` after D2 renders the SVGs. Everything it writes lands in
the gitignored build directory, so this is build output too.

Three jobs:

1. **Stamp explicit `width`/`height` on each SVG.** D2 emits `viewBox` only,
   which means a browser scales the drawing to fit the window. Diagram 03 is
   14731px wide, so fit-to-window renders it at 0.1x — technically correct and
   completely unreadable. Stamping the natural size makes Firefox render it 1:1
   and let you pan, which is how a diagram that size is actually read.

2. **Write a viewer page per diagram.** A scrollable frame, the natural size,
   and a note with the real dimensions so nobody wonders why they are panning.

3. **Write a gallery index** with thumbnails and a reading order.
"""

from __future__ import annotations

import html
import pathlib
import sys
import xml.etree.ElementTree as ET

# Reading order, and the reason each diagram exists. Keyed by the .d2 stem.
ORDER: list[tuple[str, str, str]] = [
    ("01-system-context", "System context", "Who talks to this platform and why."),
    ("02-container-view", "Container view", "What runs where, and what does each service link."),
    ("03-data-flow-tick-to-order", "Data flow: tick to order", "Trace one minute bar from broker to order."),
    ("05-state-stores", "State stores", "What is durable, what is in-memory, what crosses a boundary."),
    ("06-decision-and-risk", "Decision and risk", "A signal arrives. Which gate stops it?"),
    ("04-component-ownership", "Component ownership", "Who owns which paths, and why branches did not conflict."),
    ("07-pull-vs-push", "Pull vs push", "Why inference pulls at minute close."),
    ("08-build-and-branch-strategy", "Build and branch strategy", "How ten components were built in parallel."),
    ("09-runtime-topology", "Runtime topology", "Processes, ports, the fx graph, and a DI failure tests missed."),
    ("10-lld-kite-broker", "LLD: Kite connector", "Connector, instrument master, orders, error mapping."),
    ("11-lld-stream-reconnect", "LLD: stream reconnect", "The WebSocket lifecycle as a state machine."),
    ("12-lld-storage-writer", "LLD: storage writer", "Parquet writer and the atomic-publish invariant."),
    ("13-lld-storage-reader", "LLD: storage reader", "Reader, dedup, and the DuckDB glob seam."),
    ("14-lld-features", "LLD: features", "The 18-column table and the row-alignment guarantee."),
    ("15-lld-ingestion", "LLD: ingestion", "Pipeline, exactly-once bars, backpressure."),
    ("16-lld-risk-engine", "LLD: risk engine", "The seven checks in evaluation order."),
    ("17-lld-execution-api", "LLD: execution API", "HTTP boundary and the exactly-once guarantee."),
    ("18-lld-jobs-backfill", "LLD: jobs backfill", "Backfill, rate limiting, resumability."),
    ("19-lld-inference-model", "LLD: inference model", "The swappable model layer, and why it exists."),
    ("20-lld-inference-loop", "LLD: inference loop", "The minute-close loop, and failing closed."),
    ("21-lld-backtest", "LLD: backtest", "Walk-forward, and the no-lookahead guarantee."),
    ("22-lld-observability", "LLD: observability", "Metrics, logging, and the redaction rules."),
    ("23-crosscut-idempotency", "Exactly-once", "One key across Python, HTTP, Go, and an in-memory map."),
    ("24-crosscut-contracts", "The frozen surface", "What is frozen, who depends on it, where it leaks."),
    ("25-crosscut-failure-modes", "Failure modes", "Retry, fail-fast, fail-closed, drop, refuse-to-start."),
    ("26-crosscut-verification", "Verification", "What is proven, what is hand-verified, what never ran."),
]

HIGH = frozenset(s.split("-")[0] for s, _, _ in ORDER[:5])
CROSS = frozenset(f"{i:02d}" for i in range(23, 27))
LOW = frozenset(s.split("-")[0] for s, _, _ in ORDER) - HIGH - CROSS

GALLERY_CSS = """
  :root {{
    --bg:#fafafa; --fg:#212121; --muted:#5f6368; --line:#e0e0e0;
    --card:#fff; --accent:#1976d2;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg:#14161a; --fg:#e8eaed; --muted:#9aa0a6;
             --line:#2a2e35; --card:#1c1f24; --accent:#64b5f6; }}
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; padding:2rem 1.5rem 4rem; background:var(--bg); color:var(--fg);
         font:15px/1.55 system-ui,-apple-system,sans-serif; }}
  .wrap {{ max-width:1600px; margin:0 auto; }}
  h1 {{ font-size:1.6rem; margin:0 0 .25rem; letter-spacing:-.01em; }}
  .sub,.note {{ color:var(--muted); }}
  .note {{ font-size:.875rem; margin:0 0 2rem; border-left:3px solid var(--accent);
           padding-left:.75rem; }}
  h2 {{ font-size:1.05rem; text-transform:uppercase; letter-spacing:.06em;
        color:var(--muted); margin:2.5rem 0 1rem; font-weight:600; }}
  .grid {{ display:grid; gap:1.25rem;
           grid-template-columns:repeat(auto-fill,minmax(360px,1fr)); }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:10px;
           overflow:hidden; display:flex; flex-direction:column; }}
  .card a {{ text-decoration:none; color:inherit; }}
  .thumb {{ background:#fff; height:200px; overflow:hidden;
            border-bottom:1px solid var(--line); display:flex;
            align-items:flex-start; justify-content:center; }}
  .thumb img {{ width:100%; height:auto; }}
  .body {{ padding:.75rem .9rem 1rem; }}
  .idx {{ color:var(--muted); font-variant-numeric:tabular-nums; font-size:.8rem;
          margin-right:.4rem; }}
  .title {{ font-weight:600; }}
  .desc {{ color:var(--muted); font-size:.875rem; margin-top:.3rem; }}
  .dim {{ color:var(--muted); font-size:.75rem; margin-top:.4rem;
          font-variant-numeric:tabular-nums; }}
  footer {{ margin-top:3rem; color:var(--muted); font-size:.8rem; }}
  code {{ background:var(--bg); padding:.1rem .35rem; border-radius:4px;
          border:1px solid var(--line); font-size:.85em; }}
"""

VIEWER_CSS = """
  html,body {{ margin:0; background:#f6f7f9; color:#212121;
              font:14px/1.5 system-ui,-apple-system,sans-serif; }}
  @media (prefers-color-scheme: dark) {{
    html,body {{ background:#0f1114; color:#e8eaed; }}
  }}
  header {{ position:sticky; top:0; z-index:2; display:flex; gap:1rem;
            align-items:baseline; flex-wrap:wrap;
            padding:.6rem 1rem; background:rgba(255,255,255,.94);
            border-bottom:1px solid #e0e0e0; backdrop-filter:blur(6px); }}
  @media (prefers-color-scheme: dark) {{
    header {{ background:rgba(20,22,26,.94); border-bottom-color:#2a2e35; }}
  }}
  header b {{ font-size:1rem; }}
  header span, header a {{ color:#5f6368; font-size:.82rem; }}
  header a {{ color:#1976d2; }}
  .frame {{ overflow:auto; max-height:calc(100vh - 3rem); padding:1rem; }}
  .frame img {{ display:block; background:#fff; border:1px solid #dcdfe3;
                border-radius:6px; max-width:none; }}
  kbd {{ border:1px solid #b9bec5; border-bottom-width:2px; border-radius:4px;
         padding:0 .3rem; font-size:.78rem; }}
"""

GALLERY = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MFT platform &mdash; architecture diagrams</title>
<style>{css}</style>
</head>
<body><div class="wrap">
<h1>MFT platform &mdash; architecture diagrams</h1>
<p class="sub">{count} diagrams, D2 &rarr; SVG.</p>
<p class="note">Click a thumbnail for the full-size diagram in a scrollable frame.
   Source and reading order live in <code>docs/diagrams/README.md</code>.</p>
{sections}
<footer>Generated by <code>make diagrams</code> &mdash; build output, not committed.</footer>
</div></body>
</html>
"""

VIEWER = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{css}</style>
</head>
<body>
<header>
  <b>{title}</b>
  <span>{desc}</span>
  <span>natural size {w} &times; {h} &mdash; pan to scroll, <kbd>Ctrl</kbd>+scroll to zoom</span>
  <a href="index.html">&larr; all diagrams</a>
  <a href="{stem}.svg">raw SVG</a>
</header>
<div class="frame"><img src="{stem}.svg" width="{w}" height="{h}" alt="{title}"></div>
</body></html>
"""


def svg_size(path: pathlib.Path) -> tuple[int, int]:
    """Return the natural (width, height) from the SVG's viewBox."""
    root = ET.parse(path).getroot()
    _, _, w, h = (float(v) for v in root.get("viewBox").split())
    return int(w), int(h)


def stamp_size(path: pathlib.Path) -> tuple[int, int]:
    """Give the SVG an explicit width/height so browsers render it 1:1.

    D2 omits them and sets preserveAspectRatio=meet, so a browser scales the
    whole drawing into the window. For a 14k-px-wide diagram that is 0.1x and
    unreadable. Stamping the natural size makes it pan instead.
    """
    w, h = svg_size(path)
    text = path.read_text(encoding="utf-8")
    root_end = text.index(">", text.index("<svg"))
    root_tag = text[text.index("<svg") : root_end]
    if 'width="' not in root_tag:
        new_tag = root_tag.rstrip() + f' width="{w}" height="{h}"'
        text = text[: text.index("<svg")] + new_tag + text[root_end:]
        path.write_text(text, encoding="utf-8")
    return w, h


def publish(out_dir: pathlib.Path) -> int:
    svgs = sorted(out_dir.glob("*.svg"))
    if not svgs:
        print(f"no SVGs in {out_dir}; run 'make diagrams' first", file=sys.stderr)
        return 1

    sizes: dict[str, tuple[int, int]] = {}
    for p in svgs:
        sizes[p.stem] = stamp_size(p)

    titles = {s: (t, d) for s, t, d in ORDER}

    for stem, (w, h) in sizes.items():
        title, desc = titles.get(stem, (stem, ""))
        (out_dir / f"{stem}.html").write_text(
            VIEWER.format(
                css=VIEWER_CSS, title=html.escape(title), desc=html.escape(desc),
                stem=html.escape(stem), w=w, h=h,
            ),
            encoding="utf-8",
        )

    sections: list[str] = []
    for label, group in (
        ("Start here &mdash; the five that build from the outside in", HIGH),
        ("Component detail", LOW),
        ("Cross-cutting", CROSS),
    ):
        items = [(s, t, d) for s, t, d in ORDER if s in sizes and s.split("-")[0] in group]
        if not items:
            continue
        cards = []
        for i, (stem, title, desc) in enumerate(items, 1):
            w, h = sizes[stem]
            cards.append(
                f'    <div class="card"><a href="{html.escape(stem)}.html">\n'
                f'      <div class="thumb"><img loading="lazy" '
                f'src="{html.escape(stem)}.svg" alt=""></div>\n'
                f'      <div class="body"><div class="title">'
                f'<span class="idx">{i}</span>{html.escape(title)}</div>'
                f'<div class="desc">{html.escape(desc)}</div>'
                f'<div class="dim">{w} &times; {h} px</div></div>\n'
                f"    </a></div>"
            )
        sections.append(
            f'  <h2>{label}</h2>\n  <div class="grid">\n' + "\n".join(cards) + "\n  </div>"
        )

    (out_dir / "index.html").write_text(
        GALLERY.format(css=GALLERY_CSS, count=len(svgs), sections="\n\n".join(sections)),
        encoding="utf-8",
    )
    return len(svgs)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: mkindex.py <svg-dir>", file=sys.stderr)
        return 2
    out_dir = pathlib.Path(sys.argv[1])
    if not out_dir.is_dir():
        print(f"no such directory: {out_dir}", file=sys.stderr)
        return 2
    n = publish(out_dir)
    print(f"published {n} diagrams + viewers -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())