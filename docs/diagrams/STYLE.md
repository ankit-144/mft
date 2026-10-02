# Diagram style guide

Every `.d2` file in this directory follows these conventions so that twenty-odd
independently-authored diagrams still read as one set.

## Rendering

```sh
make diagrams          # render every .d2 to docs/diagrams/svg/
make diagrams-check    # parse-check every .d2 (the CI form)
```

D2 **v0.9.0**, layout engine `dagre`.

## D2 v0.9.0 syntax traps

These are verified against this exact version. Each one has already cost an
agent a wasted render.

1. **`direction` needs a colon.** `direction: down` is correct.
   `direction down` — no colon — does **not** fail; it silently parses as a
   node named "direction down" and renders a junk rectangle above your
   diagram. Verified: 3 shapes instead of 2.
2. **Classes go in a `classes:` block.** Writing `go: Go service { … }` at top
   level creates a real, visible node. Verified: 3 shapes instead of 2.
   ```d2
   classes: {
     go: { style.fill: "#e3f2fd"; style.stroke: "#1976d2" }
   }
   a: Ingestion { class: go }
   ```
3. **`class:` is not a glob.** `class: @go` is an *import* in v0.9.0 and will
   not resolve. Use `class: go`.
4. **`shape` is a reserved keyword.** Do not name a node `shape`.
5. **Markdown soft line breaks collapse.** In an `|md` block, every line must
   be separated by a blank line, or the whole block renders as one very long
   line.
6. **Braces in labels are painful.** `\{` escapes interpolation but the
   backslash still gets painted, and `${'{'}` does not resolve. Write
   `read_parquet(...)` rather than fighting it.
7. **dagre emits a ghost key-labelled shape layer** for any container that has
   an edge between its children. It is cosmetic; do not try to fix it by
   restructuring the diagram.
8. `font-family` is rejected inside a `classes:` block.

## The hard rules

1. **Every node label must be a real thing from the code.** Real type names,
   real function names, real config keys, real error codes, real file paths.
   If you cannot point at a symbol, the node does not belong in the diagram.
   This is the single rule that separates a useful diagram from generic
   box-and-arrow filler.
2. **No invented behaviour.** Every edge is something the code actually does.
   If you are inferring an edge, label it as such.
3. **Show the unhappy path.** A diagram that only draws the happy path is
   decoration. Where the code fails closed, rejects, retries, drops, sweeps or
   panics, show it — in `error` style, with the real reason code.
4. **One diagram, one question.** If a diagram needs a sentence of preamble to
   explain what it is for, it is two diagrams.
5. **Cite the source.** Every diagram has a footer node naming the files it was
   derived from, so a reader can check it and a future change knows what to
   update.
6. **Report contradictions, do not smooth them over.** Several of these
   diagrams were written while the repo contained real inconsistencies between
   its documentation and its code. Where you find one, draw it truthfully or
   flag it in your report.

## Shape and colour vocabulary

Colours carry meaning across the whole set. Do not invent new ones.

| Meaning | Fill | Stroke | Class |
| :--- | :--- | :--- | :--- |
| External system (Kite, HuggingFace, Docker, Prometheus) | `#eceff1` | `#90a4ae` | `ext` |
| Go service | `#e3f2fd` | `#1976d2` | `go` |
| Python service | `#ede7f6` | `#5e35b1` | `py` |
| Shared core library | `#e0f2f1` | `#00897b` | `lib` |
| Durable store (Parquet, DuckDB, disk) | `#e8f5e9` | `#2e7d32` | `store` |
| In-memory / process-local state | `#fff8e1` | `#f9a825` | `mem` |
| Control plane / HTTP | `#e5e7eb` | `#37474f` | `ctl` |
| Risk, safety, failure — **use sparingly** | `#ffebee` | `#c62828` | `bad` |

```d2
direction: down

vars: {
  d2-config: {
    theme-id: 0
    layout-engine: dagre
    pad: 40
  }
}

classes: {
  ext:   { style.fill: "#eceff1"; style.stroke: "#90a4ae" }
  go:    { style.fill: "#e3f2fd"; style.stroke: "#1976d2" }
  py:    { style.fill: "#ede7f6"; style.stroke: "#5e35b1" }
  lib:   { style.fill: "#e0f2f1"; style.stroke: "#00897b" }
  store: { style.fill: "#e8f5e9"; style.stroke: "#2e7d32" }
  mem:   { style.fill: "#fff8e1"; style.stroke: "#f9a825" }
  ctl:   { style.fill: "#e5e7eb"; style.stroke: "#37474f" }
  bad:   { style.fill: "#ffebee"; style.stroke: "#c62828" }
}
```

Rules:
- `bad` is for something that genuinely blocks or rejects. More than about four
  `bad` nodes and the signal is lost.
- `mem` and `store` must always be visually distinguishable. That distinction is
  the core design decision in `Plan.md` §4 and readers should get it at a glance.
- Prefer keeping containers for things that genuinely contain something.
  dagre handles a container with internal edges poorly (see trap 7); a flat list
  of top-level nodes with a colour class is often the more readable choice.

## Labelling

- Multi-line labels: `"line one\nline two"`.
- Go symbols as `Kite.Stream(ctx, symbols)` — receiver, method, args.
- Error and rejection codes in caps: `RISK_MAX_DRAWDOWN`, `BAD_REQUEST`.
- Config keys as plain label text: `inference.dry_run`.
- Edge labels say what flows and, where it matters, what can happen instead:
  `Append(tick) -> error`, `blocked → drop oldest`.
- Single quotes inside a quoted label for apostrophes: `'it is'`.

## Sizing

- Target 15–40 nodes. Above ~50 nobody reads it.
- `direction: down` for pipelines and sequences, `direction: right` for layered
  views and state/ownership maps.
- If a diagram is unavoidably wide, prefer nesting over arrow spaghetti.

## Housekeeping

- Do not commit the generated `.svg` files — they are build output and
  `.gitignore`d.
- File names are `NN-short-name.d2`, matching the index in `README.md`.
- Do not edit `README.md` or another agent's `.d2` file. If another diagram
  needs a change, say so in your report.