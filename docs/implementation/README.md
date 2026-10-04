# MFT responsibility map

Open [guide.html](guide.html) directly in a browser. The page is self-contained and works offline.

For setup, dummy data, public candles, paper services and integration checks, see [the local run guide](../LOCAL_RUN.md).

## Reading path

1. Start with the high-level wiring diagram.
2. Click a service to see its smaller responsibility blocks.
3. Click a block to read what it owns, what crosses its boundary, and the operations callers can rely on.
4. Use breadcrumbs or the left-hand tree to move back through the hierarchy.

Example: **Whole system → Execution → Risk → Account book · book.go**.

The guide contains 75 connected responsibility nodes tied to master `6e41b7d`. It includes a searchable tree, flow highlighting, a glossary, light/dark themes, and source links pinned to that commit. Source links require GitHub access; the guide itself works offline.

## Files

- `guide.html`: the new responsibility map.
- `map-template.html`: the page layout and navigation.
- `map-content.json`: curated responsibility descriptions, relationships and contracts.
- `map-data.json`: the combined generated map.
- `build_map.py`: combines the descriptions, validates node references and source paths, and embeds the data.

When interfaces or wiring change, update the curated descriptions and rebuild from the repository root:

```sh
python3 docs/implementation/build_map.py
```

The guide, its editable sources and the local run instructions are tracked in Git. Other documentation, local configs, experiment reports and visual artifacts remain ignored.

GitHub displays HTML as source. After cloning this branch, open `docs/implementation/guide.html` locally to use the interactive page. No web server or frontend dependencies are required.
