# `baselines/` — Upstream baseline source

This folder holds the **upstream source code** of every memory-system
baseline MemoryBench depends on. The Python code under [`../src/agent/`](../src/agent/)
imports directly from these directories, so they must be present on disk
for the corresponding baselines to load.

## What's here

| Folder            | Upstream                                                | How it ships |
|-------------------|---------------------------------------------------------|--------------|
| `A_mem/`          | https://github.com/agiresearch/A-mem                    | Tracked in git |
| `MemoryOS/`       | https://github.com/BAI-LAB/MemoryOS                     | Tracked in git |
| `mem0/`           | https://github.com/mem0ai/mem0                          | Tracked in git (editable install — see below) |
| `raptor/`         | https://github.com/parthsarthi03/raptor                 | Tracked in git |
| `BEAM/` (LIGHT)   | https://github.com/mohammadtavakoli78/BEAM              | **Reference-only — not tracked.** See below. |

## Why some are tracked and some aren't

The four baselines `src/agent/*.py` actually imports from (`a_mem`, `mem0`,
`memoryos`, `raptor`) are vendored in-tree so a fresh `git clone` of
MemoryBench is enough to run experiments — no extra fetching, no submodule
dance.

LIGHT is the exception. The MemoryBench implementation of LIGHT lives
entirely in [`../src/agent/light.py`](../src/agent/light.py) — it doesn't
import anything from `baselines/BEAM/`. The BEAM clone here is a
**reference mirror** kept on the developer's machine for code-reading,
not a runtime dependency. To keep `git status` clean we list it in the
top-level [`.gitignore`](../.gitignore).

## Getting the BEAM mirror locally (optional)

If you want to read the upstream LIGHT source alongside the MemoryBench
implementation:

```bash
cd baselines/
git clone --depth 1 https://github.com/mohammadtavakoli78/BEAM.git
```

Nothing else is required — no install, no PYTHONPATH change. MemoryBench
won't try to import from it.

## Editable-install requirement for `mem0`

`baselines/mem0/` is consumed as a Python package via the imports in
[`../src/agent/mem0.py`](../src/agent/mem0.py). After cloning MemoryBench
you have to install it once in editable mode:

```bash
cd baselines/mem0
pip install -e .
```

This is documented in the top-level [README.md](../README.md) under
"Environment Setup".

## Adding a new upstream baseline

If your baseline's MemoryBench agent imports from an upstream package
(like `mem0` and `a_mem` do), vendor the upstream source here so a fresh
clone runs without extra fetch steps:

```bash
cd baselines/
git clone --depth 1 https://github.com/<org>/<repo>.git <Name>
rm -rf <Name>/.git    # drop the nested .git so the outer repo can track files
git add <Name>
```

If your baseline reimplements the algorithm in MemoryBench style (no
upstream import — like LIGHT does), keep the upstream as a
reference-only mirror and add the folder to `.gitignore`:

```
baselines/<Name>/
```

The walkthrough in [../CONTRIBUTING.md](../CONTRIBUTING.md) covers this
trade-off in more detail.
