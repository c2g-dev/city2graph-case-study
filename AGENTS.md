# city2graph case study

## Project overview

Reproducible Liverpool case-study workflow for city2graph, supporting Paper 1.
Source code lives in `src/`, experiment configuration in
`configs/experiment_config.yaml`, and runnable entry points in `scripts/`.

Directory quirk: this Git repository is nested one level down; the outer
`city2graph-case-study/` directory is a non-Git wrapper that only carries local
`.claude/` configuration. Run all Git commands from this inner directory.

Paper 1 prose, readings, and reviews are canonical in Notion: engage the
`notion-research` skill for content work. This repository holds the code,
generated figures, and tables that the paper cites.

## Environment and commands

- Managed with `uv`; run everything through `uv run <cmd>`.
- Uses city2graph as a dependency; when a change spans the library and this
  case study, verify the library first.

## Scope and context

- Do not scan `data/` (raw, processed, and outputs), `notebooks/`,
  `notebooks_samples/`, `data.zip`, `__pycache__/`, or `.venv/` in broad
  searches; they hold large datasets and generated artefacts.
- Preserve pre-existing working-tree changes; do not commit unless asked.

## Conventions

- English for code, comments, and docstrings.
- Reproducibility first: outputs cited by Paper 1 must be regenerable from
  configs and scripts; do not hand-edit generated figures or tables.
