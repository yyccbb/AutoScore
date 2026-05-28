# Project Layout

This repository keeps runtime entrypoints at the root and groups supporting
files by purpose.

- `main.py`: OCR plus scoring pipeline entrypoint.
- `ASRO/`: ASRO training, optimization, sampling, and ASRO-specific utilities.
- `utils/`: Shared pipeline utilities such as LLM API calls, prompts, config,
  and reusable OCR helpers.
- `configs/`: Ready-to-run YAML configs for demo and full runs.
- `scripts/`: Standalone helper scripts and shell launchers.
- `tests/`: Local smoke tests and test helpers.
- `docs/`: Notes and project documentation.
- `data/`: Local datasets and sample input files.
- `results/`: Generated pipeline outputs.

Keep large data files out of source inspection. Use
`data/**/dataset_sample_schema.json` for schema checks instead of opening
`dataset.json`.
