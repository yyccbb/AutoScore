# Agent Instructions

Do not read, print, summarize, or load the full dataset files unless explicitly requested.

Large data files include:

- dataset.json
- data/*.json
- data/**/*.json
- *.jsonl
- *.csv

When inspecting datasets, only use scripts that report:
- file size
- number of records if cheap to compute
- schema / keys
- first 1-3 samples
- truncated output under 2000 characters

Never run commands like:
- type dataset.json
- cat dataset.json
- Get-Content dataset.json without -TotalCount