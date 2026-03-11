# Paper 06 Code Plan

## Objective
Produce a reproducible comparative/falsification bundle across Papers I-V.

## Initial scripts
1. `compute_series_comparison.py`
   - Aggregates key JSON outputs from paper-03, paper-04, paper-05.
   - Writes a compact comparison artifact.
2. `compute_falsification_table.py`
   - Builds explicit test statements and exclusion logic windows.
3. `compute_paper06_consistency_summary.py`
   - Merges all Paper-06 artifacts into one final summary file.

## Mandatory outputs
- `papers/paper-06/data/series_comparison.json`
- `papers/paper-06/data/falsification_table.json`
- `papers/paper-06/data/paper06_consistency_summary.json`
