#!/usr/bin/env python3
"""Run residual non-closure diagnostic for twisted-multi-ring."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.lib.tcvphi.multiring_residual_diagnostics import diagnose_residual

DATA_DIR = ROOT / 'papers' / 'paper-08' / 'data'
NOTES_DIR = ROOT / 'notes'


def write_notes(payload: dict[str, object]) -> None:
    flags = payload['interpretation_flags']
    best_sources = payload['source_overlap_summary']['best_low3_sources']
    worst_sources = payload['source_overlap_summary']['worst_tail_sources']
    workbook = f"""# Multiring residual non-closure workbook

## Purpose

This note does not scan new geometries or topologies. It diagnoses where the residual non-closure actually lives in the current best twisted-multi-ring micro-closure point.

## Main diagnostics

- residual rel Frobenius norm (keep 4 modes): {payload['residual_summary']['residual_rel_fro_keep4']:.4f}
- omitted lambda fraction: {payload['residual_summary']['omitted_lambda_fraction']:.4f}
- published projection13 error mean: {payload['consistency']['published_projection13_err_mean']:.4f}

## Flags

- tail dominated non-closure: {flags['tail_dominated_nonclosure']}
- inter-ring blocks dominate: {flags['inter_ring_blocks_dominate']}
- source bank misses low3 for some sources: {flags['source_bank_misses_low3_for_some_sources']}

## Best low3-capturing sources

{chr(10).join(f'- {name}: low3={entry["frac_low3"]:.3f}, tail={entry["frac_tail"]:.3f}' for name, entry in best_sources)}

## Worst tail-dominated sources

{chr(10).join(f'- {name}: low3={entry["frac_low3"]:.3f}, tail={entry["frac_tail"]:.3f}' for name, entry in worst_sources)}
"""
    summary = f"""# Multiring residual non-closure summary

Data source:
- `{DATA_DIR / 'multiring_residual_diagnostic_summary.json'}`

Headline:
- omitted lambda fraction = {payload['residual_summary']['omitted_lambda_fraction']:.3f}
- inter-ring blocks dominate = {payload['interpretation_flags']['inter_ring_blocks_dominate']}
- source-bank issue present = {payload['interpretation_flags']['source_bank_misses_low3_for_some_sources']}
"""
    (NOTES_DIR / 'multiring_residual_nonclosure_workbook.md').write_text(workbook, encoding='utf-8')
    (NOTES_DIR / 'multiring_residual_nonclosure_summary.md').write_text(summary, encoding='utf-8')


def main() -> None:
    payload = diagnose_residual()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / 'multiring_residual_diagnostic_summary.json'
    out.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    write_notes(payload)
    print(f'[INFO] Wrote: {out}')
    print(f"[INFO] Wrote: {NOTES_DIR / 'multiring_residual_nonclosure_workbook.md'}")
    print(f"[INFO] Wrote: {NOTES_DIR / 'multiring_residual_nonclosure_summary.md'}")


if __name__ == '__main__':
    main()
