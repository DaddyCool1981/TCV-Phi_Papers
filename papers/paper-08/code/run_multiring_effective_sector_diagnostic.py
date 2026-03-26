#!/usr/bin/env python3
"""Run effective-sector closure diagnostic for twisted-multi-ring."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.lib.tcvphi.multiring_effective_sector_diagnostics import effective_sector_diagnostic

DATA_DIR = ROOT / 'papers' / 'paper-08' / 'data'
NOTES_DIR = ROOT / 'notes'


def write_notes(payload: dict[str, object]) -> None:
    bc = payload['best_coherent_row']
    bb = payload['best_broad_row']
    workbook = f"""# Multiring effective-sector closure workbook

## Purpose

This diagnostic asks whether closure improves when the effective subspace is enlarged and when the claim is restricted to coherent source sectors instead of the full operator.

## Verdict

- {payload['verdict']}

## Best coherent-sector row

- n_keep = {bc['n_keep']}
- coherent mean error = {bc['coherent_rel_err_mean']:.4f}
- local mean error = {bc['local_rel_err_mean']:.4f}
- contrast mean error = {bc['contrast_rel_err_mean']:.4f}
- twist-pair mean error = {bc['twistpair_rel_err_mean']:.4f}

## Best broad-sector row

- n_keep = {bb['n_keep']}
- coherent mean error = {bb['coherent_rel_err_mean']:.4f}
- local mean error = {bb['local_rel_err_mean']:.4f}
- contrast mean error = {bb['contrast_rel_err_mean']:.4f}
- twist-pair mean error = {bb['twistpair_rel_err_mean']:.4f}
"""
    summary = f"""# Multiring effective-sector closure summary

Data source:
- `{DATA_DIR / 'multiring_effective_sector_diagnostic_summary.json'}`

Verdict:
- `{payload['verdict']}`

Best coherent-sector closure:
- n_keep={bc['n_keep']}, err={bc['coherent_rel_err_mean']:.4f}

Best broad-sector closure:
- n_keep={bb['n_keep']}, err={bb['local_rel_err_mean']:.4f}/{bb['twistpair_rel_err_mean']:.4f}
"""
    (NOTES_DIR / 'multiring_effective_sector_closure_workbook.md').write_text(workbook, encoding='utf-8')
    (NOTES_DIR / 'multiring_effective_sector_closure_summary.md').write_text(summary, encoding='utf-8')


def main() -> None:
    payload = effective_sector_diagnostic()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / 'multiring_effective_sector_diagnostic_summary.json'
    out.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    write_notes(payload)
    print(f'[INFO] Wrote: {out}')
    print(f"[INFO] Wrote: {NOTES_DIR / 'multiring_effective_sector_closure_workbook.md'}")
    print(f"[INFO] Wrote: {NOTES_DIR / 'multiring_effective_sector_closure_summary.md'}")


if __name__ == '__main__':
    main()
