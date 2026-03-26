#!/usr/bin/env python3
"""Run dynamic-link refinement scan on fixed twisted-multi-ring geometry."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.lib.tcvphi.multiring_link_dynamics_scan import scan_dynamic_rules

DATA_DIR = ROOT / 'papers' / 'paper-08' / 'data'
NOTES_DIR = ROOT / 'notes'


def write_notes(payload: dict[str, object]) -> None:
    best = payload['results'][0]
    workbook = f"""# Multiring link dynamics workbook

## Motivation

This scan keeps the local geometry and the simple cyclic topology fixed, and changes only the dynamic rule assigned to inter-ring links.
The purpose is to test whether the missing ingredient is a link-response rule rather than a new geometry/topology.

## Outcome

Best dynamic rule:
- {best['rule']} ({best['label']})

Final classification:
- {payload['final_classification']}

Baseline vs best:
- flavour index: {payload['baseline']['flavour_index']:.3f} -> {best['flavour_index']:.3f}
- bridge index: {payload['baseline']['bridge_index']:.3f} -> {best['bridge_index']:.3f}
- closure status: {payload['baseline']['micro_closure_status']} -> {best['micro_closure_status']}
"""
    summary = f"""# Multiring link dynamics summary

Data source:
- `{DATA_DIR / 'multiring_link_dynamics_scan_summary.json'}`

Best rule:
- `{best['rule']}` ({best['label']})

Final classification:
- `{payload['final_classification']}`
"""
    (NOTES_DIR / 'multiring_link_dynamics_workbook.md').write_text(workbook, encoding='utf-8')
    (NOTES_DIR / 'multiring_link_dynamics_summary.md').write_text(summary, encoding='utf-8')


def main() -> None:
    payload = scan_dynamic_rules()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / 'multiring_link_dynamics_scan_summary.json'
    out.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    write_notes(payload)
    print(f'[INFO] Wrote: {out}')
    print(f"[INFO] Wrote: {NOTES_DIR / 'multiring_link_dynamics_workbook.md'}")
    print(f"[INFO] Wrote: {NOTES_DIR / 'multiring_link_dynamics_summary.md'}")


if __name__ == '__main__':
    main()
