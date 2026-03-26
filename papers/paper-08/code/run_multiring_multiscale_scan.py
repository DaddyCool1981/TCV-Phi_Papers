#!/usr/bin/env python3
"""Run multiscale coarse-graining scan on fixed twisted-multi-ring geometry."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.lib.tcvphi.multiring_multiscale_scan import scan_multiscale

DATA_DIR = ROOT / 'papers' / 'paper-08' / 'data'
NOTES_DIR = ROOT / 'notes'


def write_notes(payload: dict[str, object]) -> None:
    best = payload['results'][0]
    workbook = f"""# Multiring multiscale workbook

## What this means

Here, multiscale/coarse-graining means that the same local twisted-multi-ring structure is viewed at more than one interaction scale:
- local site-to-site links,
- mesoscopic block-to-block links,
- and mild feedback between these levels.

The idea is to test whether the missing ingredient is not a new shape or a new topology, but a scale-dependent organization of the same structure.

## Outcome

Best multiscale scheme:
- {best['scheme']} ({best['label']})

Final classification:
- {payload['final_classification']}

Baseline vs best:
- flavour index: {payload['baseline']['flavour_index']:.3f} -> {best['flavour_index']:.3f}
- bridge index: {payload['baseline']['bridge_index']:.3f} -> {best['bridge_index']:.3f}
- closure status: {payload['baseline']['micro_closure_status']} -> {best['micro_closure_status']}
"""
    summary = f"""# Multiring multiscale summary

Data source:
- `{DATA_DIR / 'multiring_multiscale_scan_summary.json'}`

Best scheme:
- `{best['scheme']}` ({best['label']})

Final classification:
- `{payload['final_classification']}`
"""
    (NOTES_DIR / 'multiring_multiscale_workbook.md').write_text(workbook, encoding='utf-8')
    (NOTES_DIR / 'multiring_multiscale_summary.md').write_text(summary, encoding='utf-8')


def main() -> None:
    payload = scan_multiscale()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / 'multiring_multiscale_scan_summary.json'
    out.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    write_notes(payload)
    print(f'[INFO] Wrote: {out}')
    print(f"[INFO] Wrote: {NOTES_DIR / 'multiring_multiscale_workbook.md'}")
    print(f"[INFO] Wrote: {NOTES_DIR / 'multiring_multiscale_summary.md'}")


if __name__ == '__main__':
    main()
