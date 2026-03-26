#!/usr/bin/env python3
"""Run targeted V2 topology scan on fixed twisted-multi-ring geometry."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.lib.tcvphi.multiring_topology_scan import scan_topologies

DATA_DIR = ROOT / 'papers' / 'paper-08' / 'data'
NOTES_DIR = ROOT / 'notes'
KEYS = [
    'nearest_neighbor_loop',
    'alternating_handedness',
    'hierarchical_loop',
    'frustrated_dual_closure',
]


def write_notes(payload: dict[str, object]) -> None:
    best = payload['results'][0]
    lines = [
        '# Multiring topology V2 workbook',
        '',
        '## Motivation',
        '',
        '- V2 focuses only on targeted topological ingredients: hierarchy and closure frustration.',
        '- Local twisted-multi-ring geometry remains fixed.',
        '',
        '## Outcome',
        '',
        f"- Best topology: {best['topology']} ({best['label']})",
        f"- Final classification: {payload['final_classification']}",
        f"- Baseline flavour index: {payload['baseline']['flavour_index']:.3f}",
        f"- Best V2 flavour index: {best['flavour_index']:.3f}",
        f"- Baseline bridge index: {payload['baseline']['bridge_index']:.3f}",
        f"- Best V2 bridge index: {best['bridge_index']:.3f}",
        f"- Baseline closure status: {payload['baseline']['micro_closure_status']}",
        f"- Best V2 closure status: {best['micro_closure_status']}",
        '',
        '## Interpretation',
        '',
        '- This V2 scan asks whether a more structured coupling topology helps where the first simple topology scan did not.',
        '- The result should be read as refinement, not replacement, of the twisted-multi-ring baseline.',
    ]
    (NOTES_DIR / 'multiring_topology_v2_workbook.md').write_text('\n'.join(lines), encoding='utf-8')
    (NOTES_DIR / 'multiring_topology_v2_summary.md').write_text(
        '\n'.join([
            '# Multiring topology V2 summary',
            '',
            f"- Best topology: {best['topology']} ({best['label']})",
            f"- Final classification: {payload['final_classification']}",
            f"- Baseline vs best flavour: {payload['baseline']['flavour_index']:.3f} -> {best['flavour_index']:.3f}",
            f"- Baseline vs best bridge: {payload['baseline']['bridge_index']:.3f} -> {best['bridge_index']:.3f}",
            f"- Baseline vs best closure: {payload['baseline']['micro_closure_status']} -> {best['micro_closure_status']}",
        ]),
        encoding='utf-8',
    )


def main() -> None:
    payload = scan_topologies(n_flavour=360, n_bridge=160, seed=20260316, selected_keys=KEYS)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / 'multiring_topology_scan_v2_summary.json'
    out.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    write_notes(payload)
    print(f'[INFO] Wrote: {out}')
    print(f"[INFO] Wrote: {NOTES_DIR / 'multiring_topology_v2_workbook.md'}")
    print(f"[INFO] Wrote: {NOTES_DIR / 'multiring_topology_v2_summary.md'}")


if __name__ == '__main__':
    main()
