#!/usr/bin/env python3
"""Run additive topology scan on fixed twisted-multi-ring geometry."""

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


def write_notes(payload: dict[str, object]) -> None:
    best = payload['results'][0]
    workbook = f"""# Multiring topology workbook

## 1. Motivation

This scan keeps the local twisted-multi-ring geometry fixed and varies only the topology of inter-ring couplings.
The question is whether the remaining gap in flavour / Phi1->Phi2 closure is topological rather than geometric.

## 2. Fixed geometry, variable topology

Fixed local support:
- twisted_multi_ring

Topologies tested:
- nearest-neighbor loop
- fully connected dual-channel
- alternating handedness
- phase/holonomy-like coupling

## 3. PMNS sensitivity

Best topology by joint score:
- {best['topology']} ({best['label']})

Baseline flavour index:
- {payload['baseline']['flavour_index']:.3f}

Best topology flavour index:
- {best['flavour_index']:.3f}

## 4. Phi2 emergence sensitivity

Baseline bridge index:
- {payload['baseline']['bridge_index']:.3f}

Best topology bridge index:
- {best['bridge_index']:.3f}

## 5. Micro-closure sensitivity

Baseline closure status:
- {payload['baseline']['micro_closure_status']}

Best topology closure status:
- {best['micro_closure_status']}

## 6. Joint interpretation

Final classification:
- {payload['final_classification']}

## 7. Honest conclusion

At this stage, the scan indicates whether the missing ingredient is more likely topological than geometric without rewriting the current twisted_multi_ring baseline.
"""

    summary = f"""# Multiring topology summary

Data source:
- `{DATA_DIR / 'multiring_topology_scan_summary.json'}`

Best topology:
- `{best['topology']}` ({best['label']})

Final classification:
- `{payload['final_classification']}`

Baseline vs best:
- flavour index: {payload['baseline']['flavour_index']:.3f} -> {best['flavour_index']:.3f}
- bridge index: {payload['baseline']['bridge_index']:.3f} -> {best['bridge_index']:.3f}
- closure status: {payload['baseline']['micro_closure_status']} -> {best['micro_closure_status']}
"""
    (NOTES_DIR / 'multiring_topology_workbook.md').write_text(workbook, encoding='utf-8')
    (NOTES_DIR / 'multiring_topology_summary.md').write_text(summary, encoding='utf-8')


def main() -> None:
    payload = scan_topologies()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / 'multiring_topology_scan_summary.json'
    out.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    write_notes(payload)
    print(f'[INFO] Wrote: {out}')
    print(f"[INFO] Wrote: {NOTES_DIR / 'multiring_topology_workbook.md'}")
    print(f"[INFO] Wrote: {NOTES_DIR / 'multiring_topology_summary.md'}")


if __name__ == '__main__':
    main()
