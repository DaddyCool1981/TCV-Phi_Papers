#!/usr/bin/env python3
"""Aggregate key diagnostics from Papers III-V into a comparison JSON."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {"missing": True, "path": str(path)}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    out_data = REPO_ROOT / "papers" / "paper-06" / "data"
    out_data.mkdir(parents=True, exist_ok=True)

    p3_obs = _load_json(REPO_ROOT / "papers" / "paper-03" / "data" / "primordial_observables.json")
    p4_dir = _load_json(REPO_ROOT / "papers" / "paper-04" / "data" / "bh_direction_scan_table.json")
    p4_kerr = _load_json(REPO_ROOT / "papers" / "paper-04" / "data" / "bh_kerr_observables.json")
    p5_qg = _load_json(REPO_ROOT / "papers" / "paper-05" / "data" / "paper05_consistency_summary.json")

    comparison = {
        "paper03_primordial": {
            "n_s_local_fit": p3_obs.get("n_s_from_local_fit"),
            "alpha_s_quadratic": p3_obs.get("alpha_s_from_quadratic_fit"),
            "status": "phenomenological bridge to Boltzmann tools",
            "source": str(REPO_ROOT / "papers" / "paper-03" / "data" / "primordial_observables.json"),
        },
        "paper04_bh_branching": {
            "direction_scan_rows": p4_dir,
            "status": "numerical diagnostic with explicit branch dependence",
            "source": str(REPO_ROOT / "papers" / "paper-04" / "data" / "bh_direction_scan_table.json"),
        },
        "paper04_bh_kerr_template": {
            "photon_ring_shift": p4_kerr.get("photon_ring_relative_shift"),
            "Mf_echo_template": p4_kerr.get("f_echo_template"),
            "status": "template-level slow-rotation diagnostics",
            "source": str(REPO_ROOT / "papers" / "paper-04" / "data" / "bh_kerr_observables.json"),
        },
        "paper05_micro_qg": {
            "summary": p5_qg,
            "status": "CC-Phi1 toy microstructure + exploratory flavour bridge",
            "source": str(REPO_ROOT / "papers" / "paper-05" / "data" / "paper05_consistency_summary.json"),
        },
    }

    out_json = out_data / "series_comparison.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    print("[INFO] Wrote:", out_json)


if __name__ == "__main__":
    main()
