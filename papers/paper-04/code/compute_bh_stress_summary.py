#!/usr/bin/env python3
"""Build a compact stress-test summary for Paper-04 exclusion windows."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_json(path: Path):
    if not path.exists():
        return {"missing": True, "path": str(path)}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    out_data = REPO_ROOT / "papers" / "paper-04" / "data"
    out_data.mkdir(parents=True, exist_ok=True)

    direction_rows = _load_json(out_data / "bh_direction_scan_table.json")
    diagnostics = _load_json(out_data / "bh_diagnostics.json")
    kerr_obs = _load_json(out_data / "bh_kerr_observables.json")

    high_density = direction_rows[-1] if isinstance(direction_rows, list) and direction_rows else {}
    fmin_out = high_density.get("fmin_out")
    fmin_in = high_density.get("fmin_in")
    if isinstance(fmin_out, (int, float)) and isinstance(fmin_in, (int, float)):
        branch_delta = float(fmin_in - fmin_out)
        branch_ratio = float(fmin_in / max(fmin_out, 1.0e-30))
    else:
        branch_delta = None
        branch_ratio = None

    dense_diag = diagnostics.get("dense_core", {}) if isinstance(diagnostics, dict) else {}
    delta_f_asym = dense_diag.get("delta_f_asym_max")
    horizon_dense = dense_diag.get("horizon_radius")

    out_json = out_data / "bh_stress_summary.json"
    payload = {
        "high_density_direction_scan": {
            "rho_cc0": high_density.get("rho_cc0"),
            "fmin_out": fmin_out,
            "fmin_in": fmin_in,
            "branch_delta_fmin": branch_delta,
            "branch_ratio_fmin_in_over_out": branch_ratio,
            "out_success": high_density.get("out_success"),
            "in_success": high_density.get("in_success"),
        },
        "dense_core_static": {
            "delta_f_asym_max": delta_f_asym,
            "horizon_radius": horizon_dense,
        },
        "slow_rotation_template": {
            "photon_ring_relative_shift": kerr_obs.get("photon_ring_relative_shift"),
            "Mf_echo_template": kerr_obs.get("f_echo_template"),
            "omega_r_eikonal_template": kerr_obs.get("omega_r_eikonal_template"),
        },
        "note": "Compact stress-test summary for Paper-04 text/table insertion.",
    }

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print("[INFO] Wrote:", out_json)


if __name__ == "__main__":
    main()
