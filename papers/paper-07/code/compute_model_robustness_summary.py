#!/usr/bin/env python3
"""Compute lightweight robustness diagnostics for Paper-07 model comparison."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]


def _last_progress_row(path: Path) -> dict:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 5:
            continue
        rows.append(
            {
                "N": float(parts[0]),
                "timestamp": parts[1],
                "acceptance_rate": float(parts[2]),
                "Rminus1": float(parts[3]),
                "Rminus1_cl": None if parts[4] == "NaN" else float(parts[4]),
            }
        )
    if not rows:
        raise RuntimeError(f"No usable progress rows in {path}")
    return rows[-1]


def main() -> None:
    data_dir = REPO_ROOT / "papers" / "paper-07" / "data"
    cmp_path = data_dir / "model_comparison_lcdm_vs_tcv_minimal.json"
    cmp_data = json.loads(cmp_path.read_text(encoding="utf-8"))

    dchi2 = float(cmp_data["criteria"]["delta_chi2_tcv_minus_lcdm"])
    # k_tcv-k_lcdm = 5-7 = -2 in current setup
    delta_k = -2.0

    n_grid = [1500, 2000, 2500, 3000, 4000]
    bic_scan = []
    for n_data in n_grid:
        delta_bic = dchi2 + delta_k * np.log(float(n_data))
        bic_scan.append(
            {
                "n_data_proxy": int(n_data),
                "delta_bic_tcv_minus_lcdm": float(delta_bic),
                "favors_tcv": bool(delta_bic < 0.0),
            }
        )

    lcdm_prog = _last_progress_row(data_dir / "lcdm_planck_chain_classnative_v7_lowl.progress")
    tcv_prog = _last_progress_row(data_dir / "tcv_minimal_planck_chain_v1_lowl.progress")

    out = {
        "delta_chi2_tcv_minus_lcdm": dchi2,
        "delta_k_tcv_minus_lcdm": delta_k,
        "bic_proxy_scan": bic_scan,
        "convergence_last_rows": {
            "lcdm_v7_lowl": lcdm_prog,
            "tcv_minimal_v1_lowl": tcv_prog,
        },
        "note": "BIC scan uses explicit n_data proxy values; convergence rows are read from chain .progress files.",
    }

    out_path = data_dir / "model_comparison_robustness.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("[INFO] Wrote:", out_path)


if __name__ == "__main__":
    main()

