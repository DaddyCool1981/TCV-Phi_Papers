#!/usr/bin/env python3
"""v2.2: analytic rationale + canonical EFT reduction + stricter homogeneous test."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
mpl_dir = REPO_ROOT / ".tmp_mpl"
mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))

CORE_LIB = REPO_ROOT / "core" / "lib"
if str(CORE_LIB) not in sys.path:
    sys.path.insert(0, str(CORE_LIB))

from tcvphi.phi1_emergence import NetworkConfig  # noqa: E402
from tcvphi.phi1_emergence_diagnostics import (  # noqa: E402
    canonical_reduction_metrics,
    run_strict_homogeneous_uldm_test,
)


def _local_linear_model(v21: dict) -> dict:
    records = v21["records"]
    best = v21["best_record"]["params"]

    # Keep local neighborhood around best point for interpretable coefficients.
    near = []
    for r in records:
        p = r["params"]
        if abs(p["edge_aniso"] - best["edge_aniso"]) <= 0.02 and abs(p["drop_prob"] - best["drop_prob"]) <= 0.01:
            near.append(r)
    if len(near) < 6:
        near = records

    x = []
    y_joint = []
    y_rob = []
    for r in near:
        p = r["params"]
        x.append([1.0, p["edge_aniso"], p["drop_prob"], p["onsite_noise"]])
        y_joint.append(r["twisted_joint_mean"])
        y_rob.append(r["twisted_robust_mean"])

    X = np.array(x, dtype=float)
    yj = np.array(y_joint, dtype=float)
    yr = np.array(y_rob, dtype=float)

    bj, *_ = np.linalg.lstsq(X, yj, rcond=None)
    br, *_ = np.linalg.lstsq(X, yr, rcond=None)

    pred_j = X @ bj
    pred_r = X @ br
    r2j = 1.0 - float(np.sum((yj - pred_j) ** 2) / max(np.sum((yj - np.mean(yj)) ** 2), 1.0e-14))
    r2r = 1.0 - float(np.sum((yr - pred_r) ** 2) / max(np.sum((yr - np.mean(yr)) ** 2), 1.0e-14))

    return {
        "best_point": best,
        "n_local_points": int(len(near)),
        "model": "local_linear",
        "features": ["1", "edge_aniso", "drop_prob", "onsite_noise"],
        "coeff_joint": bj.tolist(),
        "coeff_robust": br.tolist(),
        "r2_joint": r2j,
        "r2_robust": r2r,
        "interpretation": {
            "joint": "negative coeff => increasing parameter tends to reduce joint score locally",
            "robust": "negative coeff => increasing parameter tends to reduce robustness locally",
        },
    }


def main() -> None:
    out_data = REPO_ROOT / "papers" / "paper-08" / "data"
    out_data.mkdir(parents=True, exist_ok=True)

    v21_path = out_data / "phi1_phi2_emergence_v21_summary.json"
    if not v21_path.exists():
        raise FileNotFoundError("Run v2.1 first: missing phi1_phi2_emergence_v21_summary.json")
    v21 = json.loads(v21_path.read_text())

    analytic = _local_linear_model(v21)
    out_a = out_data / "phi1_phi2_emergence_v22_analytic_summary.json"
    out_a.write_text(json.dumps(analytic, indent=2), encoding="utf-8")
    print("[INFO] Wrote:", out_a)

    bp = v21["best_record"]["params"]
    cfg = NetworkConfig(
        family="twisted_ring",
        n=128,
        edge_aniso=float(bp["edge_aniso"]),
        drop_prob=float(bp["drop_prob"]),
        onsite_noise=float(bp["onsite_noise"]),
        k_link=1.0,
        eps_pin=1.0e-4,
    )
    canon = canonical_reduction_metrics(cfg=cfg, seed=20260390, n_samples=96, inertia_eta=0.20)
    out_c = out_data / "phi1_phi2_emergence_v22_canonical_summary.json"
    out_c.write_text(json.dumps(canon, indent=2), encoding="utf-8")
    print("[INFO] Wrote:", out_c)

    homo = run_strict_homogeneous_uldm_test(
        ln_a_min=-18.0,
        ln_a_max=0.0,
        n_steps=7000,
        m_over_h0=500.0,
    )

    # Save compact diagnostics JSON + thin series for plotting.
    thin = max(1, len(homo["series"]["a"]) // 700)
    homo_thin = {
        "a": homo["series"]["a"][::thin],
        "rho_norm": homo["series"]["rho_norm"][::thin],
        "rho_a3_norm": homo["series"]["rho_a3_norm"][::thin],
        "mu_m_over_H": homo["series"]["mu_m_over_H"][::thin],
    }
    out_h = out_data / "phi1_phi2_emergence_v22_homogeneous_summary.json"
    out_h.write_text(
        json.dumps(
            {
                "settings": homo["settings"],
                "late_window_criteria": homo["late_window_criteria"],
                "diagnostics": homo["diagnostics"],
                "pass_flags": homo["pass_flags"],
                "series_thin": homo_thin,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("[INFO] Wrote:", out_h)

    readiness = {
        "status": "B_promising_but_incomplete",
        "checks": {
            "analytic_local_model_ready": bool(analytic["n_local_points"] >= 6),
            "canonical_reduction_ready": bool(
                canon["pass_flags"]["orthogonality_good"] and canon["pass_flags"]["canonical_norm_good"]
            ),
            "canonical_mode_capture_good": bool(canon["pass_flags"]["mode_capture_good"]),
            "strict_homogeneous_ready": bool(
                homo["pass_flags"]["slope_close_to_minus3"] and homo["pass_flags"]["rho_a3_quasi_constant"]
            ),
        },
        "note": "One-scalar route is technically strengthened; first-principles mass naturalness still open.",
    }
    if all(readiness["checks"].values()):
        readiness["status"] = "B_plus_strong_internal_consistency"

    out_r = out_data / "phi1_phi2_emergence_v22_readiness.json"
    out_r.write_text(json.dumps(readiness, indent=2), encoding="utf-8")
    print("[INFO] Wrote:", out_r)


if __name__ == "__main__":
    main()
