#!/usr/bin/env python3
"""Plots for v2.2 analytic/canonical/homogeneous diagnostics."""

from __future__ import annotations

import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
mpl_dir = REPO_ROOT / ".tmp_mpl"
mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    data_dir = REPO_ROOT / "papers" / "paper-08" / "data"
    figs_dir = REPO_ROOT / "papers" / "paper-08" / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    a = json.loads((data_dir / "phi1_phi2_emergence_v22_analytic_summary.json").read_text())
    c = json.loads((data_dir / "phi1_phi2_emergence_v22_canonical_summary.json").read_text())
    h = json.loads((data_dir / "phi1_phi2_emergence_v22_homogeneous_summary.json").read_text())
    r = json.loads((data_dir / "phi1_phi2_emergence_v22_readiness.json").read_text())

    # Figure 1: local linear coefficients.
    feat = a["features"]
    bj = np.array(a["coeff_joint"], dtype=float)
    br = np.array(a["coeff_robust"], dtype=float)
    x = np.arange(len(feat))
    w = 0.36
    plt.figure(figsize=(7.8, 4.8))
    plt.bar(x - w / 2, bj, width=w, label="joint coeff")
    plt.bar(x + w / 2, br, width=w, label="robust coeff")
    plt.axhline(0.0, color="k", lw=1.0)
    plt.xticks(x, feat, rotation=20, ha="right")
    plt.title(f"v2.2 local analytic coefficients (R2 joint={a['r2_joint']:.2f}, robust={a['r2_robust']:.2f})")
    plt.grid(axis="y", ls=":", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    f1 = figs_dir / "phi1_phi2_v22_local_coeffs.png"
    plt.savefig(f1, dpi=180, bbox_inches="tight")
    plt.close()
    print("[INFO] Wrote:", f1)

    # Figure 2: canonical reduction metrics.
    m = c["metrics"]
    labels = [
        "first_mode_weight",
        "orth_err_p90",
        "canon_err_p90",
    ]
    vals = [
        m["first_mode_weight_mean"],
        m["orthogonality_error_max_p90"],
        m["canonical_norm_error_p90"],
    ]
    plt.figure(figsize=(7.0, 4.6))
    plt.bar(np.arange(len(labels)), vals)
    plt.xticks(np.arange(len(labels)), labels, rotation=20, ha="right")
    plt.yscale("log")
    plt.title("v2.2 canonical reduction diagnostics")
    plt.grid(axis="y", ls=":", alpha=0.4)
    plt.tight_layout()
    f2 = figs_dir / "phi1_phi2_v22_canonical_metrics.png"
    plt.savefig(f2, dpi=180, bbox_inches="tight")
    plt.close()
    print("[INFO] Wrote:", f2)

    # Figure 3: strict homogeneous behavior.
    s = h["series_thin"]
    a_arr = np.array(s["a"], dtype=float)
    rho = np.array(s["rho_norm"], dtype=float)
    rhoa3 = np.array(s["rho_a3_norm"], dtype=float)
    mu = np.array(s["mu_m_over_H"], dtype=float)

    fig, ax = plt.subplots(1, 2, figsize=(10.8, 4.6))
    ax[0].loglog(a_arr, rho, label=r"$\rho$")
    ax[0].loglog(a_arr, rhoa3 / np.max(rhoa3), label=r"$\rho a^3$ (norm)")
    ax[0].set_xlabel("a")
    ax[0].set_title("Homogeneous evolution")
    ax[0].grid(which="both", ls=":", alpha=0.4)
    ax[0].legend()

    ax[1].semilogx(a_arr, mu)
    ax[1].axhline(20.0, ls="--", lw=1.0, color="tab:red")
    ax[1].set_xlabel("a")
    ax[1].set_ylabel("m/H")
    ax[1].set_title("Oscillatory entry condition")
    ax[1].grid(ls=":", alpha=0.4)
    fig.tight_layout()
    f3 = figs_dir / "phi1_phi2_v22_homogeneous_strict.png"
    fig.savefig(f3, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("[INFO] Wrote:", f3)

    out = data_dir / "phi1_phi2_emergence_v22_plot_meta.json"
    out.write_text(json.dumps({"readiness_status": r["status"], "checks": r["checks"]}, indent=2), encoding="utf-8")
    print("[INFO] Wrote:", out)


if __name__ == "__main__":
    main()
