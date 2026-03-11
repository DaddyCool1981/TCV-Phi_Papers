#!/usr/bin/env python3
"""Curvature response scan for CC-Phi1 microstructure toy."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
mpl_dir = REPO_ROOT / ".tmp_mpl"
mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CORE_LIB = REPO_ROOT / "core" / "lib"
if str(CORE_LIB) not in sys.path:
    sys.path.insert(0, str(CORE_LIB))

from tcvphi.ccphi1_micro import CCPhi1Params, coherence_length, mode_energies


def main() -> None:
    out_data = REPO_ROOT / "papers" / "paper-05" / "data"
    out_figs = REPO_ROOT / "papers" / "paper-05" / "figs"
    out_data.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    params = CCPhi1Params(m1=1.0e-8, xi0=0.1, n_modes=3)
    R_vals = np.linspace(0.0, 2.0e-15, 120)

    L_vals = np.array([coherence_length(R, params) for R in R_vals])
    Xi_vals = np.array([mode_energies(R, params) for R in R_vals])  # (N, n_modes)
    Xi1 = Xi_vals[:, 0]
    Xi1_ratio = Xi1 / max(Xi1[0], 1.0e-30)
    L_ratio = L_vals / max(L_vals[0], 1.0e-30)

    out_npz = out_data / "ccphi1_curvature_scan.npz"
    np.savez(
        out_npz,
        R=R_vals,
        Lc=L_vals,
        Xi_modes=Xi_vals,
        Xi1_ratio=Xi1_ratio,
        L_ratio=L_ratio,
    )
    print("[INFO] Wrote:", out_npz)

    summary = {
        "R_min": float(R_vals.min()),
        "R_max": float(R_vals.max()),
        "Xi1_ratio_minmax": [float(Xi1_ratio.min()), float(Xi1_ratio.max())],
        "Lc_ratio_minmax": [float(L_ratio.min()), float(L_ratio.max())],
        "note": "Curvature range chosen so xi0*R remains comparable to m1^2 in the toy setup.",
    }
    out_json = out_data / "ccphi1_curvature_summary.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("[INFO] Wrote:", out_json)

    fig, ax = plt.subplots(1, 2, figsize=(10.8, 4.2))
    ax[0].plot(R_vals, Xi1_ratio, lw=1.6)
    ax[0].set_xlabel(r"$R$")
    ax[0].set_ylabel(r"$\Xi_1(R)/\Xi_1(0)$")
    ax[0].set_title("Lowest-mode eigenvalue response")
    ax[0].grid(True, ls=":")

    ax[1].plot(R_vals, L_ratio, lw=1.6, color="C3")
    ax[1].set_xlabel(r"$R$")
    ax[1].set_ylabel(r"$L_c(R)/L_c(0)$")
    ax[1].set_title("Coherence-length response")
    ax[1].grid(True, ls=":")

    fig.tight_layout()
    out_fig = out_figs / "ccphi1_curvature_response.png"
    fig.savefig(out_fig, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("[INFO] Wrote:", out_fig)


if __name__ == "__main__":
    main()
