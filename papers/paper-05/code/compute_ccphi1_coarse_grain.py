#!/usr/bin/env python3
"""Coarse-graining diagnostics for CC-Phi1 toy microstructure."""

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

from tcvphi.ccphi1_micro import CCPhi1Params, coherence_length, mode_frequencies


def main() -> None:
    out_data = REPO_ROOT / "papers" / "paper-05" / "data"
    out_figs = REPO_ROOT / "papers" / "paper-05" / "figs"
    out_data.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    params = CCPhi1Params(m1=1.0e-8, xi0=0.1, n_modes=3)
    R_vals = np.linspace(0.0, 2.0e-15, 80)

    n_cc = params.m1**3
    omega1_vals = np.array([mode_frequencies(R, params)[0] for R in R_vals])
    Lc_vals = np.array([coherence_length(R, params) for R in R_vals])
    rho_cc_eff = n_cc * omega1_vals
    w_proxy = -0.5
    pressure_proxy = w_proxy * rho_cc_eff

    out_json = out_data / "ccphi1_coarse_grain.json"
    payload = {
        "n_cc_proxy": float(n_cc),
        "R_window": [float(R_vals.min()), float(R_vals.max())],
        "rho_cc_eff_minmax": [float(rho_cc_eff.min()), float(rho_cc_eff.max())],
        "pressure_proxy_minmax": [float(pressure_proxy.min()), float(pressure_proxy.max())],
        "w_proxy": float(w_proxy),
        "note": (
            "Toy coarse-graining proxy (not a full QFT derivation). "
            "w_proxy is an illustrative closure parameter."
        ),
    }
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print("[INFO] Wrote:", out_json)

    fig, ax = plt.subplots(1, 2, figsize=(10.8, 4.2))
    ax[0].plot(R_vals, rho_cc_eff, lw=1.6, label=r"$\rho_{\rm cc,eff}$")
    ax[0].plot(R_vals, np.abs(pressure_proxy), lw=1.2, ls="--", label=r"$|p_{\rm proxy}|$")
    ax[0].set_xlabel(r"$R$")
    ax[0].set_yscale("log")
    ax[0].set_title("Coarse-grained energy/pressure proxies")
    ax[0].grid(True, which="both", ls=":")
    ax[0].legend()

    ax[1].plot(R_vals, Lc_vals, lw=1.6, color="C3")
    ax[1].set_xlabel(r"$R$")
    ax[1].set_ylabel(r"$L_c$")
    ax[1].set_title("Cluster coherence length")
    ax[1].grid(True, ls=":")

    fig.tight_layout()
    out_fig = out_figs / "ccphi1_coarse_grain.png"
    fig.savefig(out_fig, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("[INFO] Wrote:", out_fig)


if __name__ == "__main__":
    main()
