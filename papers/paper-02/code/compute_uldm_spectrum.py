#!/usr/bin/env python3
"""
Paper II ULDM diagnostics:
  - solve homogeneous Phi2 background
  - verify late-time rho_phi2 ~ a^-3 scaling
  - compute effective ULDM-vs-CDM matter spectra with CLASS wrapper
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
CORE_LIB = REPO_ROOT / "core" / "lib"
if str(CORE_LIB) not in sys.path:
    sys.path.insert(0, str(CORE_LIB))

from tcvphi.uldm import (
    ULDMBackgroundParams,
    get_tcv_canonical_params_uldm,
    solve_phi2_background,
    compute_pk_uldm_effective,
)


def main() -> None:
    out_figs = REPO_ROOT / "papers" / "paper-02" / "figs"
    out_data = REPO_ROOT / "papers" / "paper-02" / "data"
    out_figs.mkdir(parents=True, exist_ok=True)
    out_data.mkdir(parents=True, exist_ok=True)

    theta = get_tcv_canonical_params_uldm()
    bg = solve_phi2_background(
        params=ULDMBackgroundParams(a_ini=1.0e-6, a_end=1.0, n_steps=2600, phi2_ini=1.1e17),
        theta_tcv=theta,
    )

    k_vals = np.logspace(-3, 0.7, 350)
    pk_cdm = compute_pk_uldm_effective(theta_tcv=theta, k_vals=k_vals, z=0.0)
    pk_uldm = compute_pk_uldm_effective(
        theta_tcv=theta,
        k_vals=k_vals,
        z=0.0,
        Omega_phi2_h2=theta["omch2"],  # same benchmark density, ULDM-as-effective-CDM mode
    )

    ratio = pk_uldm["pk"] / np.maximum(pk_cdm["pk"], 1.0e-300)

    np.savez(
        out_data / "uldm_background.npz",
        a=bg["a"],
        N=bg["N"],
        H_gev=bg["H_gev"],
        phi2=bg["phi2"],
        dphi2_dN=bg["dphi2_dN"],
        rho_phi2=bg["rho_phi2"],
        h_over_m=bg["h_over_m"],
    )
    np.savez(
        out_data / "uldm_pk_comparison.npz",
        k=pk_cdm["k"],
        pk_cdm=pk_cdm["pk"],
        pk_uldm=pk_uldm["pk"],
        ratio=ratio,
    )

    summary = {
        "m2_GeV": float(bg["m2_GeV"]),
        "rho_scaling_slope_late": float(bg["scaling_slope_late"]),
        "expected_slope": -3.0,
        "pk_ratio_min": float(np.min(ratio)),
        "pk_ratio_max": float(np.max(ratio)),
        "note": "Current ULDM P(k) mode is effective-CDM placeholder; full perturbations are deferred.",
    }
    with (out_data / "uldm_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Figure 1: homogeneous background
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    axes[0].loglog(bg["a"], np.abs(bg["phi2"]), lw=1.6, label=r"$|\Phi_2(a)|$")
    axes[0].set_ylabel(r"$|\Phi_2|$ [GeV]")
    axes[0].grid(True, which="both", ls=":")
    axes[0].legend(loc="best")
    axes[0].set_title("ULDM homogeneous evolution")

    axes[1].loglog(bg["a"], np.maximum(bg["rho_phi2"], 1.0e-300), lw=1.8, label=r"$\rho_{\Phi_2}$")
    rho_ref = np.maximum(bg["rho_phi2"][0] * (bg["a"] / bg["a"][0]) ** (-3.0), 1.0e-300)
    axes[1].loglog(bg["a"], rho_ref, "--", lw=1.2, label=r"$a^{-3}$ reference")
    axes[1].set_xlabel(r"$a$")
    axes[1].set_ylabel(r"$\rho_{\Phi_2}$ [GeV$^4$]")
    axes[1].grid(True, which="both", ls=":")
    axes[1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_figs / "uldm_background_scaling.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: P(k) comparison
    plt.figure(figsize=(8, 6))
    plt.loglog(pk_cdm["k"], pk_cdm["pk"], lw=1.8, label="CDM reference (CLASS)")
    plt.loglog(pk_uldm["k"], pk_uldm["pk"], lw=1.8, label="ULDM effective mode (CLASS)")
    plt.xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
    plt.ylabel(r"$P(k)$")
    plt.title("ULDM effective P(k) vs CDM")
    plt.grid(True, which="both", ls=":")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_figs / "uldm_pk_comparison.png", dpi=180, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 4.8))
    plt.semilogx(pk_cdm["k"], ratio, lw=1.8)
    plt.axhline(1.0, ls="--", color="k", alpha=0.6)
    plt.xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
    plt.ylabel(r"$P_{\rm ULDM}/P_{\rm CDM}$")
    plt.title("ULDM/CDM spectrum ratio (effective mode)")
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.savefig(out_figs / "uldm_pk_ratio.png", dpi=180, bbox_inches="tight")
    plt.close()

    print("[INFO] Wrote:", out_data / "uldm_background.npz")
    print("[INFO] Wrote:", out_data / "uldm_pk_comparison.npz")
    print("[INFO] Wrote:", out_data / "uldm_summary.json")
    print("[INFO] Wrote:", out_figs / "uldm_background_scaling.png")
    print("[INFO] Wrote:", out_figs / "uldm_pk_comparison.png")
    print("[INFO] Wrote:", out_figs / "uldm_pk_ratio.png")
    print("[INFO] Late-time slope d ln rho / d ln a =", f"{summary['rho_scaling_slope_late']:.4f}")


if __name__ == "__main__":
    main()
