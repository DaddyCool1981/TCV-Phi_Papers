#!/usr/bin/env python3
"""
Paper II baryogenesis diagnostics from the bounce history.

Outputs:
  - papers/paper-02/data/baryogenesis_summary.json
  - papers/paper-02/figs/baryogenesis_history.png
"""

from __future__ import annotations

import json
import sys
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
CORE_LIB = REPO_ROOT / "core" / "lib"
if str(CORE_LIB) not in sys.path:
    sys.path.insert(0, str(CORE_LIB))

from tcvphi.bounce import simulate_cohesive_bounce
from tcvphi.baryogenesis import (
    BaryogenesisParams,
    compute_etaB,
    lambda_B_for_target_TB,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Paper II baryogenesis diagnostics.")
    parser.add_argument(
        "--target-TB-frac",
        type=float,
        default=0.3,
        help="Target T_B as a fraction of max temperature in bounce history.",
    )
    args = parser.parse_args()

    out_figs = REPO_ROOT / "papers" / "paper-02" / "figs"
    out_data = REPO_ROOT / "papers" / "paper-02" / "data"
    out_figs.mkdir(parents=True, exist_ok=True)
    out_data.mkdir(parents=True, exist_ok=True)

    bounce_file = out_data / "bounce_history.npz"
    if bounce_file.exists():
        d = np.load(bounce_file)
        # Minimal history payload expected by compute_etaB.
        bounce_history = {
            "eta": d["eta"],
            "a": d["a"],
            "dphi1_deta": d["dphi1_deta"],
            "rho_r": d["rho_r"],
        }
    else:
        bounce_history = simulate_cohesive_bounce()

    bg_params = BaryogenesisParams()
    params = {"baryogenesis": bg_params}
    out_baseline = compute_etaB(params, bounce_history)

    T_hist = np.asarray(out_baseline["diagnostics"]["T_of_eta"])
    T_target = float(max(1.0e-30, args.target_TB_frac * np.max(T_hist)))
    lambda_cal = lambda_B_for_target_TB(
        T_target,
        M_pl=2.435e18,
        g_star=bg_params.g_star,
        alpha_B=bg_params.alpha_B,
        rate_power=bg_params.rate_power,
        lambda_power=bg_params.lambda_power,
    )
    out = compute_etaB(
        {
            "baryogenesis": bg_params,
            "Lambda_B_override": lambda_cal,
        },
        bounce_history,
    )
    diag = out["diagnostics"]
    idx_B = int(diag["idx_B"])

    summary = {
        "Lambda_B_baseline": out_baseline["Lambda_B"],
        "Lambda_B_calibrated": out["Lambda_B"],
        "T_B_baseline": out_baseline["T_B"],
        "T_B": out["T_B"],
        "eta_B": out["eta_B"],
        "eta_B_at_TB": diag["eta_B_at_TB"],
        "Gamma_over_H_at_TB": out["consistency"]["Gamma_over_H_at_TB"],
        "T_reh_over_TB": out["consistency"]["T_reh_over_TB"],
        "freezeout_inside_history": out["consistency"]["freezeout_inside_history"],
        "eta_at_TB": float(diag["eta"][idx_B]),
    }

    with (out_data / "baryogenesis_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    eta = np.asarray(diag["eta"])
    mu_over_T = np.asarray(diag["mu_B"]) / np.maximum(np.asarray(diag["T_of_eta"]), 1.0e-300)
    gamma_over_h = np.asarray(diag["Gamma_over_H"])
    gamma_over_h_base = np.asarray(out_baseline["diagnostics"]["Gamma_over_H"])
    eta_eq = np.asarray(diag["eta_eq"])

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(eta, mu_over_T, lw=1.6, label=r"$\mu_B/T$")
    axes[0].plot(eta, eta_eq, lw=1.6, label=r"$\eta_B^{\rm eq}$")
    axes[0].axvline(float(eta[idx_B]), ls="--", color="k", alpha=0.6)
    axes[0].set_ylabel("dimensionless")
    axes[0].set_title("Baryogenesis source terms from cohesive bounce history")
    axes[0].grid(True, ls=":")
    axes[0].legend(loc="best")

    axes[1].semilogy(eta, np.maximum(gamma_over_h_base, 1.0e-30), lw=1.2, label=r"$\Gamma_B/H$ (baseline)")
    axes[1].semilogy(eta, np.maximum(gamma_over_h, 1.0e-30), lw=1.8, label=r"$\Gamma_B/H$ (calibrated)")
    axes[1].axhline(1.0, ls="--", color="k", alpha=0.6)
    axes[1].axvline(float(eta[idx_B]), ls="--", color="k", alpha=0.6)
    axes[1].set_xlabel(r"$\eta$")
    axes[1].set_ylabel(r"$\Gamma_B/H$")
    axes[1].grid(True, which="both", ls=":")
    axes[1].legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_figs / "baryogenesis_history.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("[INFO] Wrote:", out_data / "baryogenesis_summary.json")
    print("[INFO] Wrote:", out_figs / "baryogenesis_history.png")
    print("[INFO] Lambda_B baseline   =", f"{summary['Lambda_B_baseline']:.6e}", "GeV")
    print("[INFO] Lambda_B calibrated =", f"{summary['Lambda_B_calibrated']:.6e}", "GeV")
    print("[INFO] T_B baseline =", f"{summary['T_B_baseline']:.6e}", "GeV")
    print("[INFO] T_B =", f"{summary['T_B']:.6e}", "GeV")
    print("[INFO] eta_B =", f"{summary['eta_B']:.6e}")


if __name__ == "__main__":
    main()
