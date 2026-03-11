#!/usr/bin/env python3
"""
Generate baseline Paper II bounce history outputs from core/lib/tcvphi/bounce.py.
Outputs:
  - papers/paper-02/figs/bounce_history_background.png
  - papers/paper-02/figs/bounce_history_energy_ratio.png
  - papers/paper-02/data/bounce_history.npz
  - papers/paper-02/data/bounce_summary.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Ensure core/lib is importable without changing Paper I public APIs.
REPO_ROOT = Path(__file__).resolve().parents[3]
CORE_LIB = REPO_ROOT / "core" / "lib"
if str(CORE_LIB) not in sys.path:
    sys.path.insert(0, str(CORE_LIB))

from tcvphi.bounce import BounceSimulationParams, simulate_cohesive_bounce


def main() -> None:
    out_figs = REPO_ROOT / "papers" / "paper-02" / "figs"
    out_data = REPO_ROOT / "papers" / "paper-02" / "data"
    out_figs.mkdir(parents=True, exist_ok=True)
    out_data.mkdir(parents=True, exist_ok=True)

    params = {
        "bounce": BounceSimulationParams(
            eta_min=-50.0,
            eta_max=50.0,
            n_eta=5000,
            a_b=1.0,
            eta0=5.0,
            reheating_eta_factor=2.0,
            reheating_efficiency=0.8,
        )
    }
    result = simulate_cohesive_bounce(params)

    eta = result["eta"]
    a = result["a"]
    H = result["H"]
    phi1 = result["phi1"]
    ratio = result["rho_phi1"] / np.maximum(result["rho_r"], 1.0e-120)
    summary = result["summary"]

    np.savez(
        out_data / "bounce_history.npz",
        eta=eta,
        a=a,
        H=H,
        phi1=phi1,
        dphi1_deta=result["dphi1_deta"],
        Meff2=result["Meff2"],
        rho_phi1=result["rho_phi1"],
        rho_r=result["rho_r"],
    )

    with (out_data / "bounce_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    axes[0].plot(eta, a, lw=1.8)
    axes[0].set_ylabel(r"$a(\eta)$")
    axes[0].grid(True, ls=":")
    axes[0].set_title("Cohesive bounce baseline history")

    axes[1].plot(eta, H, lw=1.8)
    axes[1].set_ylabel(r"$H(\eta)$")
    axes[1].grid(True, ls=":")

    axes[2].plot(eta, phi1, lw=1.8)
    axes[2].set_ylabel(r"$\Phi_1(\eta)$")
    axes[2].set_xlabel(r"$\eta$")
    axes[2].grid(True, ls=":")

    fig.tight_layout()
    fig.savefig(out_figs / "bounce_history_background.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    plt.figure(figsize=(8, 5))
    plt.semilogy(eta, np.maximum(ratio, 1.0e-40), lw=1.8)
    plt.xlabel(r"$\eta$")
    plt.ylabel(r"$\rho_{\Phi_1} / \rho_r$")
    plt.title("Structural-to-radiation energy ratio")
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.savefig(out_figs / "bounce_history_energy_ratio.png", dpi=180, bbox_inches="tight")
    plt.close()

    print("[INFO] Wrote:", out_data / "bounce_history.npz")
    print("[INFO] Wrote:", out_data / "bounce_summary.json")
    print("[INFO] Wrote:", out_figs / "bounce_history_background.png")
    print("[INFO] Wrote:", out_figs / "bounce_history_energy_ratio.png")
    print("[INFO] T_reh =", f"{summary['T_reh']:.6e}", "GeV")
    print("[INFO] Phi1_0 =", f"{summary['Phi1_0']:.6e}", "GeV")
    print("[INFO] dPhi1_0 =", f"{summary['dPhi1_0']:.6e}", "GeV^2")


if __name__ == "__main__":
    main()
