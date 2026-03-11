#!/usr/bin/env python3
"""Compute Paper-04 diagnostics from stored static profile families."""

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


def _matching_residuals(r: np.ndarray, m: np.ndarray, f_metric: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m_inf = float(m[-1])
    with np.errstate(divide="ignore", invalid="ignore"):
        delta_m = np.abs(2.0 * (m - m_inf) / np.maximum(r, 1.0e-16))
        f_schw = 1.0 - 2.0 * m_inf / np.maximum(r, 1.0e-16)
        delta_f = np.abs(f_metric - f_schw)
    return delta_m, delta_f


def main() -> None:
    out_data = REPO_ROOT / "papers" / "paper-04" / "data"
    out_figs = REPO_ROOT / "papers" / "paper-04" / "figs"
    in_npz = out_data / "bh_profiles.npz"
    if not in_npz.exists():
        raise FileNotFoundError(f"Missing {in_npz}; run compute_bh_profiles.py first.")

    data = np.load(in_npz, allow_pickle=False)
    family_names = data["family_names"]
    r = data["r"]
    f_metric = data["f_metric"]
    ricci_scalar = data["ricci_scalar"]
    ricci_sq = data["ricci_sq"]
    kretschmann_proxy = data["kretschmann_proxy"]
    m = data["m"]
    horizon_radius = data["horizon_radius"]
    solver_success = data["solver_success"]

    summary: dict[str, dict[str, float | bool | str]] = {}
    all_delta_m = []
    all_delta_f = []

    for i, family in enumerate(family_names):
        name = str(family)
        ri = r[i]
        fi = f_metric[i]
        mi = m[i]
        d_m, d_f = _matching_residuals(ri, mi, fi)
        all_delta_m.append(d_m)
        all_delta_f.append(d_f)

        # Use first 2% points for central diagnostics, last 20% for matching.
        n = ri.size
        center_slice = slice(0, max(5, int(0.02 * n)))
        asym_slice = slice(max(0, int(0.8 * n)), n)

        summary[name] = {
            "solver_success": bool(solver_success[i, 0]),
            "adm_mass": float(mi[-1]),
            "horizon_radius": (
                float(horizon_radius[i, 0]) if np.isfinite(horizon_radius[i, 0]) else None
            ),
            "R_center_max_abs": float(np.max(np.abs(ricci_scalar[i, center_slice]))),
            "Ricci2_center_max": float(np.max(ricci_sq[i, center_slice])),
            "Kproxy_center_max": float(np.max(kretschmann_proxy[i, center_slice])),
            "delta_m_asym_max": float(np.max(d_m[asym_slice])),
            "delta_f_asym_max": float(np.max(d_f[asym_slice])),
            "note": "Kproxy is the Schwarzschild-like 48(m/r^3)^2 diagnostic.",
        }

    with (out_data / "bh_diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("[INFO] Wrote:", out_data / "bh_diagnostics.json")

    # Figure 2: curvature diagnostics.
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    for i, family in enumerate(family_names):
        label = str(family)
        ax[0].plot(r[i], np.abs(ricci_scalar[i]), lw=1.2, label=label)
        ax[1].plot(r[i], ricci_sq[i], lw=1.2, label=label)
        ax[2].plot(r[i], kretschmann_proxy[i], lw=1.2, label=label)
    for axi in ax:
        axi.set_xscale("log")
        axi.set_yscale("log")
        axi.grid(True, which="both", ls=":")
        axi.set_xlabel(r"$r$")
    ax[0].set_ylabel(r"$|R|$")
    ax[1].set_ylabel(r"$R_{\mu\nu}R^{\mu\nu}$")
    ax[2].set_ylabel(r"$K_{\rm proxy}$")
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(out_figs / "bh_curvature_regularization.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("[INFO] Wrote:", out_figs / "bh_curvature_regularization.png")

    # Figure 3: asymptotic matching residuals.
    delta_m = np.stack(all_delta_m, axis=0)
    delta_f = np.stack(all_delta_f, axis=0)
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for i, family in enumerate(family_names):
        label = str(family)
        ax[0].plot(r[i], delta_m[i], lw=1.2, label=label)
        ax[1].plot(r[i], delta_f[i], lw=1.2, label=label)
    for axi in ax:
        axi.set_xscale("log")
        axi.set_yscale("log")
        axi.grid(True, which="both", ls=":")
        axi.set_xlabel(r"$r$")
    ax[0].set_ylabel(r"$\delta_M(r)$")
    ax[1].set_ylabel(r"$\delta_F(r)$")
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(out_figs / "bh_matching_residuals.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("[INFO] Wrote:", out_figs / "bh_matching_residuals.png")


if __name__ == "__main__":
    main()
