#!/usr/bin/env python3
"""Compare center->outward and exterior->inward integrations for Paper-04."""

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

from tcvphi.bh_static import StaticBHParams, solve_static_ivp, solve_static_ivp_from_state  # noqa: E402


def _float_or_none(val: float | None) -> float | None:
    if val is None:
        return None
    return float(val) if np.isfinite(val) else None


def _horizon_from_profile(r: np.ndarray, f_metric: np.ndarray) -> float | None:
    idx = np.where(np.diff(np.sign(f_metric)) != 0)[0]
    if idx.size == 0:
        return None
    return float(r[idx[0]])


def _reverse_profile(profile: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, arr in profile.items():
        if arr.ndim == 1 and arr.size > 1:
            out[key] = arr[::-1]
        else:
            out[key] = arr
    return out


def main() -> None:
    out_data = REPO_ROOT / "papers" / "paper-04" / "data"
    out_figs = REPO_ROOT / "papers" / "paper-04" / "figs"
    out_data.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    # The third benchmark intentionally pushes toward a near-horizon branch
    # in center->outward integration, to compare with exterior->inward runs.
    rho_values = [1.0e-12, 1.0e-10, 1.0e-8]
    scan_rows: list[dict[str, object]] = []

    r_plot_out = []
    f_plot_out = []
    r_plot_in = []
    f_plot_in = []

    for rho_cc0 in rho_values:
        params = StaticBHParams(
            rho_cc0=rho_cc0,
            alpha_cc=0.0,
            beta_cc=0.0,
            xi0=0.0,
            r_max=1.0e4,
            r_points=2200,
            atol=1.0e-8,
            rtol=1.0e-7,
        )

        prof_out, sol_out = solve_static_ivp(phi1_center=1.0, params=params, method="RK45")
        r_out = prof_out["r"]
        f_out = prof_out["f_metric"]
        m_out = prof_out["m"]

        y0_in = np.array([m_out[-1], 0.0, params.phi1_0, 0.0], dtype=float)
        prof_in_raw, sol_in = solve_static_ivp_from_state(
            y0=y0_in,
            r_start=params.r_max,
            r_end=params.r_min,
            params=params,
            method="Radau",
            n_eval=params.r_points,
        )
        prof_in = _reverse_profile(prof_in_raw)
        r_in = prof_in["r"]
        f_in = prof_in["f_metric"]

        row = {
            "rho_cc0": float(rho_cc0),
            "outward_success": bool(sol_out.success),
            "outward_message": str(sol_out.message),
            "outward_fmin": float(np.min(f_out)),
            "outward_horizon_radius": _horizon_from_profile(r_out, f_out),
            "outward_r_end": float(r_out[-1]),
            "inward_success": bool(sol_in.success),
            "inward_message": str(sol_in.message),
            "inward_fmin": float(np.min(f_in)),
            "inward_horizon_radius": _horizon_from_profile(r_in, f_in),
            "inward_r_start": float(r_in[0]),
            "adm_mass_used": float(m_out[-1]),
        }
        scan_rows.append(row)

        r_plot_out.append(r_out)
        f_plot_out.append(f_out)
        r_plot_in.append(r_in)
        f_plot_in.append(f_in)

        print(
            f"[INFO] rho={rho_cc0:.1e} "
            f"outward(success={sol_out.success}, fmin={np.min(f_out):.3e}) "
            f"inward(success={sol_in.success}, fmin={np.min(f_in):.3e})"
        )

    with (out_data / "bh_direction_scan.json").open("w", encoding="utf-8") as f:
        json.dump(scan_rows, f, indent=2)
    print("[INFO] Wrote:", out_data / "bh_direction_scan.json")

    # Plot 1: F(r) comparison for the highest-density benchmark.
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    i_ref = len(rho_values) - 1
    ax[0].plot(r_plot_out[i_ref], f_plot_out[i_ref], lw=1.6, label="center->outward")
    ax[0].plot(r_plot_in[i_ref], f_plot_in[i_ref], lw=1.6, ls="--", label="exterior->inward")
    ax[0].axhline(0.0, color="k", lw=0.9, alpha=0.6)
    ax[0].set_xscale("log")
    ax[0].set_xlabel(r"$r$")
    ax[0].set_ylabel(r"$F(r)=1-2m/r$")
    ax[0].set_title(rf"Direction comparison (rho_cc0={rho_values[i_ref]:.1e})")
    ax[0].grid(True, which="both", ls=":")
    ax[0].legend()

    # Plot 2: minimum F reached by each branch versus benchmark density.
    x = np.arange(len(rho_values))
    out_fmin = [row["outward_fmin"] for row in scan_rows]
    in_fmin = [row["inward_fmin"] for row in scan_rows]
    ax[1].plot(x, out_fmin, "o-", lw=1.5, label="center->outward")
    ax[1].plot(x, in_fmin, "s--", lw=1.5, label="exterior->inward")
    ax[1].axhline(0.0, color="k", lw=0.9, alpha=0.6)
    ax[1].set_xticks(x, [f"{rho:.0e}" for rho in rho_values], rotation=0)
    ax[1].set_xlabel(r"benchmark $\rho_{\rm cc,0}$")
    ax[1].set_ylabel(r"$\min F(r)$")
    ax[1].set_title("Near-horizon approach by integration direction")
    ax[1].grid(True, which="both", ls=":")
    ax[1].legend()

    fig.tight_layout()
    fig.savefig(out_figs / "bh_direction_dependence.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("[INFO] Wrote:", out_figs / "bh_direction_dependence.png")

    # Compact table for manuscript insertion.
    rows_table = []
    for row in scan_rows:
        rows_table.append(
            {
                "rho_cc0": float(row["rho_cc0"]),
                "fmin_out": float(row["outward_fmin"]),
                "fmin_in": float(row["inward_fmin"]),
                "rH_out": _float_or_none(row["outward_horizon_radius"]),
                "rH_in": _float_or_none(row["inward_horizon_radius"]),
                "out_success": bool(row["outward_success"]),
                "in_success": bool(row["inward_success"]),
            }
        )
    with (out_data / "bh_direction_scan_table.json").open("w", encoding="utf-8") as f:
        json.dump(rows_table, f, indent=2)
    print("[INFO] Wrote:", out_data / "bh_direction_scan_table.json")


if __name__ == "__main__":
    main()
