#!/usr/bin/env python3
"""Slow-rotation (Kerr-like) extension on top of Paper-04 static backgrounds."""

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


def _solve_omega_profile(
    r_arr: np.ndarray,
    f_arr: np.ndarray,
    phi1_arr: np.ndarray,
    *,
    r_ref: float,
    a_over_m: float,
    xi0: float,
) -> tuple[np.ndarray, np.ndarray]:
    rr = np.asarray(r_arr, dtype=float)
    # Use mass from asymptotic compactness scale:
    m_est = float(np.max((1.0 - f_arr) * r_arr / 2.0))
    m_adm = max(m_est, 1.0e-16)
    a_phys = a_over_m * m_adm
    meff2_inf = 1.0 + xi0 * float(phi1_arr[-1]) ** 2
    c_target = (2.0 * a_phys * m_adm) / max(meff2_inf, 1.0e-14)

    # Template-level slow-rotation profile:
    # asymptotically omega ~ c_target / r^3 with cohesive suppression near core.
    smooth = 1.0 - np.exp(-(rr / max(r_ref, rr[0] + 1.0e-12)) ** 2)
    omega = (c_target / np.maximum(rr, 1.0e-12) ** 3) * smooth
    return rr, omega


def main() -> None:
    out_data = REPO_ROOT / "papers" / "paper-04" / "data"
    out_figs = REPO_ROOT / "papers" / "paper-04" / "figs"
    in_npz = out_data / "bh_profiles.npz"
    if not in_npz.exists():
        raise FileNotFoundError(f"Missing {in_npz}; run compute_bh_profiles.py first.")

    data = np.load(in_npz, allow_pickle=False)
    families = [str(x) for x in data["family_names"]]
    if "dense_core" in families:
        i_ref = families.index("dense_core")
    else:
        i_ref = len(families) - 1

    r = data["r"][i_ref]
    f_metric = data["f_metric"][i_ref]
    phi1 = data["phi1"][i_ref]

    i_fmin = int(np.argmin(f_metric))
    r_ref = float(r[i_fmin])
    f_min = float(f_metric[i_fmin])

    a_values = np.array([0.0, 0.05, 0.10, 0.20], dtype=float)
    xi0 = 0.1

    omega_rows = []
    r_omega = None
    for a_over_m in a_values:
        if a_over_m == 0.0:
            rr = r.copy()
            omega = np.zeros_like(rr)
        else:
            rr, omega = _solve_omega_profile(
                r,
                f_metric,
                phi1,
                r_ref=r_ref,
                a_over_m=float(a_over_m),
                xi0=xi0,
            )
        if r_omega is None:
            r_omega = rr
            omega_rows.append(omega)
        else:
            omega_rows.append(np.interp(r_omega, rr, omega))

    omega_arr = np.stack(omega_rows, axis=0)
    out_npz = out_data / "bh_kerr_slow.npz"
    np.savez(
        out_npz,
        a_over_m=a_values,
        r=r_omega,
        omega=omega_arr,
        family=np.array([families[i_ref]]),
        r_ref=np.array([r_ref]),
        f_min=np.array([f_min]),
        xi0=np.array([xi0]),
    )
    print("[INFO] Wrote:", out_npz)

    summary = {
        "family_used": families[i_ref],
        "r_ref": r_ref,
        "f_min": f_min,
        "xi0": xi0,
        "a_over_m_values": [float(x) for x in a_values],
        "note": "Slow-rotation template built on static branch; r_ref is the radius where F reaches its minimum.",
    }
    with (out_data / "bh_kerr_slow_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("[INFO] Wrote:", out_data / "bh_kerr_slow_summary.json")

    plt.figure(figsize=(8, 4.8))
    for i, a_over_m in enumerate(a_values):
        plt.plot(r_omega, omega_arr[i], lw=1.5, label=rf"$a/M={a_over_m:.2f}$")
    plt.xscale("log")
    plt.xlabel(r"$r$")
    plt.ylabel(r"$\omega(r)$")
    plt.title("Paper-04 slow-rotation frame-dragging profiles")
    plt.grid(True, which="both", ls=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_figs / "bh_kerr_frame_dragging.png", dpi=180, bbox_inches="tight")
    plt.close()
    print("[INFO] Wrote:", out_figs / "bh_kerr_frame_dragging.png")


if __name__ == "__main__":
    main()
