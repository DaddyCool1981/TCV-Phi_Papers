#!/usr/bin/env python3
"""Compute Paper-03 primordial observables from tabulated P_R(k)."""

from __future__ import annotations

import argparse
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

from tcvphi.tcv_boltzmann import infer_powerlaw_from_primordial_file


def _fit_running(
    k: np.ndarray,
    p_r: np.ndarray,
    k_pivot: float,
    fit_window_points: int,
) -> tuple[float, float]:
    logk = np.log(k)
    logp = np.log(np.maximum(p_r, 1.0e-300))
    x = logk - np.log(k_pivot)
    idx = np.argsort(np.abs(x))[: min(max(3, fit_window_points), x.size)]
    # log P = c2 x^2 + c1 x + c0
    c2, c1, _ = np.polyfit(x[idx], logp[idx], 2)
    ns = float(1.0 + c1)
    alpha_s = float(2.0 * c2)
    return ns, alpha_s


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute primordial observables for Paper-03.")
    parser.add_argument(
        "--primordial-file",
        type=Path,
        default=REPO_ROOT / "papers" / "paper-03" / "data" / "PR_tcvphi.dat",
    )
    parser.add_argument("--k-pivot", type=float, default=0.05)
    parser.add_argument(
        "--fit-window",
        type=int,
        default=35,
        help="Number of k-points closest to k_pivot used in the quadratic local fit.",
    )
    args = parser.parse_args()

    out_figs = REPO_ROOT / "papers" / "paper-03" / "figs"
    out_data = REPO_ROOT / "papers" / "paper-03" / "data"
    out_figs.mkdir(parents=True, exist_ok=True)
    out_data.mkdir(parents=True, exist_ok=True)

    data = np.loadtxt(args.primordial_file)
    k = np.asarray(data[:, 0], dtype=float)
    p_r = np.asarray(data[:, 1], dtype=float)
    mask = (k > 0.0) & (p_r > 0.0)
    k = k[mask]
    p_r = p_r[mask]

    As_eff, ns_eff = infer_powerlaw_from_primordial_file(args.primordial_file, k_pivot=args.k_pivot)
    ns_quad, alpha_s = _fit_running(
        k,
        p_r,
        k_pivot=args.k_pivot,
        fit_window_points=args.fit_window,
    )

    logk = np.log(k)
    logp = np.log(p_r)
    ns_local = 1.0 + np.gradient(logp, logk)

    summary = {
        "k_pivot": float(args.k_pivot),
        "A_s_from_local_fit": float(As_eff),
        "n_s_from_local_fit": float(ns_eff),
        "n_s_from_quadratic_fit": float(ns_quad),
        "alpha_s_from_quadratic_fit": float(alpha_s),
        "fit_window_points": int(args.fit_window),
        "r": None,
        "note": "Tensor-to-scalar ratio r is not extracted by this scalar-only script.",
        "primordial_file_used": str(args.primordial_file),
    }
    with (out_data / "primordial_observables.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plt.figure(figsize=(8, 5))
    plt.semilogx(k, ns_local, lw=1.4)
    plt.axhline(ns_eff, ls="--", color="k", alpha=0.7, label=rf"$n_s^{{\rm fit}}={ns_eff:.4f}$")
    plt.xlabel(r"$k$")
    plt.ylabel(r"$n_s(k)$")
    plt.title("Local spectral-index diagnostic")
    plt.grid(True, which="both", ls=":")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_figs / "primordial_observables_diagnostics.png", dpi=180, bbox_inches="tight")
    plt.close()

    print("[INFO] Wrote:", out_data / "primordial_observables.json")
    print("[INFO] Wrote:", out_figs / "primordial_observables_diagnostics.png")
    print("[INFO] n_s (local fit) =", f"{ns_eff:.6f}")
    print("[INFO] alpha_s (quad fit) =", f"{alpha_s:.6e}")


if __name__ == "__main__":
    main()
