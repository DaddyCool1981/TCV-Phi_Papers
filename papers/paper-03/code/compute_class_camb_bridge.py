#!/usr/bin/env python3
"""Compute CLASS bridge diagnostics for Paper-03 primordial table usage."""

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

from tcvphi.tcv_boltzmann import compute_pk_class, get_tcv_canonical_params


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CLASS bridge checks for Paper-03.")
    parser.add_argument(
        "--primordial-file",
        type=Path,
        default=REPO_ROOT / "papers" / "paper-03" / "data" / "PR_tcvphi.dat",
    )
    args = parser.parse_args()

    out_figs = REPO_ROOT / "papers" / "paper-03" / "figs"
    out_data = REPO_ROOT / "papers" / "paper-03" / "data"
    out_figs.mkdir(parents=True, exist_ok=True)
    out_data.mkdir(parents=True, exist_ok=True)

    theta = get_tcv_canonical_params()
    k_vals = np.logspace(-3, 0.7, 320)

    pk_powerlaw = compute_pk_class(
        theta_tcv=theta,
        k_vals=k_vals,
        primordial_mode="powerlaw",
        z=0.0,
    )
    pk_external = compute_pk_class(
        theta_tcv=theta,
        k_vals=k_vals,
        primordial_mode="external_file",
        primordial_file=args.primordial_file,
        z=0.0,
    )

    ratio = pk_external["pk"] / np.maximum(pk_powerlaw["pk"], 1.0e-300)

    np.savez(
        out_data / "class_bridge_pk_comparison.npz",
        k=k_vals,
        pk_powerlaw=pk_powerlaw["pk"],
        pk_external=pk_external["pk"],
        ratio=ratio,
    )

    summary = {
        "primordial_file_used": str(args.primordial_file),
        "powerlaw_used_fallback": bool(pk_powerlaw["used_fallback"][0]),
        "external_used_fallback": bool(pk_external["used_fallback"][0]),
        "powerlaw_error": str(pk_powerlaw["error"][0]),
        "external_error": str(pk_external["error"][0]),
        "pk_ratio_min": float(np.min(ratio)),
        "pk_ratio_max": float(np.max(ratio)),
        "camb_status": "not_run_in_this_script",
        "note": "This bridge script validates CLASS table-mode consistency. CAMB checks can be added in a dedicated follow-up script.",
    }
    with (out_data / "class_camb_bridge_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plt.figure(figsize=(8, 6))
    plt.loglog(k_vals, pk_powerlaw["pk"], lw=1.8, label="CLASS powerlaw mode")
    plt.loglog(k_vals, pk_external["pk"], lw=1.8, label="CLASS external-table mode")
    plt.xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
    plt.ylabel(r"$P(k)$")
    plt.title("CLASS bridge check: powerlaw vs external table")
    plt.grid(True, which="both", ls=":")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_figs / "class_bridge_pk_comparison.png", dpi=180, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 4.8))
    plt.semilogx(k_vals, ratio, lw=1.8)
    plt.axhline(1.0, ls="--", color="k", alpha=0.6)
    plt.xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
    plt.ylabel(r"$P_{\rm external}/P_{\rm powerlaw}$")
    plt.title("CLASS bridge ratio diagnostic")
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.savefig(out_figs / "class_bridge_pk_ratio.png", dpi=180, bbox_inches="tight")
    plt.close()

    print("[INFO] Wrote:", out_data / "class_bridge_pk_comparison.npz")
    print("[INFO] Wrote:", out_data / "class_camb_bridge_summary.json")
    print("[INFO] Wrote:", out_figs / "class_bridge_pk_comparison.png")
    print("[INFO] Wrote:", out_figs / "class_bridge_pk_ratio.png")


if __name__ == "__main__":
    main()
