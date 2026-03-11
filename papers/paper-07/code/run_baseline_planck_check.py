#!/usr/bin/env python3
"""Baseline technical check before Planck likelihood runs (Paper 07)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CORE_LIB = REPO_ROOT / "core" / "lib"
if str(CORE_LIB) not in sys.path:
    sys.path.insert(0, str(CORE_LIB))

mpl_dir = REPO_ROOT / ".tmp_mpl"
mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tcvphi.tcv_boltzmann import compute_pk_class, get_tcv_canonical_params


def _safe_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a / np.maximum(np.abs(b), 1.0e-300)


def main() -> None:
    out_data = REPO_ROOT / "papers" / "paper-07" / "data"
    out_figs = REPO_ROOT / "papers" / "paper-07" / "figs"
    out_data.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    theta = get_tcv_canonical_params()
    k_vals = np.logspace(-3, 0.7, 320)

    pk_powerlaw = compute_pk_class(theta_tcv=theta, k_vals=k_vals, primordial_mode="powerlaw")

    p3_primordial = REPO_ROOT / "papers" / "paper-03" / "data" / "PR_tcvphi.dat"
    if p3_primordial.exists():
        pk_external = compute_pk_class(
            theta_tcv=theta,
            k_vals=k_vals,
            primordial_mode="external_file",
            primordial_file=p3_primordial,
        )
        external_note = "External primordial file from Paper III used."
    else:
        pk_external = {
            "k": pk_powerlaw["k"],
            "pk": pk_powerlaw["pk"].copy(),
            "used_fallback": np.array([True]),
            "error": np.array(["Paper-03 primordial file missing; copied powerlaw for baseline sanity."], dtype=object),
        }
        external_note = "Paper-03 primordial file missing; external branch replaced by powerlaw copy."

    ratio = _safe_ratio(np.asarray(pk_external["pk"]), np.asarray(pk_powerlaw["pk"]))

    summary = {
        "paper": "paper-07",
        "scope": "baseline pre-likelihood validation",
        "canonical_params": theta,
        "k_range_h_Mpc": [float(np.min(k_vals)), float(np.max(k_vals))],
        "pk_ratio_external_over_powerlaw_min": float(np.min(ratio)),
        "pk_ratio_external_over_powerlaw_max": float(np.max(ratio)),
        "powerlaw_used_fallback": bool(np.asarray(pk_powerlaw["used_fallback"])[0]),
        "external_used_fallback": bool(np.asarray(pk_external["used_fallback"])[0]),
        "powerlaw_error": str(np.asarray(pk_powerlaw["error"])[0]),
        "external_error": str(np.asarray(pk_external["error"])[0]),
        "note": external_note,
    }

    out_json = out_data / "baseline_validation.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("[INFO] Wrote:", out_json)

    fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.2))

    ax[0].loglog(k_vals, pk_powerlaw["pk"], lw=1.6, label="Power-law mode")
    ax[0].loglog(k_vals, pk_external["pk"], lw=1.4, ls="--", label="External-file mode")
    ax[0].set_xlabel(r"$k\,[h/{\rm Mpc}]$")
    ax[0].set_ylabel(r"$P(k)$")
    ax[0].set_title("Baseline CLASS bridge check")
    ax[0].grid(True, which="both", ls=":", alpha=0.5)
    ax[0].legend(loc="best")

    ax[1].semilogx(k_vals, ratio, lw=1.6)
    ax[1].axhline(1.0, color="k", lw=1.0, ls=":")
    ax[1].set_xlabel(r"$k\,[h/{\rm Mpc}]$")
    ax[1].set_ylabel(r"$P_{\rm ext}(k)/P_{\rm pl}(k)$")
    ax[1].set_title("Ratio (external/power-law)")
    ax[1].grid(True, ls=":", alpha=0.5)

    fig.tight_layout()
    out_fig = out_figs / "baseline_pk_comparison.png"
    fig.savefig(out_fig, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("[INFO] Wrote:", out_fig)


if __name__ == "__main__":
    main()
