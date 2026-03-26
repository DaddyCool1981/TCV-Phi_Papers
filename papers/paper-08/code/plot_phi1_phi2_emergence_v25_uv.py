#!/usr/bin/env python3
"""Plot v2.5 UV-compact diagnostics."""

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


def main() -> None:
    data_dir = REPO_ROOT / "papers" / "paper-08" / "data"
    figs_dir = REPO_ROOT / "papers" / "paper-08" / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    s = json.loads((data_dir / "phi1_phi2_emergence_v25_uv_summary.json").read_text())

    rows = s["scale_rows"]
    n = np.array([r["N"] for r in rows], dtype=float)
    lam_tw = np.array([r["twisted_ring"]["lambda1_mean"] for r in rows], dtype=float)
    lam_rg = np.array([r["ring"]["lambda1_mean"] for r in rows], dtype=float)
    z0_tw = np.array([r["twisted_ring"]["z0_mean"] for r in rows], dtype=float)
    z0_rg = np.array([r["ring"]["z0_mean"] for r in rows], dtype=float)

    fit_lam_tw = np.array(s["fits"]["lambda_twisted"]["y_fit"], dtype=float)
    fit_lam_rg = np.array(s["fits"]["lambda_ring"]["y_fit"], dtype=float)

    # Fig1: lambda scaling ring vs twisted.
    plt.figure(figsize=(7.8, 4.8))
    plt.loglog(n, lam_tw, "o", label="twisted lambda1")
    plt.loglog(n, fit_lam_tw, "-", label="twisted fit")
    plt.loglog(n, lam_rg, "s", label="ring lambda1")
    plt.loglog(n, fit_lam_rg, "-", label="ring fit")
    plt.xlabel("N")
    plt.ylabel("lambda1")
    plt.title("v2.5 soft-mode scaling")
    plt.grid(which="both", ls=":", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    f1 = figs_dir / "phi1_phi2_v25_lambda_scaling.png"
    plt.savefig(f1, dpi=180, bbox_inches="tight")
    plt.close()
    print("[INFO] Wrote:", f1)

    # Fig2: z0 scaling and twist effect.
    ratio = lam_tw / np.maximum(lam_rg, 1.0e-30)
    plt.figure(figsize=(10.0, 4.6))
    ax1 = plt.subplot(1, 2, 1)
    ax1.loglog(n, z0_tw, "o-", label="twisted z0")
    ax1.loglog(n, z0_rg, "s-", label="ring z0")
    ax1.set_xlabel("N")
    ax1.set_ylabel("z0")
    ax1.grid(which="both", ls=":", alpha=0.4)
    ax1.legend()

    ax2 = plt.subplot(1, 2, 2)
    ax2.semilogx(n, ratio, "o-")
    ax2.axhline(1.0, ls="--", lw=1.0, color="k")
    ax2.set_xlabel("N")
    ax2.set_ylabel("lambda1_twisted / lambda1_ring")
    ax2.grid(ls=":", alpha=0.4)
    plt.suptitle("v2.5 inertia/topology diagnostics")
    plt.tight_layout()
    f2 = figs_dir / "phi1_phi2_v25_z0_and_twist_ratio.png"
    plt.savefig(f2, dpi=180, bbox_inches="tight")
    plt.close()
    print("[INFO] Wrote:", f2)

    # Fig3: pass/fail dashboard.
    checks = s["checks"]
    names = list(checks.keys())
    vals = np.array([1.0 if checks[k] else 0.0 for k in names], dtype=float)
    plt.figure(figsize=(8.0, 4.6))
    plt.bar(np.arange(len(names)), vals)
    plt.ylim(-0.05, 1.05)
    plt.xticks(np.arange(len(names)), names, rotation=25, ha="right")
    plt.ylabel("pass=1 / fail=0")
    plt.title(f"v2.5 UV compact checks ({s['verdict']})")
    plt.grid(axis="y", ls=":", alpha=0.4)
    plt.tight_layout()
    f3 = figs_dir / "phi1_phi2_v25_uv_checks.png"
    plt.savefig(f3, dpi=180, bbox_inches="tight")
    plt.close()
    print("[INFO] Wrote:", f3)

    out = data_dir / "phi1_phi2_emergence_v25_plot_meta.json"
    out.write_text(json.dumps({"verdict": s["verdict"], "checks": checks}, indent=2), encoding="utf-8")
    print("[INFO] Wrote:", out)


if __name__ == "__main__":
    main()
