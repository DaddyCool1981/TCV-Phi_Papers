#!/usr/bin/env python3
"""Plot v2.4 naturalness diagnostics."""

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

    s = json.loads((data_dir / "phi1_phi2_emergence_v24_naturalness_summary.json").read_text())

    # Figure 1: size scaling
    rows = s["size_scaling"]["rows"]
    n = np.array([r["N"] for r in rows], dtype=float)
    y = np.array([r["sqrt_proxy_mean"] for r in rows], dtype=float)
    coef = np.polyfit(np.log(n), np.log(y), 1)
    fit = np.exp(coef[1]) * n ** coef[0]

    plt.figure(figsize=(7.4, 4.8))
    plt.loglog(n, y, "o", label="data")
    plt.loglog(n, fit, "-", label=f"fit alpha={s['size_scaling']['alpha']:.2f}")
    plt.xlabel("N")
    plt.ylabel(r"$\sqrt{m2\_proxy}$")
    plt.title("v2.4 size scaling")
    plt.grid(which="both", ls=":", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    f1 = figs_dir / "phi1_phi2_v24_size_scaling.png"
    plt.savefig(f1, dpi=180, bbox_inches="tight")
    plt.close()
    print("[INFO] Wrote:", f1)

    # Figure 2: sensitivity bars
    sens = s["local_sensitivity"]["rows"]
    labels = [r["param"] for r in sens]
    vals = np.array([r["delta_log_sensitivity"] for r in sens], dtype=float)
    x = np.arange(len(labels))
    plt.figure(figsize=(8.0, 4.8))
    plt.bar(x, vals)
    plt.axhline(s["gates"]["max_delta_max"], ls="--", lw=1.0, color="tab:red")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel(r"$|\partial \ln m / \partial \ln p|$")
    plt.title("v2.4 local naturalness sensitivities")
    plt.grid(axis="y", ls=":", alpha=0.4)
    plt.tight_layout()
    f2 = figs_dir / "phi1_phi2_v24_sensitivity.png"
    plt.savefig(f2, dpi=180, bbox_inches="tight")
    plt.close()
    print("[INFO] Wrote:", f2)

    # Figure 3: overlap distribution
    ov = np.array([r["uldm_prior_overlap_fraction"] for r in s["uldm_overlap"]["rows"]], dtype=float)
    plt.figure(figsize=(7.2, 4.6))
    plt.hist(ov, bins=12, alpha=0.85)
    plt.axvline(s["gates"]["overlap_median_min"], ls="--", lw=1.0, color="tab:red")
    plt.xlabel("ULDM prior overlap fraction")
    plt.ylabel("count")
    plt.title("v2.4 structural overlap volume")
    plt.grid(axis="y", ls=":", alpha=0.4)
    plt.tight_layout()
    f3 = figs_dir / "phi1_phi2_v24_overlap_hist.png"
    plt.savefig(f3, dpi=180, bbox_inches="tight")
    plt.close()
    print("[INFO] Wrote:", f3)

    meta = {
        "verdict": s["verdict"],
        "checks": s["checks"],
        "alpha": s["size_scaling"]["alpha"],
        "max_delta": s["local_sensitivity"]["max_delta_log_sensitivity"],
        "overlap_summary": s["uldm_overlap"]["summary"],
    }
    out = data_dir / "phi1_phi2_emergence_v24_plot_meta.json"
    out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("[INFO] Wrote:", out)


if __name__ == "__main__":
    main()
