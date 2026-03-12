#!/usr/bin/env python3
"""Standalone plotting helper for tetrahedral flavour scan diagnostics."""

from __future__ import annotations

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
    data_file = REPO_ROOT / "papers" / "paper-05" / "data" / "tetrahedral_flavour_scan.npz"
    if not data_file.exists():
        raise FileNotFoundError(f"Missing scan data: {data_file}")

    out_figs = REPO_ROOT / "papers" / "paper-05" / "figs"
    out_figs.mkdir(parents=True, exist_ok=True)

    d = np.load(data_file)
    t12 = np.asarray(d["theta12"], dtype=float)
    t13 = np.asarray(d["theta13"], dtype=float)
    t23 = np.asarray(d["theta23"], dtype=float)
    x = np.asarray(d["rel_deform"], dtype=float)
    natural = np.asarray(d["natural_mask"], dtype=int) > 0
    target = np.asarray(d["target_mask"], dtype=int) > 0
    target_relaxed = np.asarray(d["target_relaxed_mask"], dtype=int) > 0 if "target_relaxed_mask" in d else np.zeros_like(target)
    geom_target = np.asarray(d["geom_target_mask"], dtype=int) > 0 if "geom_target_mask" in d else np.zeros_like(target)

    plt.figure(figsize=(6.4, 4.2))
    plt.scatter(t12, t13, s=14, alpha=0.35, label="all")
    if np.any(natural):
        plt.scatter(t12[natural], t13[natural], s=16, alpha=0.55, label="natural")
    if np.any(target):
        plt.scatter(t12[target], t13[target], s=20, alpha=0.85, label="target-like")
    plt.xlabel(r"$\theta_{12}$ (deg)")
    plt.ylabel(r"$\theta_{13}$ (deg)")
    plt.title(r"Tetrahedral scan: $(\theta_{12},\theta_{13})$")
    plt.grid(ls=":", alpha=0.4)
    plt.legend(fontsize=8)
    plt.tight_layout()
    f1 = out_figs / "tetrahedral_theta12_theta13_scatter.png"
    plt.savefig(f1, dpi=180, bbox_inches="tight")
    plt.close()
    print("[INFO] Wrote:", f1)

    plt.figure(figsize=(6.4, 4.2))
    plt.scatter(x, t23, s=12, alpha=0.35, label="all")
    if np.any(target):
        plt.scatter(x[target], t23[target], s=22, alpha=0.85, label="PMNS-target-like")
    if np.any(target_relaxed):
        plt.scatter(x[target_relaxed], t23[target_relaxed], s=22, alpha=0.85, label="PMNS-target-like (relaxed)")
    if np.any(geom_target):
        plt.scatter(x[geom_target], t23[geom_target], s=22, alpha=0.85, label="geometry-target-like")
    plt.axhline(45.0, color="k", ls="--", lw=1.0)
    plt.xlabel(r"relative deformation")
    plt.ylabel(r"$\theta_{23}$ (deg)")
    plt.title(r"Tetrahedral scan: $\theta_{23}$ vs deformation")
    plt.grid(ls=":", alpha=0.4)
    plt.legend(fontsize=8)
    plt.tight_layout()
    f2 = out_figs / "tetrahedral_theta23_scatter_alt.png"
    plt.savefig(f2, dpi=180, bbox_inches="tight")
    plt.close()
    print("[INFO] Wrote:", f2)


if __name__ == "__main__":
    main()
