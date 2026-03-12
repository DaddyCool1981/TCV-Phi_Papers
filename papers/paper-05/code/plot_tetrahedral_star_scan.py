#!/usr/bin/env python3
"""Extra plotting helper for tetrahedral-star scan outputs."""

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
    data_file = REPO_ROOT / "papers" / "paper-05" / "data" / "tetrahedral_star_scan.npz"
    if not data_file.exists():
        raise FileNotFoundError(f"Missing scan data: {data_file}")

    out_figs = REPO_ROOT / "papers" / "paper-05" / "figs"
    out_figs.mkdir(parents=True, exist_ok=True)

    d = np.load(data_file)
    t12 = np.asarray(d["theta12"], dtype=float)
    t13 = np.asarray(d["theta13"], dtype=float)
    t23 = np.asarray(d["theta23"], dtype=float)
    spread = np.asarray(d["triplet_spread"], dtype=float)
    ratio = np.asarray(d["triplet_pair_ratio"], dtype=float)
    target = np.asarray(d["target_mask"], dtype=int) > 0

    plt.figure(figsize=(6.6, 4.2))
    plt.scatter(spread, t12, s=12, alpha=0.35, label=r"$\theta_{12}$")
    plt.scatter(spread, t13, s=12, alpha=0.35, label=r"$\theta_{13}$")
    plt.scatter(spread, t23, s=12, alpha=0.35, label=r"$\theta_{23}$")
    if np.any(target):
        plt.scatter(spread[target], t23[target], s=26, alpha=0.85, label="target-like")
    plt.xlabel("triplet spread (abs)")
    plt.ylabel("angle (deg)")
    plt.title("Tetrahedral-star angle response vs triplet splitting")
    plt.grid(ls=":", alpha=0.4)
    plt.legend(fontsize=8)
    plt.tight_layout()
    f1 = out_figs / "tetrahedral_star_angles_vs_split_overlay.png"
    plt.savefig(f1, dpi=180, bbox_inches="tight")
    plt.close()
    print("[INFO] Wrote:", f1)

    plt.figure(figsize=(6.4, 4.2))
    plt.scatter(ratio, t23, s=12, alpha=0.35, label="all")
    if np.any(target):
        plt.scatter(ratio[target], t23[target], s=22, alpha=0.85, label="target-like")
    plt.axhline(45.0, color="k", ls="--", lw=1.0)
    plt.xlabel("closest/farthest triplet split ratio")
    plt.ylabel(r"$\theta_{23}$ (deg)")
    plt.title(r"$\theta_{23}$ vs branch quasi-doublet ratio")
    plt.grid(ls=":", alpha=0.4)
    plt.legend(fontsize=8)
    plt.tight_layout()
    f2 = out_figs / "tetrahedral_star_theta23_vs_pair_ratio.png"
    plt.savefig(f2, dpi=180, bbox_inches="tight")
    plt.close()
    print("[INFO] Wrote:", f2)


if __name__ == "__main__":
    main()

