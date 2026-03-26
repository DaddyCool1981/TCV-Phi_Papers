#!/usr/bin/env python3
"""Plot v29 conservative no-fine-tuning scan."""

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

    s = json.loads((data_dir / "phi1_phi2_micro_closure_v29_summary.json").read_text())
    recs = s["records"]
    mixes = s["mix_grid"]

    pass_frac = []
    best_ratio = []
    for mv in mixes:
        sub = [r for r in recs if abs(r["mix_twist"] - mv) < 1e-12]
        pass_frac.append(np.mean([1.0 if r["all_pass"] else 0.0 for r in sub]))
        best_ratio.append(min(r["m0sq_ratio_tw_over_ring"] for r in sub))

    fig, ax = plt.subplots(1, 2, figsize=(10.2, 4.6), constrained_layout=True)
    ax[0].plot(mixes, pass_frac, "-o")
    ax[0].axhline(0.20, ls="--", lw=1.0, color="tab:red")
    ax[0].set_xlabel("mix_twist")
    ax[0].set_ylabel("pass fraction in neighborhood")
    ax[0].set_title("No-fine-tuning criterion")
    ax[0].grid(ls=":", alpha=0.4)

    ax[1].plot(mixes, best_ratio, "-o")
    ax[1].axhline(1.0, ls="--", lw=1.0, color="k")
    ax[1].set_xlabel("mix_twist")
    ax[1].set_ylabel("best m0^2 ratio (tw/ring)")
    ax[1].set_title("Best ratio per mix")
    ax[1].grid(ls=":", alpha=0.4)

    f1 = figs_dir / "phi1_phi2_micro_v29_mix_scan.png"
    fig.savefig(f1, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("[INFO] Wrote:", f1)

    # Scatter for all records
    ratio = np.array([r["m0sq_ratio_tw_over_ring"] for r in recs], dtype=float)
    mixv = np.array([r["mix_twist"] for r in recs], dtype=float)
    passf = np.array([1 if r["all_pass"] else 0 for r in recs], dtype=int)

    plt.figure(figsize=(7.6, 4.8))
    plt.scatter(mixv[passf == 0], ratio[passf == 0], s=20, alpha=0.5, label="NO")
    if np.any(passf == 1):
        plt.scatter(mixv[passf == 1], ratio[passf == 1], s=34, alpha=0.85, label="PASS")
    plt.axhline(1.0, ls="--", lw=1.0, color="k")
    plt.xlabel("mix_twist")
    plt.ylabel("m0^2 ratio tw/ring")
    plt.title("v29 pass frontier")
    plt.grid(ls=":", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    f2 = figs_dir / "phi1_phi2_micro_v29_scatter.png"
    plt.savefig(f2, dpi=180, bbox_inches="tight")
    plt.close()
    print("[INFO] Wrote:", f2)

    out = data_dir / "phi1_phi2_micro_closure_v29_plot_meta.json"
    out.write_text(
        json.dumps(
            {
                "verdict": s["verdict"],
                "n_all_pass": s["n_all_pass"],
                "max_pass_fraction_over_mix": s["max_pass_fraction_over_mix"],
                "best_ratio_record": s["best_ratio_record"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("[INFO] Wrote:", out)


if __name__ == "__main__":
    main()
