#!/usr/bin/env python3
"""Plot v27 micro-closure scan with loop kinetic term."""

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

    s = json.loads((data_dir / "phi1_phi2_micro_closure_v27_summary.json").read_text())
    recs = s["records"]
    loop_vals = s["grid"]["inertia_loop"]

    # Figure 1: best m0 ratio vs loop strength.
    loop_best = []
    loop_pass = []
    for lv in loop_vals:
        sub = [r for r in recs if abs(r["inertia_loop"] - lv) < 1e-12]
        ratios = [r["m0sq_ratio_tw_over_ring"] for r in sub]
        loop_best.append(min(ratios))
        loop_pass.append(sum(1 for r in sub if r["all_pass"]))

    fig, ax = plt.subplots(1, 2, figsize=(10.0, 4.6))
    ax[0].plot(loop_vals, loop_best, "-o")
    ax[0].axhline(1.0, ls="--", lw=1.0, color="k")
    ax[0].set_xlabel("inertia_loop")
    ax[0].set_ylabel("best m0^2 ratio (tw/ring)")
    ax[0].set_title("Best ratio vs loop kinetic strength")
    ax[0].grid(ls=":", alpha=0.4)

    ax[1].bar(np.arange(len(loop_vals)), loop_pass)
    ax[1].set_xticks(np.arange(len(loop_vals)))
    ax[1].set_xticklabels([f"{x:.1f}" for x in loop_vals])
    ax[1].set_xlabel("inertia_loop")
    ax[1].set_ylabel("# all-pass points")
    ax[1].set_title("All-pass count vs loop strength")
    ax[1].grid(axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    f1 = figs_dir / "phi1_phi2_micro_v27_loop_scan.png"
    fig.savefig(f1, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("[INFO] Wrote:", f1)

    # Figure 2: scatter of ratio with pass/fail.
    ratio = np.array([r["m0sq_ratio_tw_over_ring"] for r in recs], dtype=float)
    passf = np.array([1 if r["all_pass"] else 0 for r in recs], dtype=int)
    eloop = np.array([r["inertia_loop"] for r in recs], dtype=float)

    plt.figure(figsize=(7.6, 4.8))
    plt.scatter(eloop[passf == 0], ratio[passf == 0], s=24, alpha=0.5, label="NO")
    if np.any(passf == 1):
        plt.scatter(eloop[passf == 1], ratio[passf == 1], s=36, alpha=0.8, label="PASS")
    plt.axhline(1.0, ls="--", lw=1.0, color="k")
    plt.xlabel("inertia_loop")
    plt.ylabel("m0^2 ratio tw/ring")
    plt.title("v27 micro-closure pass frontier")
    plt.grid(ls=":", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    f2 = figs_dir / "phi1_phi2_micro_v27_scatter.png"
    plt.savefig(f2, dpi=180, bbox_inches="tight")
    plt.close()
    print("[INFO] Wrote:", f2)

    meta = {
        "verdict": s["verdict"],
        "n_all_pass": s["n_all_pass"],
        "best_ratio_record": s["best_ratio_record"],
    }
    out = data_dir / "phi1_phi2_micro_closure_v27_plot_meta.json"
    out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("[INFO] Wrote:", out)


if __name__ == "__main__":
    main()
