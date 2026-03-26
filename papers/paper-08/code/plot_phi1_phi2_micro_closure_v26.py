#!/usr/bin/env python3
"""Plot v26 micro-closure scan."""

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

    s = json.loads((data_dir / "phi1_phi2_micro_closure_v26_summary.json").read_text())
    recs = s["records"]
    egrid = s["grid"]["inertia_eta"]
    tgrid = s["grid"]["inertia_twist"]

    z_ratio = np.full((len(egrid), len(tgrid)), np.nan)
    z_pass = np.zeros((len(egrid), len(tgrid)))
    for r in recs:
        i = egrid.index(r["inertia_eta"])
        j = tgrid.index(r["inertia_twist"])
        z_ratio[i, j] = r["m0sq_ratio_tw_over_ring"]
        z_pass[i, j] = 1.0 if r["all_pass"] else 0.0

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.6), constrained_layout=True)
    im0 = ax[0].imshow(z_ratio, origin="lower", aspect="auto", cmap="viridis")
    ax[0].set_title("m0^2(twisted)/m0^2(ring)")
    ax[0].set_xticks(np.arange(len(tgrid)))
    ax[0].set_xticklabels([f"{x:.1f}" for x in tgrid])
    ax[0].set_yticks(np.arange(len(egrid)))
    ax[0].set_yticklabels([f"{x:.1f}" for x in egrid])
    ax[0].set_xlabel("inertia_twist")
    ax[0].set_ylabel("inertia_eta")
    fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.02)

    im1 = ax[1].imshow(z_pass, origin="lower", aspect="auto", cmap="Greens", vmin=0.0, vmax=1.0)
    ax[1].set_title("all checks pass")
    ax[1].set_xticks(np.arange(len(tgrid)))
    ax[1].set_xticklabels([f"{x:.1f}" for x in tgrid])
    ax[1].set_yticks(np.arange(len(egrid)))
    ax[1].set_yticklabels([f"{x:.1f}" for x in egrid])
    ax[1].set_xlabel("inertia_twist")
    ax[1].set_ylabel("inertia_eta")
    fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.02)

    f1 = figs_dir / "phi1_phi2_micro_v26_scan.png"
    fig.savefig(f1, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("[INFO] Wrote:", f1)

    meta = {
        "verdict": s["verdict"],
        "n_all_pass": s["n_all_pass"],
        "best_ratio_record": s["best_ratio_record"],
    }
    out = data_dir / "phi1_phi2_micro_closure_v26_plot_meta.json"
    out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("[INFO] Wrote:", out)


if __name__ == "__main__":
    main()
