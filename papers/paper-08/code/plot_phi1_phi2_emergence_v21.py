#!/usr/bin/env python3
"""Plot focused v2.1 emergence diagnostics (grid + GO/NO-GO maps)."""

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


def _build_grid(records: list[dict], noise_value: float, key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    subset = [r for r in records if abs(float(r["params"]["onsite_noise"]) - noise_value) < 1.0e-12]
    edges = sorted({float(r["params"]["edge_aniso"]) for r in subset})
    drops = sorted({float(r["params"]["drop_prob"]) for r in subset})
    z = np.full((len(edges), len(drops)), np.nan, dtype=float)
    for r in subset:
        e = float(r["params"]["edge_aniso"])
        d = float(r["params"]["drop_prob"])
        i = edges.index(e)
        j = drops.index(d)
        z[i, j] = float(r[key])
    return np.array(edges), np.array(drops), z


def main() -> None:
    data_dir = REPO_ROOT / "papers" / "paper-08" / "data"
    figs_dir = REPO_ROOT / "papers" / "paper-08" / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    s = json.loads((data_dir / "phi1_phi2_emergence_v21_summary.json").read_text())
    records = s["records"]

    # Figure 1: scatter of joint vs robustness with GO/NO-GO coloring.
    joint = np.array([r["twisted_joint_mean"] for r in records], dtype=float)
    robust = np.array([r["twisted_robust_mean"] for r in records], dtype=float)
    go = np.array([1 if r["overall_go"] else 0 for r in records], dtype=int)

    plt.figure(figsize=(7.4, 5.0))
    plt.scatter(joint[go == 0], robust[go == 0], s=46, c="tab:red", alpha=0.75, label="NO-GO")
    plt.scatter(joint[go == 1], robust[go == 1], s=52, c="tab:green", alpha=0.85, label="GO")
    plt.axvline(0.35, ls="--", lw=1.0, color="tab:blue", alpha=0.8)
    plt.axhline(0.20, ls="--", lw=1.0, color="tab:orange", alpha=0.8)
    plt.xlabel("twisted_joint_mean")
    plt.ylabel("twisted_robust_mean")
    plt.title("v2.1 GO/NO-GO frontier")
    plt.grid(ls=":", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    f1 = figs_dir / "phi1_phi2_v21_go_nogo_scatter.png"
    plt.savefig(f1, dpi=180, bbox_inches="tight")
    plt.close()
    print("[INFO] Wrote:", f1)

    # Figure 2: heatmaps at fixed onsite_noise values.
    noise_vals = sorted({float(r["params"]["onsite_noise"]) for r in records})
    fig, ax = plt.subplots(len(noise_vals), 2, figsize=(10.0, 3.4 * len(noise_vals)), constrained_layout=True)
    if len(noise_vals) == 1:
        ax = np.array([ax])

    for i, nv in enumerate(noise_vals):
        edges, drops, z_joint = _build_grid(records, nv, "twisted_joint_mean")
        _, _, z_rob = _build_grid(records, nv, "twisted_robust_mean")

        im0 = ax[i, 0].imshow(z_joint, origin="lower", aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
        ax[i, 0].set_title(f"joint mean (noise={nv})")
        ax[i, 0].set_xticks(np.arange(len(drops)))
        ax[i, 0].set_xticklabels([f"{d:.3f}" for d in drops], rotation=25, ha="right")
        ax[i, 0].set_yticks(np.arange(len(edges)))
        ax[i, 0].set_yticklabels([f"{e:.2f}" for e in edges])
        ax[i, 0].set_xlabel("drop_prob")
        ax[i, 0].set_ylabel("edge_aniso")

        im1 = ax[i, 1].imshow(z_rob, origin="lower", aspect="auto", vmin=0.0, vmax=0.5, cmap="magma")
        ax[i, 1].set_title(f"robustness mean (noise={nv})")
        ax[i, 1].set_xticks(np.arange(len(drops)))
        ax[i, 1].set_xticklabels([f"{d:.3f}" for d in drops], rotation=25, ha="right")
        ax[i, 1].set_yticks(np.arange(len(edges)))
        ax[i, 1].set_yticklabels([f"{e:.2f}" for e in edges])
        ax[i, 1].set_xlabel("drop_prob")
        ax[i, 1].set_ylabel("edge_aniso")

    fig.colorbar(im0, ax=ax[:, 0], fraction=0.035, pad=0.02)
    fig.colorbar(im1, ax=ax[:, 1], fraction=0.035, pad=0.02)
    fig.suptitle("v2.1 parameter maps", y=1.01)
    f2 = figs_dir / "phi1_phi2_v21_parameter_heatmaps.png"
    fig.savefig(f2, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("[INFO] Wrote:", f2)

    # Figure 3: top-10 table-like bar chart (robustness sorted).
    top = sorted(records, key=lambda r: r["twisted_robust_mean"], reverse=True)[:10]
    labels = [
        f"e={r['params']['edge_aniso']:.2f}, d={r['params']['drop_prob']:.3f}, n={r['params']['onsite_noise']:.2f}"
        for r in top
    ]
    top_joint = np.array([r["twisted_joint_mean"] for r in top], dtype=float)
    top_rob = np.array([r["twisted_robust_mean"] for r in top], dtype=float)

    y = np.arange(len(top))
    plt.figure(figsize=(10.2, 5.6))
    h = 0.38
    plt.barh(y + h / 2, top_joint, height=h, label="joint")
    plt.barh(y - h / 2, top_rob, height=h, label="robustness")
    plt.axvline(0.35, ls="--", lw=1.0, color="tab:blue", alpha=0.8)
    plt.axvline(0.20, ls="--", lw=1.0, color="tab:orange", alpha=0.8)
    plt.yticks(y, labels)
    plt.gca().invert_yaxis()
    plt.xlabel("fraction")
    plt.title("v2.1 top-10 parameter points")
    plt.grid(axis="x", ls=":", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    f3 = figs_dir / "phi1_phi2_v21_top10.png"
    plt.savefig(f3, dpi=180, bbox_inches="tight")
    plt.close()
    print("[INFO] Wrote:", f3)

    meta = {
        "go_count": s["go_count"],
        "grid_size": s["v21_settings"]["grid_size"],
        "best_record": s["best_record"],
    }
    out = data_dir / "phi1_phi2_emergence_v21_plot_meta.json"
    out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("[INFO] Wrote:", out)


if __name__ == "__main__":
    main()
