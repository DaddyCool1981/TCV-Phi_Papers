#!/usr/bin/env python3
"""Plot focused v2 Phi1->Phi2 emergence diagnostics."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

REPO_ROOT = Path(__file__).resolve().parents[3]
mpl_dir = REPO_ROOT / ".tmp_mpl"
mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    data_dir = REPO_ROOT / "papers" / "paper-08" / "data"
    figs_dir = REPO_ROOT / "papers" / "paper-08" / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    s = json.loads((data_dir / "phi1_phi2_emergence_v2_summary.json").read_text())
    points = s["local_sensitivity"]["points"]
    scale = s["scale_scan"]
    ms = s["multi_seed"]["stats"]

    # Figure 1: local sensitivity (twisted_ring).
    labels = [p["label"] for p in points]
    joint = np.array([p["twisted_ring"]["joint_fraction"] for p in points], dtype=float)
    robust = np.array([p["twisted_ring"]["robustness_pass_fraction"] for p in points], dtype=float)
    x = np.arange(len(points))
    plt.figure(figsize=(12.0, 4.8))
    w = 0.42
    plt.bar(x - w / 2, joint, width=w, label="joint")
    plt.bar(x + w / 2, robust, width=w, label="robustness")
    plt.axhline(0.35, ls="--", lw=1.0, color="tab:blue", alpha=0.8)
    plt.axhline(0.20, ls="--", lw=1.0, color="tab:orange", alpha=0.8)
    plt.xticks(x, labels, rotation=70, ha="right")
    plt.ylabel("fraction")
    plt.title("v2 local sensitivity around twisted_ring baseline")
    plt.legend()
    plt.grid(axis="y", ls=":", alpha=0.4)
    plt.tight_layout()
    f1 = figs_dir / "phi1_phi2_v2_local_sensitivity.png"
    plt.savefig(f1, dpi=180, bbox_inches="tight")
    plt.close()
    print("[INFO] Wrote:", f1)

    # Figure 2: scale scan comparison.
    nvals = np.array([e["N"] for e in scale], dtype=int)
    tw_joint = np.array([e["twisted_ring"]["joint_fraction"] for e in scale], dtype=float)
    tw_rob = np.array([e["twisted_ring"]["robustness_pass_fraction"] for e in scale], dtype=float)
    ring_joint = np.array([e["ring"]["joint_fraction"] for e in scale], dtype=float)
    ring_rob = np.array([e["ring"]["robustness_pass_fraction"] for e in scale], dtype=float)
    plt.figure(figsize=(8.0, 4.8))
    plt.plot(nvals, tw_joint, "-o", label="twisted joint")
    plt.plot(nvals, tw_rob, "-o", label="twisted robust")
    plt.plot(nvals, ring_joint, "--s", label="ring joint")
    plt.plot(nvals, ring_rob, "--s", label="ring robust")
    plt.axhline(0.35, ls=":", lw=1.0, color="tab:blue", alpha=0.7)
    plt.axhline(0.20, ls=":", lw=1.0, color="tab:orange", alpha=0.7)
    plt.xlabel("N")
    plt.ylabel("fraction")
    plt.title("v2 scale scan: twisted_ring vs ring")
    plt.grid(ls=":", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    f2 = figs_dir / "phi1_phi2_v2_scale_scan.png"
    plt.savefig(f2, dpi=180, bbox_inches="tight")
    plt.close()
    print("[INFO] Wrote:", f2)

    # Figure 3: multi-seed CI bars.
    names = ["twisted joint", "twisted robust", "ring joint", "ring robust"]
    means = np.array(
        [
            ms["twisted_joint_mean"],
            ms["twisted_robust_mean"],
            ms["ring_joint_mean"],
            ms["ring_robust_mean"],
        ],
        dtype=float,
    )
    lows = np.array(
        [
            ms["twisted_joint_p16_p84"][0],
            ms["twisted_robust_p16_p84"][0],
            ms["ring_joint_mean"],
            ms["ring_robust_mean"],
        ],
        dtype=float,
    )
    highs = np.array(
        [
            ms["twisted_joint_p16_p84"][1],
            ms["twisted_robust_p16_p84"][1],
            ms["ring_joint_mean"],
            ms["ring_robust_mean"],
        ],
        dtype=float,
    )
    err = np.vstack([means - lows, highs - means])
    xx = np.arange(len(names))
    plt.figure(figsize=(7.6, 4.6))
    plt.errorbar(xx, means, yerr=err, fmt="o", capsize=3)
    plt.axhline(0.35, ls="--", lw=1.0, color="tab:blue", alpha=0.8, label="joint target")
    plt.axhline(0.20, ls="--", lw=1.0, color="tab:orange", alpha=0.8, label="robust target")
    plt.xticks(xx, names, rotation=20, ha="right")
    plt.ylabel("fraction")
    plt.title("v2 multi-seed diagnostics (mean with p16-p84)")
    plt.grid(axis="y", ls=":", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    f3 = figs_dir / "phi1_phi2_v2_multiseed_ci.png"
    plt.savefig(f3, dpi=180, bbox_inches="tight")
    plt.close()
    print("[INFO] Wrote:", f3)

    # Figure 4: calibrated mass vs N.
    tw_m2 = np.array([e["twisted_m2_eV_calibrated"] for e in scale], dtype=float)
    ring_m2 = np.array([e["ring_m2_eV_calibrated"] for e in scale], dtype=float)
    plt.figure(figsize=(8.0, 4.8))
    plt.plot(nvals, tw_m2, "-o", label="twisted_ring")
    plt.plot(nvals, ring_m2, "--s", label="ring")
    plt.axhline(1.0e-22, ls="--", lw=1.0, color="k", label=r"$10^{-22}$ eV")
    plt.yscale("log")
    plt.xlabel("N")
    plt.ylabel(r"calibrated $m_2$ (eV)")
    plt.title("v2 calibrated mass stability vs scale")
    plt.grid(which="both", axis="y", ls=":", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    f4 = figs_dir / "phi1_phi2_v2_mass_vs_scale.png"
    plt.savefig(f4, dpi=180, bbox_inches="tight")
    plt.close()
    print("[INFO] Wrote:", f4)

    meta = {
        "overall_go": s["decision"]["overall_go"],
        "decision": s["decision"],
    }
    out_meta = data_dir / "phi1_phi2_emergence_v2_plot_meta.json"
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("[INFO] Wrote:", out_meta)


if __name__ == "__main__":
    main()
