#!/usr/bin/env python3
"""Plot emergence scan summaries for internal decision-making."""

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

    scan = json.loads((data_dir / "phi1_phi2_emergence_scan_summary.json").read_text())
    cal = json.loads((data_dir / "phi1_phi2_emergence_calibration_summary.json").read_text())
    cls = json.loads((data_dir / "phi1_phi2_emergence_classification.json").read_text())

    fam_scan = {r["family"]: r for r in scan["family_results"]}
    fam_cal = {r["family"]: r for r in cal["mapped_families"]}
    families = list(fam_scan.keys())

    m2_ref = float(cal["assumptions"]["m2_ref_eV"])

    # Figure 1: calibrated mass scales.
    m2 = np.array([fam_cal[f]["m2_eV_mean_calibrated"] for f in families], dtype=float)
    m2_lo = np.array([fam_cal[f]["m2_eV_window_norm_only"][0] for f in families], dtype=float)
    m2_hi = np.array([fam_cal[f]["m2_eV_window_norm_only"][1] for f in families], dtype=float)
    x = np.arange(len(families))
    plt.figure(figsize=(7.4, 4.6))
    plt.errorbar(x, m2, yerr=[m2 - m2_lo, m2_hi - m2], fmt="o", capsize=3)
    plt.axhline(m2_ref, ls="--", lw=1.0, color="k", label=r"$10^{-22}$ eV reference")
    plt.yscale("log")
    plt.xticks(x, families, rotation=20, ha="right")
    plt.ylabel(r"$m_2$ calibrated (eV)")
    plt.title("Emergent mass scale by family")
    plt.grid(which="both", axis="y", ls=":", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    f1 = figs_dir / "phi1_phi2_emergent_mass_by_family.png"
    plt.savefig(f1, dpi=180, bbox_inches="tight")
    plt.close()
    print("[INFO] Wrote:", f1)

    # Figure 2: support/collectiveness.
    support = np.array([fam_cal[f]["support_index"] for f in families], dtype=float)
    part_mean = np.array([fam_scan[f]["participation_mean"] for f in families], dtype=float)
    fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.4))
    ax[0].bar(x, support)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(families, rotation=20, ha="right")
    ax[0].set_ylabel("support index")
    ax[0].set_title("Support/robustness index")
    ax[0].grid(axis="y", ls=":", alpha=0.4)

    ax[1].bar(x, part_mean)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(families, rotation=20, ha="right")
    ax[1].set_ylabel("participation mean")
    ax[1].set_title("Collectiveness proxy")
    ax[1].grid(axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    f2 = figs_dir / "phi1_phi2_support_collectiveness.png"
    fig.savefig(f2, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("[INFO] Wrote:", f2)

    # Figure 3: dashboard-style score.
    joint = np.array([fam_scan[f]["joint_fraction"] for f in families], dtype=float)
    robust = np.array([fam_scan[f]["robustness_pass_fraction"] for f in families], dtype=float)
    plt.figure(figsize=(7.4, 4.6))
    w = 0.35
    plt.bar(x - w / 2, joint, width=w, label="joint fraction")
    plt.bar(x + w / 2, robust, width=w, label="robustness fraction")
    plt.xticks(x, families, rotation=20, ha="right")
    plt.ylabel("fraction")
    plt.title("ULDM-compatibility diagnostics")
    plt.grid(axis="y", ls=":", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    f3 = figs_dir / "phi1_phi2_dashboard_scores.png"
    plt.savefig(f3, dpi=180, bbox_inches="tight")
    plt.close()
    print("[INFO] Wrote:", f3)

    # Figure 4: low-spectrum example from best family in classification.
    sel = cls["selected_family"]
    eig = np.array(fam_scan[sel]["best_sample_by_m2_proxy"]["light_mode"]["lambda0"])
    # Keep simple with bar summary for best sample metrics.
    best = fam_scan[sel]["best_sample_by_m2_proxy"]["light_mode"]
    keys = ["lambda0", "epsilon_soft", "gap_ratio_12", "support_index", "m2_proxy"]
    vals = np.array([float(best[k]) for k in keys], dtype=float)
    plt.figure(figsize=(7.6, 4.6))
    plt.bar(np.arange(len(keys)), vals)
    plt.xticks(np.arange(len(keys)), keys, rotation=20, ha="right")
    plt.yscale("log")
    plt.title(f"Best-sample diagnostics ({sel})")
    plt.grid(axis="y", ls=":", alpha=0.4)
    plt.tight_layout()
    f4 = figs_dir / "phi1_phi2_best_sample_diagnostics.png"
    plt.savefig(f4, dpi=180, bbox_inches="tight")
    plt.close()
    print("[INFO] Wrote:", f4)

    # Save one compact plot metadata JSON for quick reading.
    plot_meta = {
        "selected_family": sel,
        "classification": cls["category"],
        "families": families,
    }
    out_meta = data_dir / "phi1_phi2_plot_meta.json"
    out_meta.write_text(json.dumps(plot_meta, indent=2), encoding="utf-8")
    print("[INFO] Wrote:", out_meta)


if __name__ == "__main__":
    main()
