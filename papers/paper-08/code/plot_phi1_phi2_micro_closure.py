#!/usr/bin/env python3
"""Plot micro-closure diagnostics."""

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

    s = json.loads((data_dir / "phi1_phi2_micro_closure_summary.json").read_text())
    tw = s["twisted_ring"]
    rg = s["ring"]

    # Fig1 capture
    tw_names = list(tw["capture_mode13"].keys())
    tw_vals = np.array([tw["capture_mode13"][k] for k in tw_names], dtype=float)
    plt.figure(figsize=(7.6, 4.6))
    plt.bar(np.arange(len(tw_names)), tw_vals)
    plt.axhline(0.75, ls="--", lw=1.0, color="tab:red")
    plt.xticks(np.arange(len(tw_names)), tw_names, rotation=25, ha="right")
    plt.ylabel("capture modes 1-3")
    plt.title("Micro-closure source capture (twisted ring)")
    plt.grid(axis="y", ls=":", alpha=0.4)
    plt.tight_layout()
    f1 = figs_dir / "phi1_phi2_micro_capture.png"
    plt.savefig(f1, dpi=180, bbox_inches="tight")
    plt.close()
    print("[INFO] Wrote:", f1)

    # Fig2 projection error comparison
    labels = ["mode1 trunc", "mode13 trunc"]
    tw_err = [tw["projection_error_mode1"]["rel_err_mean"], tw["projection_error_mode13"]["rel_err_mean"]]
    rg_err = [rg["projection_error_mode1"]["rel_err_mean"], rg["projection_error_mode13"]["rel_err_mean"]]
    x = np.arange(len(labels))
    w = 0.35
    plt.figure(figsize=(7.4, 4.6))
    plt.bar(x - w / 2, tw_err, width=w, label="twisted")
    plt.bar(x + w / 2, rg_err, width=w, label="ring")
    plt.axhline(0.20, ls="--", lw=1.0, color="tab:red")
    plt.xticks(x, labels)
    plt.ylabel("relative error mean")
    plt.title("Low-mode projection quality")
    plt.grid(axis="y", ls=":", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    f2 = figs_dir / "phi1_phi2_micro_projection_error.png"
    plt.savefig(f2, dpi=180, bbox_inches="tight")
    plt.close()
    print("[INFO] Wrote:", f2)

    # Fig3 check dashboard
    checks = s["checks"]
    names = list(checks.keys())
    vals = np.array([1.0 if checks[k] else 0.0 for k in names], dtype=float)
    plt.figure(figsize=(8.4, 4.8))
    plt.bar(np.arange(len(names)), vals)
    plt.ylim(-0.05, 1.05)
    plt.xticks(np.arange(len(names)), names, rotation=25, ha="right")
    plt.title(f"Micro-closure checks ({s['verdict']})")
    plt.ylabel("pass=1/fail=0")
    plt.grid(axis="y", ls=":", alpha=0.4)
    plt.tight_layout()
    f3 = figs_dir / "phi1_phi2_micro_checks.png"
    plt.savefig(f3, dpi=180, bbox_inches="tight")
    plt.close()
    print("[INFO] Wrote:", f3)

    meta = {"verdict": s["verdict"], "checks": checks}
    out = data_dir / "phi1_phi2_micro_closure_plot_meta.json"
    out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("[INFO] Wrote:", out)


if __name__ == "__main__":
    main()
