#!/usr/bin/env python3
"""Plot v2.3 capture diagnostics."""

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

    s = json.loads((data_dir / "phi1_phi2_emergence_v23_capture_summary.json").read_text())
    rows = s["eta_scan"]

    eta = np.array([r["eta"] for r in rows], dtype=float)
    m1 = np.array([r["mode1_capture_best_mean"] for r in rows], dtype=float)
    m13 = np.array([r["mode13_capture_best_mean"] for r in rows], dtype=float)
    mu = np.array([r["mode1_capture_uniform_mean"] for r in rows], dtype=float)

    plt.figure(figsize=(7.8, 4.8))
    plt.plot(eta, m1, "-o", label="mode-1 best source")
    plt.plot(eta, m13, "-o", label="modes 1-3 best source")
    plt.plot(eta, mu, "--s", label="mode-1 uniform source")
    plt.axhline(0.40, ls="--", lw=1.0, color="tab:blue", alpha=0.8)
    plt.axhline(0.70, ls="--", lw=1.0, color="tab:orange", alpha=0.8)
    plt.xlabel("inertia eta")
    plt.ylabel("capture fraction")
    plt.title("v2.3 topology-aware capture recovery")
    plt.grid(ls=":", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    f1 = figs_dir / "phi1_phi2_v23_capture_vs_eta.png"
    plt.savefig(f1, dpi=180, bbox_inches="tight")
    plt.close()
    print("[INFO] Wrote:", f1)

    meta = {
        "best_eta": s["best_eta_record"]["eta"],
        "mode1_capture_recovered": s["readout"]["mode1_capture_recovered"],
        "mode13_capture_recovered": s["readout"]["mode13_capture_recovered"],
    }
    out = data_dir / "phi1_phi2_emergence_v23_plot_meta.json"
    out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("[INFO] Wrote:", out)


if __name__ == "__main__":
    main()
