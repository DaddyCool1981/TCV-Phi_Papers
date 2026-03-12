#!/usr/bin/env python3
"""Targeted tetrahedral flavour scan (geometry + response controls)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
mpl_dir = REPO_ROOT / ".tmp_mpl"
mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CORE_LIB = REPO_ROOT / "core" / "lib"
if str(CORE_LIB) not in sys.path:
    sys.path.insert(0, str(CORE_LIB))

from tcvphi.flavor_tetrahedron import TetrahedralParams, run_tetrahedral_targeted_scan  # noqa: E402


def main() -> None:
    out_data = REPO_ROOT / "papers" / "paper-05" / "data"
    out_figs = REPO_ROOT / "papers" / "paper-05" / "figs"
    out_data.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    base = TetrahedralParams(wt2=1.35, wb2=1.00, kt=0.20, kb=0.14)
    scan = run_tetrahedral_targeted_scan(base=base, n_samples=1500, seed=20260312)

    out_json = out_data / "tetrahedral_targeted_scan_summary.json"
    out_json.write_text(json.dumps(scan, indent=2), encoding="utf-8")
    print("[INFO] Wrote:", out_json)

    best = scan.get("best_candidates", [])
    if not best:
        return

    t12 = np.array([b["angles_deg"]["theta12"] for b in best], dtype=float)
    t13 = np.array([b["angles_deg"]["theta13"] for b in best], dtype=float)
    t23 = np.array([b["angles_deg"]["theta23"] for b in best], dtype=float)
    labels = [f"#{i+1}" for i in range(len(best))]
    x = np.arange(len(best))

    plt.figure(figsize=(7.0, 4.0))
    plt.plot(x, t12, "o-", label=r"$\theta_{12}$")
    plt.plot(x, t13, "s-", label=r"$\theta_{13}$")
    plt.plot(x, t23, "^-", label=r"$\theta_{23}$")
    plt.axhspan(20.0, 40.0, color="C0", alpha=0.08)
    plt.axhspan(2.0, 13.0, color="C1", alpha=0.08)
    plt.axhspan(40.0, 50.0, color="C2", alpha=0.08)
    plt.xticks(x, labels)
    plt.ylabel("angle (deg)")
    plt.title("Top targeted tetrahedral candidates")
    plt.grid(ls=":", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    f = out_figs / "tetrahedral_targeted_top_candidates.png"
    plt.savefig(f, dpi=180, bbox_inches="tight")
    plt.close()
    print("[INFO] Wrote:", f)


if __name__ == "__main__":
    main()

