#!/usr/bin/env python3
"""Compute exploratory spherical-shell flavour toy diagnostics."""

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

from tcvphi.flavor_spherical_shell import (  # noqa: E402
    SphericalShellParams,
    evaluate_spherical_point,
    run_spherical_scan,
)


def _serialize(rec: dict) -> dict:
    return {
        "params": rec["params"],
        "controls": rec["controls"],
        "evals": np.asarray(rec["evals"]).tolist(),
        "shell_structure": rec["shell_structure"],
        "angles_deg": rec["angles_deg"],
        "natural": bool(rec["natural"]),
        "angle_targets_ok": bool(rec["angle_targets_ok"]),
    }


def main() -> None:
    out_data = REPO_ROOT / "papers" / "paper-05" / "data"
    out_figs = REPO_ROOT / "papers" / "paper-05" / "figs"
    out_data.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    sym = SphericalShellParams(wc2=1.35, ws2=1.00, kc=0.20, ks=0.12)
    deform = SphericalShellParams(
        wc2=1.35,
        ws2=1.00,
        kc=0.20,
        ks=0.12,
        d0=0.06,
        d3=-0.05,
        a1=0.08,
        a4=-0.07,
        shell_scale=0.05,
    )

    rec_sym = evaluate_spherical_point(sym)
    rec_def = evaluate_spherical_point(deform)
    scan = run_spherical_scan(base=sym, n_samples=5000, seed=20260312)

    summary = {
        "status": "exploratory cohesive flavour toy: spherical-shell octahedral cluster",
        "baseline_symmetric": _serialize(rec_sym),
        "baseline_deformed": _serialize(rec_def),
        "scan_summary": {
            "n_samples": scan["n_samples"],
            "n_natural": scan["n_natural"],
            "n_target_like_strict": scan["n_target_like_strict"],
            "n_target_like_relaxed": scan["n_target_like_relaxed"],
            "fractions": scan["fractions"],
        },
        "best_candidates": scan["best_candidates"],
        "notes": [
            "Discrete spherical proxy uses center + octahedral shell graph.",
            "Symmetry breaking is mild (<=20%) and tests singlet+doublet shell response.",
            "This remains exploratory and additive to other Paper-05 toys.",
        ],
    }

    out_json = out_data / "spherical_flavour_toy_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[INFO] Wrote:", out_json)

    # Figure 1: symmetric vs deformed spectrum.
    plt.figure(figsize=(6.6, 4.0))
    plt.plot(np.arange(7), rec_sym["evals"], "o-", label="symmetric")
    plt.plot(np.arange(7), rec_def["evals"], "s--", label="deformed")
    plt.xticks(np.arange(7), [f"m{i+1}" for i in range(7)])
    plt.ylabel("eigenvalue (toy units)")
    plt.title("Spherical-shell toy spectrum")
    plt.grid(ls=":", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    f1 = out_figs / "spherical_flavour_spectrum.png"
    plt.savefig(f1, dpi=180, bbox_inches="tight")
    plt.close()
    print("[INFO] Wrote:", f1)

    # Figure 2: top candidate angles.
    best = scan["best_candidates"][:10]
    if best:
        idx = np.arange(len(best))
        t12 = np.array([float(b["angles_deg"]["theta12"]) for b in best], dtype=float)
        t13 = np.array([float(b["angles_deg"]["theta13"]) for b in best], dtype=float)
        t23 = np.array([float(b["angles_deg"]["theta23"]) for b in best], dtype=float)
        plt.figure(figsize=(7.0, 4.2))
        plt.plot(idx, t12, "o-", label=r"$\theta_{12}$")
        plt.plot(idx, t13, "s-", label=r"$\theta_{13}$")
        plt.plot(idx, t23, "^-", label=r"$\theta_{23}$")
        plt.xticks(idx, [f"#{i+1}" for i in idx])
        plt.ylabel("angle (deg)")
        plt.title("Spherical-shell top candidates")
        plt.grid(ls=":", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        f2 = out_figs / "spherical_flavour_top_candidates.png"
        plt.savefig(f2, dpi=180, bbox_inches="tight")
        plt.close()
        print("[INFO] Wrote:", f2)


if __name__ == "__main__":
    main()
