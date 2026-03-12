#!/usr/bin/env python3
"""Compute exploratory torus-ring flavour toy diagnostics."""

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

from tcvphi.flavor_torus_ring import (  # noqa: E402
    TorusRingParams,
    evaluate_ring_point,
    run_ring_scan,
)


def _serialize(rec: dict) -> dict:
    return {
        "params": rec["params"],
        "controls": rec["controls"],
        "evals_site": np.asarray(rec["evals_site"]).tolist(),
        "evals_fourier": np.asarray(rec["evals_fourier"]).tolist(),
        "doublets": rec["doublets"],
        "angles_deg": rec["angles_deg"],
        "natural": bool(rec["natural"]),
        "angle_targets_ok": bool(rec["angle_targets_ok"]),
    }


def main() -> None:
    out_data = REPO_ROOT / "papers" / "paper-05" / "data"
    out_figs = REPO_ROOT / "papers" / "paper-05" / "figs"
    out_data.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    sym = TorusRingParams(w0=1.0, k=0.20)
    deform = TorusRingParams(
        w0=1.0,
        k=0.20,
        d2=0.06,
        d5=-0.05,
        a1=0.08,
        a4=-0.07,
    )

    rec_sym = evaluate_ring_point(sym, use_doublet="m1")
    rec_def = evaluate_ring_point(deform, use_doublet="m1")
    scan = run_ring_scan(base=sym, n_samples=5000, seed=20260312)

    summary = {
        "status": "exploratory cohesive flavour toy: torus-like periodic ring",
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
            "The symmetric ring exhibits Fourier-like sectors with doublet structure.",
            "The test checks if mild anisotropies naturally yield a PMNS-like hierarchy.",
            "This is exploratory and additive to existing Paper-05 toys.",
        ],
    }

    out_json = out_data / "torus_flavour_toy_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[INFO] Wrote:", out_json)

    # Quick visual diagnostics.
    best = scan["best_candidates"]
    if best:
        t12 = np.array([float(b["angles_deg"]["theta12"]) for b in best], dtype=float)
        t13 = np.array([float(b["angles_deg"]["theta13"]) for b in best], dtype=float)
        t23 = np.array([float(b["angles_deg"]["theta23"]) for b in best], dtype=float)
        idx = np.arange(len(best))
        plt.figure(figsize=(7.0, 4.0))
        plt.plot(idx, t12, "o-", label=r"$\theta_{12}$")
        plt.plot(idx, t13, "s-", label=r"$\theta_{13}$")
        plt.plot(idx, t23, "^-", label=r"$\theta_{23}$")
        plt.xticks(idx, [f"#{i+1}" for i in idx])
        plt.ylabel("angle (deg)")
        plt.title("Torus-ring toy top candidates")
        plt.grid(ls=":", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        f1 = out_figs / "torus_flavour_top_candidates.png"
        plt.savefig(f1, dpi=180, bbox_inches="tight")
        plt.close()
        print("[INFO] Wrote:", f1)

    plt.figure(figsize=(6.2, 4.0))
    plt.plot(np.arange(6), rec_sym["evals_fourier"], "o-", label="symmetric")
    plt.plot(np.arange(6), rec_def["evals_fourier"], "s--", label="deformed")
    plt.xticks(np.arange(6), ["m0", "c1", "s1", "c2", "s2", "m3"])
    plt.ylabel("eigenvalue (toy units)")
    plt.title("Torus-ring spectrum in Fourier basis")
    plt.grid(ls=":", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    f2 = out_figs / "torus_flavour_spectrum.png"
    plt.savefig(f2, dpi=180, bbox_inches="tight")
    plt.close()
    print("[INFO] Wrote:", f2)


if __name__ == "__main__":
    main()
