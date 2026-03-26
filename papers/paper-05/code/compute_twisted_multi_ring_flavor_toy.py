#!/usr/bin/env python3
"""Compute exploratory twisted-multi-ring flavour toy diagnostics."""

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

from tcvphi.flavor_twisted_multi_ring import (  # noqa: E402
    TwistedMultiRingParams,
    evaluate_twisted_multi_ring_point,
    run_twisted_multi_ring_scan,
)


def _serialize(rec: dict) -> dict:
    return {
        "params": rec["params"],
        "controls": rec["controls"],
        "evals": np.asarray(rec["evals"]).tolist(),
        "low_structure": rec["low_structure"],
        "angles_deg": rec["angles_deg"],
        "natural": bool(rec["natural"]),
        "angle_targets_ok": bool(rec["angle_targets_ok"]),
    }


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    out_data = REPO_ROOT / "papers" / "paper-05" / "data"
    out_figs = REPO_ROOT / "papers" / "paper-05" / "figs"
    out_data.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    sym = TwistedMultiRingParams(w0=1.0, k_ring=0.19, k_cross=0.13, twist_shift=1)
    deform = TwistedMultiRingParams(
        w0=1.0,
        k_ring=0.19,
        k_cross=0.13,
        twist_shift=2,
        d2=0.05,
        d7=-0.04,
        d11=0.03,
        ring_scale=0.05,
        cross_scale=-0.04,
    )

    rec_sym = evaluate_twisted_multi_ring_point(sym)
    rec_def = evaluate_twisted_multi_ring_point(deform)
    scan = run_twisted_multi_ring_scan(base=sym, n_samples=5000, seed=20260314)

    summary = {
        "status": "exploratory cohesive flavour toy: twisted multi-ring geometry",
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
            "This toy tests whether multiple twisted closed channels outperform a single twisted loop.",
            "It is exploratory and additive to the existing Paper-05 geometry family tests.",
            "A good PMNS outcome here would justify a dedicated Phi2/micro-closure bridge follow-up.",
        ],
    }

    out_json = out_data / "twisted_multi_ring_flavour_toy_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[INFO] Wrote:", out_json)

    plt.figure(figsize=(6.8, 4.0))
    plt.plot(np.arange(12), rec_sym["evals"], "o-", label="symmetric")
    plt.plot(np.arange(12), rec_def["evals"], "s--", label="deformed")
    plt.xticks(np.arange(0, 12, 1), [f"m{i+1}" for i in range(12)], rotation=30, ha="right")
    plt.ylabel("eigenvalue (toy units)")
    plt.title("Twisted multi-ring toy spectrum")
    plt.grid(ls=":", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    f1 = out_figs / "twisted_multi_ring_flavour_spectrum.png"
    plt.savefig(f1, dpi=180, bbox_inches="tight")
    plt.close()
    print("[INFO] Wrote:", f1)

    best = scan["best_candidates"][:10]
    if best:
        idx = np.arange(len(best))
        t12 = np.array([float(b["angles_deg"]["theta12"]) for b in best], dtype=float)
        t13 = np.array([float(b["angles_deg"]["theta13"]) for b in best], dtype=float)
        t23 = np.array([float(b["angles_deg"]["theta23"]) for b in best], dtype=float)
        shifts = np.array([float(b["params"]["twist_shift"]) for b in best], dtype=float)

        plt.figure(figsize=(7.2, 4.2))
        plt.plot(idx, t12, "o-", label=r"$\theta_{12}$")
        plt.plot(idx, t13, "s-", label=r"$\theta_{13}$")
        plt.plot(idx, t23, "^-", label=r"$\theta_{23}$")
        plt.xticks(idx, [f"#{i+1}" for i in idx])
        plt.ylabel("angle (deg)")
        plt.title("Twisted multi-ring top candidates")
        plt.grid(ls=":", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        f2 = out_figs / "twisted_multi_ring_flavour_top_candidates.png"
        plt.savefig(f2, dpi=180, bbox_inches="tight")
        plt.close()
        print("[INFO] Wrote:", f2)

        plt.figure(figsize=(6.8, 4.0))
        plt.scatter(shifts, t23, c=t13, cmap="viridis", s=55)
        plt.xlabel("twist shift")
        plt.ylabel(r"$\theta_{23}$ (deg)")
        plt.title(r"Twisted multi-ring: $\theta_{23}$ vs twist shift")
        cb = plt.colorbar()
        cb.set_label(r"$\theta_{13}$ (deg)")
        plt.grid(ls=":", alpha=0.4)
        plt.tight_layout()
        f3 = out_figs / "twisted_multi_ring_theta23_vs_shift.png"
        plt.savefig(f3, dpi=180, bbox_inches="tight")
        plt.close()
        print("[INFO] Wrote:", f3)

    tw = _load_json(out_data / "twisted_torus_flavour_toy_summary.json")
    multi_best = summary["best_candidates"][0]
    tw_best = tw["best_candidates"][0]

    def _score(angles: dict) -> float:
        d12 = (float(angles["theta12"]) - 33.0) / 10.0
        d13 = (float(angles["theta13"]) - 8.6) / 4.0
        d23 = (float(angles["theta23"]) - 45.0) / 7.0
        return float(np.sqrt(d12 * d12 + d13 * d13 + d23 * d23))

    comp = {
        "twisted_multi_ring_best": {
            "angles_deg": multi_best["angles_deg"],
            "score_common": _score(multi_best["angles_deg"]),
            "strict_fraction_within_natural": summary["scan_summary"]["fractions"]["strict_within_natural"],
            "twist_shift": int(multi_best["params"]["twist_shift"]),
        },
        "twisted_torus_best": {
            "angles_deg": tw_best["angles_deg"],
            "score_common": _score(tw_best["angles_deg"]),
            "strict_fraction_within_natural": tw["scan_summary"]["fractions"]["strict_within_natural"],
        },
        "notes": [
            "This is a direct PMNS-level comparison only.",
            "If twisted_multi_ring remains competitive here, it deserves a dedicated Phi2 bridge and micro-closure follow-up.",
        ],
    }
    comp_path = out_data / "twisted_multi_ring_vs_twisted_torus_comparison.json"
    comp_path.write_text(json.dumps(comp, indent=2), encoding="utf-8")
    print("[INFO] Wrote:", comp_path)


if __name__ == "__main__":
    main()
