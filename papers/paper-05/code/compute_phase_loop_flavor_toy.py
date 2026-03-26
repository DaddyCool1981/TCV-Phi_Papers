#!/usr/bin/env python3
"""Compute exploratory phase-loop flavour toy diagnostics."""

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

from tcvphi.flavor_phase_loop import (  # noqa: E402
    PhaseLoopParams,
    evaluate_phase_loop_point,
    run_phase_loop_scan,
)


def _serialize(rec: dict) -> dict:
    return {
        "params": rec["params"],
        "controls": rec["controls"],
        "evals": np.asarray(rec["evals"]).tolist(),
        "phase_structure": rec["phase_structure"],
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

    sym = PhaseLoopParams(w0=1.0, k=0.20, phase=0.20 * np.pi)
    deform = PhaseLoopParams(
        w0=1.0,
        k=0.20,
        phase=0.28 * np.pi,
        d2=0.05,
        d5=-0.04,
        a1=0.07,
        a4=-0.06,
    )

    rec_sym = evaluate_phase_loop_point(sym)
    rec_def = evaluate_phase_loop_point(deform)
    scan = run_phase_loop_scan(base=sym, n_samples=5000, seed=20260313)

    summary = {
        "status": "exploratory cohesive flavour toy: phase-loop closed geometry",
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
            "The loop phase is a toy structural variable, not an imported external formalism.",
            "The test checks whether a minimal circulation degree of freedom improves PMNS-like emergence over a plain ring.",
            "Exploratory only, additive to the existing Paper-05 geometry family tests.",
        ],
    }

    out_json = out_data / "phase_loop_flavour_toy_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[INFO] Wrote:", out_json)

    plt.figure(figsize=(6.6, 4.0))
    plt.plot(np.arange(6), rec_sym["evals"], "o-", label="symmetric")
    plt.plot(np.arange(6), rec_def["evals"], "s--", label="deformed")
    plt.xticks(np.arange(6), [f"m{i+1}" for i in range(6)])
    plt.ylabel("eigenvalue (toy units)")
    plt.title("Phase-loop toy spectrum")
    plt.grid(ls=":", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    f1 = out_figs / "phase_loop_flavour_spectrum.png"
    plt.savefig(f1, dpi=180, bbox_inches="tight")
    plt.close()
    print("[INFO] Wrote:", f1)

    best = scan["best_candidates"][:10]
    if best:
        idx = np.arange(len(best))
        t12 = np.array([float(b["angles_deg"]["theta12"]) for b in best], dtype=float)
        t13 = np.array([float(b["angles_deg"]["theta13"]) for b in best], dtype=float)
        t23 = np.array([float(b["angles_deg"]["theta23"]) for b in best], dtype=float)
        phase = np.array([float(b["params"]["phase"]) / np.pi for b in best], dtype=float)

        plt.figure(figsize=(7.2, 4.2))
        plt.plot(idx, t12, "o-", label=r"$\theta_{12}$")
        plt.plot(idx, t13, "s-", label=r"$\theta_{13}$")
        plt.plot(idx, t23, "^-", label=r"$\theta_{23}$")
        plt.xticks(idx, [f"#{i+1}" for i in idx])
        plt.ylabel("angle (deg)")
        plt.title("Phase-loop top candidates")
        plt.grid(ls=":", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        f2 = out_figs / "phase_loop_flavour_top_candidates.png"
        plt.savefig(f2, dpi=180, bbox_inches="tight")
        plt.close()
        print("[INFO] Wrote:", f2)

        plt.figure(figsize=(6.8, 4.0))
        plt.scatter(phase, t23, c=t13, cmap="viridis", s=55)
        plt.xlabel(r"loop phase / $\pi$")
        plt.ylabel(r"$\theta_{23}$ (deg)")
        plt.title(r"Phase-loop: $\theta_{23}$ vs loop phase")
        cb = plt.colorbar()
        cb.set_label(r"$\theta_{13}$ (deg)")
        plt.grid(ls=":", alpha=0.4)
        plt.tight_layout()
        f3 = out_figs / "phase_loop_theta23_vs_phase.png"
        plt.savefig(f3, dpi=180, bbox_inches="tight")
        plt.close()
        print("[INFO] Wrote:", f3)

    # Direct comparison against the current twisted baseline.
    tw = _load_json(out_data / "twisted_torus_flavour_toy_summary.json")
    phase_best = summary["best_candidates"][0]
    tw_best = tw["best_candidates"][0]

    def _score(angles: dict) -> float:
        d12 = (float(angles["theta12"]) - 33.0) / 10.0
        d13 = (float(angles["theta13"]) - 8.6) / 4.0
        d23 = (float(angles["theta23"]) - 45.0) / 7.0
        return float(np.sqrt(d12 * d12 + d13 * d13 + d23 * d23))

    comp = {
        "phase_loop_best": {
            "angles_deg": phase_best["angles_deg"],
            "score_common": _score(phase_best["angles_deg"]),
            "strict_fraction_within_natural": summary["scan_summary"]["fractions"]["strict_within_natural"],
            "phase_over_pi": float(phase_best["params"]["phase"]) / float(np.pi),
        },
        "twisted_torus_best": {
            "angles_deg": tw_best["angles_deg"],
            "score_common": _score(tw_best["angles_deg"]),
            "strict_fraction_within_natural": tw["scan_summary"]["fractions"]["strict_within_natural"],
        },
        "notes": [
            "This is a direct PMNS-level comparison only.",
            "Phi2 bridge and micro-closure for phase_loop still need a dedicated follow-up if the flavour results remain competitive.",
        ],
    }
    comp_path = out_data / "phase_loop_vs_twisted_torus_comparison.json"
    comp_path.write_text(json.dumps(comp, indent=2), encoding="utf-8")
    print("[INFO] Wrote:", comp_path)


if __name__ == "__main__":
    main()
