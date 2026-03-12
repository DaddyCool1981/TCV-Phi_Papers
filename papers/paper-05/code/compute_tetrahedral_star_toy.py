#!/usr/bin/env python3
"""Compute exploratory tetrahedral-star flavour toy diagnostics."""

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

from tcvphi.flavor_tetrahedral_star import (  # noqa: E402
    StarParams,
    evaluate_star_point,
    evaluate_star_point_pheno_v2,
    run_star_scan,
    run_star_targeted_scan,
    run_star_targeted_scan_pheno_v2,
)


def _serialize_point(rec: dict) -> dict:
    return {
        "params": rec["params"],
        "controls": rec["controls"],
        "evals_site": np.asarray(rec["evals_site"]).tolist(),
        "evals_adapted": np.asarray(rec["evals_adapted"]).tolist(),
        "triplet": rec["triplet"],
        "angles_deg": rec["angles_deg"],
        "natural": bool(rec["natural"]),
        "angle_targets_ok": bool(rec["angle_targets_ok"]),
    }


def main() -> None:
    out_data = REPO_ROOT / "papers" / "paper-05" / "data"
    out_figs = REPO_ROOT / "papers" / "paper-05" / "figs"
    out_data.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    sym = StarParams(wc2=1.30, wb2=1.00, k=0.22)
    deformed = StarParams(
        wc2=1.30,
        wb2=1.00,
        k=0.22,
        delta_w3=0.08,
        delta_k1=0.05,
        delta_k4=-0.06,
        k12=0.006,
        k34=0.010,
    )

    rec_sym = evaluate_star_point(sym, coupling_boost=3.0, isolated_weight=0.6, epsilon_l=0.01)
    rec_def = evaluate_star_point(deformed, coupling_boost=3.8, isolated_weight=0.7, epsilon_l=0.01)
    rec_def_v2 = evaluate_star_point_pheno_v2(
        deformed,
        nu_branch_weight=0.78,
        nu_bridge=1.5,
        epsilon_l=8.0e-3,
        cl_cs_weight=1.2,
        cl_branch_suppression=0.18,
    )
    scan = run_star_scan(base=sym, n_samples=700, seed=20260312)
    scan_targeted = run_star_targeted_scan(base=sym, n_samples=3000, seed=20260312)
    scan_targeted_v2 = run_star_targeted_scan_pheno_v2(base=sym, n_samples=5000, seed=20260313)

    theta12 = np.asarray(scan["angles_deg_all"]["theta12"], dtype=float)
    theta13 = np.asarray(scan["angles_deg_all"]["theta13"], dtype=float)
    theta23 = np.asarray(scan["angles_deg_all"]["theta23"], dtype=float)

    records = scan["records"]
    trip_spread = np.array([float(r["triplet"]["triplet_spread_abs"]) for r in records], dtype=float)
    pair_ratio = np.array([float(r["triplet"]["triplet_pair_ratio"]) for r in records], dtype=float)
    natural = np.array([bool(r["natural"]) for r in records], dtype=bool)
    target = np.array([bool(r["angle_targets_ok"]) for r in records], dtype=bool)

    summary = {
        "status": "exploratory cohesive flavour toy: tetrahedral-star geometry",
        "baseline_symmetric": _serialize_point(rec_sym),
        "baseline_deformed": _serialize_point(rec_def),
        "scan_summary": {
            "n_samples": int(scan["n_samples"]),
            "n_natural": int(scan["n_natural"]),
            "n_target_like": int(scan["n_target_like"]),
            "fractions": scan["fractions"],
            "theta_means": {
                "theta12": float(np.mean(theta12)),
                "theta13": float(np.mean(theta13)),
                "theta23": float(np.mean(theta23)),
            },
            "triplet_spread_mean": float(np.mean(trip_spread)),
            "triplet_pair_ratio_mean": float(np.mean(pair_ratio)),
        },
        "best_candidates": scan["best_candidates"],
        "targeted_scan_summary": {
            "n_samples": int(scan_targeted["n_samples"]),
            "n_natural": int(scan_targeted["n_natural"]),
            "n_target_like_strict": int(scan_targeted["n_target_like_strict"]),
            "n_target_like_relaxed": int(scan_targeted["n_target_like_relaxed"]),
            "fractions": scan_targeted["fractions"],
        },
        "pheno_v2_baseline_deformed": _serialize_point(rec_def_v2),
        "pheno_v2_targeted_scan_summary": {
            "n_samples": int(scan_targeted_v2["n_samples"]),
            "n_natural": int(scan_targeted_v2["n_natural"]),
            "n_target_like_strict": int(scan_targeted_v2["n_target_like_strict"]),
            "n_target_like_relaxed": int(scan_targeted_v2["n_target_like_relaxed"]),
            "fractions": scan_targeted_v2["fractions"],
        },
        "notes": [
            "Symmetric star exhibits center-coupled symmetric branch mode and a branch-triplet subspace.",
            "Mild deformations split the triplet and can realize a quasi-degenerate pair.",
            "pheno_v2 tests a constrained center/branch reduction to avoid unconstrained coefficient tuning.",
            "This remains an exploratory geometry test, not a precision PMNS fit.",
        ],
    }

    out_json = out_data / "tetrahedral_star_toy_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[INFO] Wrote:", out_json)

    out_json_targeted = out_data / "tetrahedral_star_targeted_scan_summary.json"
    out_json_targeted.write_text(json.dumps(scan_targeted, indent=2), encoding="utf-8")
    print("[INFO] Wrote:", out_json_targeted)

    out_json_targeted_v2 = out_data / "tetrahedral_star_targeted_scan_pheno_v2_summary.json"
    out_json_targeted_v2.write_text(json.dumps(scan_targeted_v2, indent=2), encoding="utf-8")
    print("[INFO] Wrote:", out_json_targeted_v2)

    out_npz = out_data / "tetrahedral_star_scan.npz"
    np.savez(
        out_npz,
        theta12=theta12,
        theta13=theta13,
        theta23=theta23,
        triplet_spread=trip_spread,
        triplet_pair_ratio=pair_ratio,
        natural_mask=natural.astype(int),
        target_mask=target.astype(int),
        evals_sym=np.asarray(rec_sym["evals_adapted"], dtype=float),
        evals_deformed=np.asarray(rec_def["evals_adapted"], dtype=float),
    )
    print("[INFO] Wrote:", out_npz)

    # Figure 1: spectrum comparison
    x = np.arange(5)
    plt.figure(figsize=(6.6, 4.0))
    plt.plot(x, rec_sym["evals_adapted"], "o-", label="symmetric")
    plt.plot(x, rec_def["evals_adapted"], "s--", label="deformed")
    plt.xticks(x, ["m1", "m2", "m3", "m4", "m5"])
    plt.ylabel("eigenvalue (toy units)")
    plt.title("Tetrahedral-star adapted spectrum")
    plt.grid(ls=":", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    f1 = out_figs / "tetrahedral_star_spectrum_comparison.png"
    plt.savefig(f1, dpi=180, bbox_inches="tight")
    plt.close()
    print("[INFO] Wrote:", f1)

    # Figure 2: triplet splitting proxy
    plt.figure(figsize=(6.6, 4.2))
    plt.scatter(trip_spread, pair_ratio, s=12, alpha=0.35, label="all")
    if np.any(natural):
        plt.scatter(trip_spread[natural], pair_ratio[natural], s=14, alpha=0.5, label="natural")
    if np.any(target):
        plt.scatter(trip_spread[target], pair_ratio[target], s=18, alpha=0.8, label="target-like")
    plt.xlabel("triplet spread (abs)")
    plt.ylabel("closest/farthest split ratio")
    plt.title("Branch triplet splitting under deformation")
    plt.grid(ls=":", alpha=0.4)
    plt.legend(fontsize=8)
    plt.tight_layout()
    f2 = out_figs / "tetrahedral_star_triplet_splitting.png"
    plt.savefig(f2, dpi=180, bbox_inches="tight")
    plt.close()
    print("[INFO] Wrote:", f2)

    # Figure 3: theta23 vs deformation proxy
    plt.figure(figsize=(6.6, 4.2))
    plt.scatter(trip_spread, theta23, s=12, alpha=0.35, label="all")
    if np.any(target):
        plt.scatter(trip_spread[target], theta23[target], s=20, alpha=0.85, label="target-like")
    plt.axhline(45.0, color="k", lw=1.0, ls="--")
    plt.xlabel("triplet spread (abs)")
    plt.ylabel(r"$\theta_{23}$ (deg)")
    plt.title(r"$\theta_{23}$ response to triplet splitting")
    plt.grid(ls=":", alpha=0.4)
    plt.legend(fontsize=8)
    plt.tight_layout()
    f3 = out_figs / "tetrahedral_star_theta23_vs_split.png"
    plt.savefig(f3, dpi=180, bbox_inches="tight")
    plt.close()
    print("[INFO] Wrote:", f3)

    # Figure 4: scatter (theta12, theta13)
    plt.figure(figsize=(6.2, 4.2))
    plt.scatter(theta12, theta13, s=12, alpha=0.35, label="all")
    if np.any(target):
        plt.scatter(theta12[target], theta13[target], s=20, alpha=0.85, label="target-like")
    plt.xlabel(r"$\theta_{12}$ (deg)")
    plt.ylabel(r"$\theta_{13}$ (deg)")
    plt.title(r"Tetrahedral-star scan in $(\theta_{12},\theta_{13})$")
    plt.grid(ls=":", alpha=0.4)
    plt.legend(fontsize=8)
    plt.tight_layout()
    f4 = out_figs / "tetrahedral_star_theta12_theta13_scatter.png"
    plt.savefig(f4, dpi=180, bbox_inches="tight")
    plt.close()
    print("[INFO] Wrote:", f4)

    # Figure 5: top candidate angles.
    best = scan_targeted["best_candidates"][:8]
    if best:
        idx = np.arange(len(best))
        t12b = np.array([float(b["angles_deg"]["theta12"]) for b in best], dtype=float)
        t13b = np.array([float(b["angles_deg"]["theta13"]) for b in best], dtype=float)
        t23b = np.array([float(b["angles_deg"]["theta23"]) for b in best], dtype=float)
        plt.figure(figsize=(7.0, 4.2))
        plt.plot(idx, t12b, "o-", label=r"$\theta_{12}$")
        plt.plot(idx, t13b, "s-", label=r"$\theta_{13}$")
        plt.plot(idx, t23b, "^-", label=r"$\theta_{23}$")
        plt.xticks(idx, [f"#{i+1}" for i in idx])
        plt.ylabel("angle (deg)")
        plt.title("Top tetrahedral-star candidates")
        plt.grid(ls=":", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        f5 = out_figs / "tetrahedral_star_top_candidates.png"
        plt.savefig(f5, dpi=180, bbox_inches="tight")
        plt.close()
        print("[INFO] Wrote:", f5)

    # Figure 6: comparison with previous toys if available.
    ref_old = out_data / "ccphi2_flavour_toy.json"
    ref_tet = out_data / "tetrahedral_flavour_toy_summary.json"
    labels = [r"$\theta_{12}$", r"$\theta_{13}$", r"$\theta_{23}$"]
    xloc = np.arange(3)
    plt.figure(figsize=(6.8, 4.2))
    if ref_old.exists():
        d_old = json.loads(ref_old.read_text(encoding="utf-8"))
        a = d_old.get("pmns_angles_deg", {})
        y_old = [float(a.get("theta12", np.nan)), float(a.get("theta13", np.nan)), float(a.get("theta23", np.nan))]
        plt.plot(xloc, y_old, "o--", label="previous pyramidal toy")
    if ref_tet.exists():
        d_tet = json.loads(ref_tet.read_text(encoding="utf-8"))
        y_tet = [
            float(d_tet["baseline_deformed"]["angles_deg"]["theta12"]),
            float(d_tet["baseline_deformed"]["angles_deg"]["theta13"]),
            float(d_tet["baseline_deformed"]["angles_deg"]["theta23"]),
        ]
        plt.plot(xloc, y_tet, "s-.", label="tetrahedral apex/base toy")
    y_star = [
        float(rec_def["angles_deg"]["theta12"]),
        float(rec_def["angles_deg"]["theta13"]),
        float(rec_def["angles_deg"]["theta23"]),
    ]
    plt.plot(xloc, y_star, "d-", label="tetrahedral-star toy")
    plt.xticks(xloc, labels)
    plt.ylabel("angle (deg)")
    plt.title("Toy-geometry comparison")
    plt.grid(axis="y", ls=":", alpha=0.4)
    plt.legend(fontsize=8)
    plt.tight_layout()
    f6 = out_figs / "tetrahedral_star_vs_previous_toys.png"
    plt.savefig(f6, dpi=180, bbox_inches="tight")
    plt.close()
    print("[INFO] Wrote:", f6)


if __name__ == "__main__":
    main()
