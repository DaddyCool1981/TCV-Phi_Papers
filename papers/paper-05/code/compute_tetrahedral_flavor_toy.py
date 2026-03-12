#!/usr/bin/env python3
"""Run tetrahedral CC-Phi1 flavour toy benchmarks and a mild deformation scan."""

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

from tcvphi.flavor_tetrahedron import (  # noqa: E402
    TetrahedralParams,
    evaluate_tetrahedral_point,
    run_tetrahedral_scan,
)


def _to_serializable_point(point: dict) -> dict:
    return {
        "params": point["params"],
        "evals_site": np.asarray(point["evals_site"]).tolist(),
        "evals_adapted": np.asarray(point["evals_adapted"]).tolist(),
        "doublet": point["doublet"],
        "angles_deg": point["angles_deg"],
        "theta23_geometric_deg": float(point["theta23_geometric_deg"]),
        "natural": bool(point["natural"]),
        "angle_targets_ok": bool(point["angle_targets_ok"]),
        "angle_targets_relaxed_ok": bool(point["angle_targets_relaxed_ok"]),
        "geometry_targets_ok": bool(point["geometry_targets_ok"]),
    }


def main() -> None:
    out_data = REPO_ROOT / "papers" / "paper-05" / "data"
    out_figs = REPO_ROOT / "papers" / "paper-05" / "figs"
    out_data.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    sym = TetrahedralParams(wt2=1.35, wb2=1.00, kt=0.20, kb=0.14)
    # Mild branch-3 and edge asymmetry to lift D1/D2 degeneracy.
    deformed = TetrahedralParams(
        wt2=1.35,
        wb2=1.00,
        kt=0.20,
        kb=0.14,
        delta_w3=0.06,
        delta_kb12=0.02,
        delta_kb23=-0.04,
        delta_kb31=0.03,
        delta_kt3=-0.05,
    )

    sym_res = evaluate_tetrahedral_point(sym)
    def_res = evaluate_tetrahedral_point(deformed)
    scan = run_tetrahedral_scan(base=sym, n_samples=500, seed=20260312)

    theta12 = np.asarray(scan["angles_deg_all"]["theta12"], dtype=float)
    theta13 = np.asarray(scan["angles_deg_all"]["theta13"], dtype=float)
    theta23 = np.asarray(scan["angles_deg_all"]["theta23"], dtype=float)

    records = scan["records"]
    rel_deform = np.array(
        [
            abs(r["params"]["delta_w3"]) / max(abs(r["params"]["wb2"]), 1.0e-14)
            for r in records
        ],
        dtype=float,
    )
    natural_mask = np.array([bool(r["natural"]) for r in records], dtype=bool)
    target_mask = np.array([bool(r["angle_targets_ok"]) for r in records], dtype=bool)
    target_relaxed_mask = np.array([bool(r["angle_targets_relaxed_ok"]) for r in records], dtype=bool)
    geom_target_mask = np.array([bool(r["geometry_targets_ok"]) for r in records], dtype=bool)

    summary = {
        "status": "exploratory tetrahedral toy for CC-Phi1/CC-Phi2 flavour geometry",
        "baseline_symmetric": _to_serializable_point(sym_res),
        "baseline_deformed": _to_serializable_point(def_res),
        "scan_summary": {
            "n_samples": int(scan["n_samples"]),
            "n_natural": int(scan["n_natural"]),
            "n_target_like": int(scan["n_target_like"]),
            "n_target_like_relaxed": int(scan["n_target_like_relaxed"]),
            "n_geometry_target_like": int(scan["n_geometry_target_like"]),
            "fractions": scan["fractions"],
            "theta12_mean": float(np.mean(theta12)),
            "theta13_mean": float(np.mean(theta13)),
            "theta23_mean": float(np.mean(theta23)),
            "theta23_natural_mean": float(np.mean(theta23[natural_mask])) if np.any(natural_mask) else None,
        },
        "notes": [
            "Symmetric tetrahedron exhibits a singlet+doublet pattern in adapted basis.",
            "Mild deformations lift D1-D2 degeneracy and move theta23 away from exact maximal mixing.",
            "This is a geometric viability test, not a precision PMNS fit.",
        ],
    }

    out_json = out_data / "tetrahedral_flavour_toy_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[INFO] Wrote:", out_json)

    out_npz = out_data / "tetrahedral_flavour_scan.npz"
    np.savez(
        out_npz,
        theta12=theta12,
        theta13=theta13,
        theta23=theta23,
        rel_deform=rel_deform,
        natural_mask=natural_mask.astype(int),
        target_mask=target_mask.astype(int),
        target_relaxed_mask=target_relaxed_mask.astype(int),
        geom_target_mask=geom_target_mask.astype(int),
        evals_sym=np.asarray(sym_res["evals_adapted"], dtype=float),
        evals_deformed=np.asarray(def_res["evals_adapted"], dtype=float),
    )
    print("[INFO] Wrote:", out_npz)

    # Figure 1: adapted spectrum, symmetric vs deformed.
    x = np.arange(4)
    plt.figure(figsize=(6.4, 4.0))
    plt.plot(x, sym_res["evals_adapted"], "o-", lw=1.8, label="symmetric")
    plt.plot(x, def_res["evals_adapted"], "s--", lw=1.6, label="deformed")
    plt.xticks(x, ["mode1", "mode2", "mode3", "mode4"])
    plt.ylabel("eigenvalue (toy units)")
    plt.title("Tetrahedral adapted spectrum: singlet + doublet splitting")
    plt.grid(ls=":", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    fig1 = out_figs / "tetrahedral_spectrum_comparison.png"
    plt.savefig(fig1, dpi=180, bbox_inches="tight")
    plt.close()
    print("[INFO] Wrote:", fig1)

    # Figure 2: theta23 vs deformation strength.
    plt.figure(figsize=(6.4, 4.2))
    plt.scatter(rel_deform, theta23, s=12, alpha=0.35, label="all scan points")
    if np.any(natural_mask):
        plt.scatter(rel_deform[natural_mask], theta23[natural_mask], s=16, alpha=0.55, label="natural")
    if np.any(target_mask):
        plt.scatter(rel_deform[target_mask], theta23[target_mask], s=20, alpha=0.85, label="PMNS-target-like")
    if np.any(target_relaxed_mask):
        plt.scatter(
            rel_deform[target_relaxed_mask],
            theta23[target_relaxed_mask],
            s=20,
            alpha=0.85,
            label="PMNS-target-like (relaxed)",
        )
    if np.any(geom_target_mask):
        plt.scatter(rel_deform[geom_target_mask], theta23[geom_target_mask], s=20, alpha=0.85, label="geometry-target-like")
    plt.axhline(45.0, color="k", lw=1.0, ls="--", label=r"$45^\circ$")
    plt.xlabel(r"relative base onsite deformation $|\delta w_3|/w_{b}^2$")
    plt.ylabel(r"$\theta_{23}$ (deg)")
    plt.title(r"Departure of $\theta_{23}$ from maximal mixing")
    plt.grid(ls=":", alpha=0.4)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    fig2 = out_figs / "tetrahedral_theta23_vs_deformation.png"
    plt.savefig(fig2, dpi=180, bbox_inches="tight")
    plt.close()
    print("[INFO] Wrote:", fig2)

    # Figure 3: representative PMNS-like angles.
    labels = [r"$\theta_{12}$", r"$\theta_{13}$", r"$\theta_{23}$"]
    vals_sym = [sym_res["angles_deg"]["theta12"], sym_res["angles_deg"]["theta13"], sym_res["angles_deg"]["theta23"]]
    vals_def = [def_res["angles_deg"]["theta12"], def_res["angles_deg"]["theta13"], def_res["angles_deg"]["theta23"]]
    xloc = np.arange(len(labels))
    width = 0.36
    plt.figure(figsize=(6.4, 4.0))
    plt.bar(xloc - width / 2.0, vals_sym, width=width, label="symmetric")
    plt.bar(xloc + width / 2.0, vals_def, width=width, label="deformed")
    plt.xticks(xloc, labels)
    plt.ylabel("angle (deg)")
    plt.title("Representative tetrahedral PMNS-like angles")
    plt.grid(axis="y", ls=":", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    fig3 = out_figs / "tetrahedral_angles_benchmarks.png"
    plt.savefig(fig3, dpi=180, bbox_inches="tight")
    plt.close()
    print("[INFO] Wrote:", fig3)

    # Figure 4: optional comparison to previous toy (if file exists).
    old_json = out_data / "ccphi2_flavour_toy.json"
    if old_json.exists():
        old = json.loads(old_json.read_text(encoding="utf-8"))
        old_angles = old.get("pmns_angles_deg", {})
        old_vals = [
            float(old_angles.get("theta12", np.nan)),
            float(old_angles.get("theta13", np.nan)),
            float(old_angles.get("theta23", np.nan)),
        ]
    else:
        old_vals = [np.nan, np.nan, np.nan]

    plt.figure(figsize=(6.4, 4.0))
    plt.plot(xloc, old_vals, "o--", label="previous toy reference")
    plt.plot(xloc, vals_def, "s-", label="tetrahedral deformed")
    plt.xticks(xloc, labels)
    plt.ylabel("angle (deg)")
    plt.title("Previous toy vs tetrahedral deformed benchmark")
    plt.grid(axis="y", ls=":", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    fig4 = out_figs / "tetrahedral_vs_previous_toy_angles.png"
    plt.savefig(fig4, dpi=180, bbox_inches="tight")
    plt.close()
    print("[INFO] Wrote:", fig4)


if __name__ == "__main__":
    main()
