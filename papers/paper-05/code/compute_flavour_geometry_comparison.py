#!/usr/bin/env python3
"""Compare flavour toy geometries with a common score/criteria."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
mpl_dir = REPO_ROOT / ".tmp_mpl"
mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def pmns_score(angles: Dict[str, float]) -> float:
    """Common PMNS proximity score used across all geometries."""
    d12 = (float(angles["theta12"]) - 33.0) / 10.0
    d13 = (float(angles["theta13"]) - 8.6) / 4.0
    d23 = (float(angles["theta23"]) - 45.0) / 7.0
    return float(np.sqrt(d12 * d12 + d13 * d13 + d23 * d23))


def strict_ok(angles: Dict[str, float]) -> bool:
    return (
        40.0 <= float(angles["theta23"]) <= 50.0
        and 2.0 <= float(angles["theta13"]) <= 12.0
        and 10.0 <= float(angles["theta12"]) <= 40.0
    )


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def get_best_angles_from_list(candidates: list) -> Optional[Dict[str, float]]:
    if not candidates:
        return None
    return candidates[0]["angles_deg"]


def main() -> None:
    out_data = REPO_ROOT / "papers" / "paper-05" / "data"
    out_figs = REPO_ROOT / "papers" / "paper-05" / "figs"
    out_data.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    d_pyr = load_json(out_data / "ccphi2_flavour_toy.json")
    d_tet = load_json(out_data / "tetrahedral_targeted_scan_summary.json")
    d_star = load_json(out_data / "tetrahedral_star_targeted_scan_summary.json")
    d_torus = load_json(out_data / "torus_flavour_toy_summary.json")
    sph_path = out_data / "spherical_flavour_toy_summary.json"
    d_sph = load_json(sph_path) if sph_path.exists() else None
    tw_path = out_data / "twisted_torus_flavour_toy_summary.json"
    d_tw = load_json(tw_path) if tw_path.exists() else None
    tw_sec_path = out_data / "twisted_torus_sector_compatibility_summary.json"
    d_tw_sec = load_json(tw_sec_path) if tw_sec_path.exists() else None

    # Representative angles from each geometry.
    rep = {
        "pyramidal": d_pyr["pmns_angles_deg"],
        "tetrahedral_apex_base": get_best_angles_from_list(d_tet.get("best_candidates", [])),
        "tetrahedral_star": get_best_angles_from_list(d_star.get("best_candidates", [])),
        "torus_ring": get_best_angles_from_list(d_torus.get("best_candidates", [])),
    }
    if d_sph is not None:
        rep["spherical_shell"] = get_best_angles_from_list(d_sph.get("best_candidates", []))
    if d_tw is not None:
        rep["twisted_torus"] = get_best_angles_from_list(d_tw.get("best_candidates", []))

    rows = []
    for name, angles in rep.items():
        if angles is None:
            continue
        row = {
            "geometry": name,
            "angles_deg": {k: float(angles[k]) for k in ("theta12", "theta13", "theta23")},
            "score_common": pmns_score(angles),
            "strict_ok_representative": bool(strict_ok(angles)),
        }
        rows.append(row)

    # Strict counts/fractions from available scans.
    scan_stats = {
        "tetrahedral_apex_base": {
            "n_samples": int(d_tet["n_samples"]),
            "n_natural": int(d_tet["n_natural"]),
            "n_target_like_strict": int(d_tet.get("n_target_like", 0)),
            "n_target_like_relaxed": int(d_tet.get("n_target_like_relaxed", 0)),
        },
        "tetrahedral_star": {
            "n_samples": int(d_star["n_samples"]),
            "n_natural": int(d_star["n_natural"]),
            "n_target_like_strict": int(d_star.get("n_target_like_strict", 0)),
            "n_target_like_relaxed": int(d_star.get("n_target_like_relaxed", 0)),
        },
        "torus_ring": {
            "n_samples": int(d_torus["scan_summary"]["n_samples"]),
            "n_natural": int(d_torus["scan_summary"]["n_natural"]),
            "n_target_like_strict": int(d_torus["scan_summary"]["n_target_like_strict"]),
            "n_target_like_relaxed": int(d_torus["scan_summary"]["n_target_like_relaxed"]),
        },
    }
    if d_sph is not None:
        scan_stats["spherical_shell"] = {
            "n_samples": int(d_sph["scan_summary"]["n_samples"]),
            "n_natural": int(d_sph["scan_summary"]["n_natural"]),
            "n_target_like_strict": int(d_sph["scan_summary"]["n_target_like_strict"]),
            "n_target_like_relaxed": int(d_sph["scan_summary"]["n_target_like_relaxed"]),
        }
    if d_tw is not None:
        scan_stats["twisted_torus"] = {
            "n_samples": int(d_tw["scan_summary"]["n_samples"]),
            "n_natural": int(d_tw["scan_summary"]["n_natural"]),
            "n_target_like_strict": int(d_tw["scan_summary"]["n_target_like_strict"]),
            "n_target_like_relaxed": int(d_tw["scan_summary"]["n_target_like_relaxed"]),
        }
    for _, st in scan_stats.items():
        n_nat = max(st["n_natural"], 1)
        st["strict_fraction_within_natural"] = float(st["n_target_like_strict"] / n_nat)
        st["relaxed_fraction_within_natural"] = float(st["n_target_like_relaxed"] / n_nat)

    rep_by_name = {r["geometry"]: r for r in rows}

    # Global indices:
    # - pmns_viability_index: combines representative score and strict scan yield.
    # - sector_extension_index: adds leptonic robustness + quark-like compatibility
    #   when available (currently twisted-torus only).
    score_max = max(float(r["score_common"]) for r in rows) if rows else 1.0
    score_min = min(float(r["score_common"]) for r in rows) if rows else 0.0
    score_span = max(score_max - score_min, 1.0e-14)

    global_table = []
    for g, rep_row in rep_by_name.items():
        score = float(rep_row["score_common"])
        score_quality = float((score_max - score) / score_span)  # higher is better
        strict_fraction = float(scan_stats.get(g, {}).get("strict_fraction_within_natural", 0.0))
        pmns_index = float(0.5 * score_quality + 0.5 * strict_fraction)

        entry = {
            "geometry": g,
            "score_common": score,
            "score_quality_0to1": score_quality,
            "strict_fraction_within_natural": strict_fraction,
            "pmns_viability_index": pmns_index,
        }

        if g == "twisted_torus" and d_tw_sec is not None:
            lep_frac = float(d_tw_sec["lepton_robustness"]["strict_fraction"])
            q12p95 = float(d_tw_sec["quark_like_compatibility"]["theta12_p95"])
            q13p95 = float(d_tw_sec["quark_like_compatibility"]["theta13_p95"])
            q23p95 = float(d_tw_sec["quark_like_compatibility"]["theta23_p95"])
            q_mix_sum = q12p95 + q13p95 + q23p95
            # Small quark-like mixing quality: near 1 for very small angles.
            q_quality = float(1.0 / (1.0 + q_mix_sum))
            sector_index = float(0.4 * pmns_index + 0.4 * lep_frac + 0.2 * q_quality)
            entry["lepton_strict_fraction"] = lep_frac
            entry["quark_small_mixing_quality"] = q_quality
            entry["sector_extension_index"] = sector_index
        global_table.append(entry)

    global_table.sort(key=lambda x: x["pmns_viability_index"], reverse=True)

    summary = {
        "status": "additive cross-geometry comparison with common PMNS score and strict criteria",
        "criteria": {
            "strict_window_deg": {"theta12": [10.0, 40.0], "theta13": [2.0, 12.0], "theta23": [40.0, 50.0]},
            "score_targets_deg": {"theta12": 33.0, "theta13": 8.6, "theta23": 45.0},
            "score_scales_deg": {"theta12": 10.0, "theta13": 4.0, "theta23": 7.0},
        },
        "representative_points": rows,
        "scan_statistics": scan_stats,
        "global_viability_table": global_table,
        "notes": [
            "Representative points use best-candidate entries when scans are available.",
            "Pyramidal toy has only a benchmark point, not a matched scan in this file.",
            "Strict fractions compare only geometries with scan outputs in-repo.",
            "Sector extension index is available only when dedicated compatibility diagnostics exist.",
        ],
    }

    out_json = out_data / "flavour_geometry_comparison_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[INFO] Wrote:", out_json)

    # Figure with two panels:
    # left: representative score by geometry
    # right: strict fraction within natural for scanned geometries
    geom_names = [r["geometry"] for r in rows]
    scores = [float(r["score_common"]) for r in rows]

    strict_names = list(scan_stats.keys())
    strict_fracs = [float(scan_stats[g]["strict_fraction_within_natural"]) for g in strict_names]

    fig, ax = plt.subplots(1, 3, figsize=(14.8, 4.2))
    ax[0].bar(np.arange(len(geom_names)), scores, color="#4C72B0")
    ax[0].set_xticks(np.arange(len(geom_names)))
    ax[0].set_xticklabels(geom_names, rotation=20, ha="right")
    ax[0].set_ylabel("common PMNS score (lower is better)")
    ax[0].set_title("Representative benchmark score")
    ax[0].grid(axis="y", ls=":", alpha=0.4)

    ax[1].bar(np.arange(len(strict_names)), strict_fracs, color="#55A868")
    ax[1].set_xticks(np.arange(len(strict_names)))
    ax[1].set_xticklabels(strict_names, rotation=20, ha="right")
    ax[1].set_ylabel("strict fraction within natural scan")
    ax[1].set_title("Strict-window yield")
    ax[1].grid(axis="y", ls=":", alpha=0.4)

    v_names = [r["geometry"] for r in global_table]
    v_vals = [float(r["pmns_viability_index"]) for r in global_table]
    ax[2].bar(np.arange(len(v_names)), v_vals, color="#C44E52")
    ax[2].set_xticks(np.arange(len(v_names)))
    ax[2].set_xticklabels(v_names, rotation=20, ha="right")
    ax[2].set_ylabel("PMNS viability index (0-1)")
    ax[2].set_title("Global ranking")
    ax[2].grid(axis="y", ls=":", alpha=0.4)

    fig.tight_layout()
    out_fig = out_figs / "flavour_geometry_comparison_overview.png"
    fig.savefig(out_fig, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("[INFO] Wrote:", out_fig)

    if d_tw_sec is not None:
        tw = next((r for r in global_table if r["geometry"] == "twisted_torus"), None)
        if tw is not None and "sector_extension_index" in tw:
            fig2, ax2 = plt.subplots(1, 1, figsize=(6.0, 4.2))
            labels = ["PMNS index", "Lepton robust.", "Quark quality", "Sector index"]
            vals = [
                float(tw["pmns_viability_index"]),
                float(tw["lepton_strict_fraction"]),
                float(tw["quark_small_mixing_quality"]),
                float(tw["sector_extension_index"]),
            ]
            ax2.bar(np.arange(len(labels)), vals, color=["#4C72B0", "#55A868", "#8172B2", "#C44E52"])
            ax2.set_xticks(np.arange(len(labels)))
            ax2.set_xticklabels(labels, rotation=20, ha="right")
            ax2.set_ylim(0.0, 1.0)
            ax2.set_ylabel("normalized metric (0-1)")
            ax2.set_title("Twisted-torus sector dashboard")
            ax2.grid(axis="y", ls=":", alpha=0.4)
            fig2.tight_layout()
            out_fig2 = out_figs / "twisted_torus_sector_dashboard.png"
            fig2.savefig(out_fig2, dpi=180, bbox_inches="tight")
            plt.close(fig2)
            print("[INFO] Wrote:", out_fig2)


if __name__ == "__main__":
    main()
