#!/usr/bin/env python3
"""Extend the inverse/direct workflow with richer loop-based topology families."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CORE_LIB = REPO_ROOT / "core" / "lib"
if str(CORE_LIB) not in sys.path:
    sys.path.insert(0, str(CORE_LIB))

from tcvphi.topological_geometry_families import topological_extension_specs
from tcvphi.topological_geometry_scan import scan_new_topological_families


DATA_DIR = REPO_ROOT / "papers" / "paper-08" / "data"
NOTES_DIR = REPO_ROOT / "notes"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("[INFO] Wrote:", path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print("[INFO] Wrote:", path)


def build_note(summary: dict) -> str:
    lines = [
        "# Topological geometry extension workbook",
        "",
        "## Why test richer loop-based topologies?",
        "",
        "The current inverse/direct workflow already points toward closed twisted geometries.",
        "The remaining question is whether the present twisted torus is the right effective geometry,",
        "or only the best low-complexity approximation to a richer topological microstructure.",
        "",
        "## Added families",
        "",
    ]
    for row in summary["new_family_rankings"]:
        lines.append(
            f"- {row['geometry']}: joint_score={row['joint_score']:.3f}, class={row['classification']}, "
            f"angles={row['representative_angles_deg']}"
        )
    lines += [
        "",
        "## Structural ingredient tested by each family",
        "",
        "- coupled_loop_network: multiple collective loop channels",
        "- loop_lace: frustration and woven collective paths",
        "- twisted_multi_ring: several twisted channels rather than one",
        "- phase_loop: minimal loop-phase / holonomy-like structure",
        "",
        "## Honest conclusion",
        "",
        f"- Best verified baseline remains: {summary['best_verified_baseline']['geometry']} ({summary['best_verified_baseline']['classification']}).",
        f"- Best new proxy family: {summary['best_new_family']['geometry']} ({summary['best_new_family']['classification']}).",
        f"- Best raw proxy score overall: {summary['best_proxy_overall']['geometry']} ({summary['best_proxy_overall']['classification']}).",
        "- The new families are still extension-proxy results, not dedicated flavour + micro-closure solves.",
        "- Therefore they are informative hints, not sufficient grounds to displace the current twisted_torus baseline.",
        "- The current reading is that richer topologies deserve dedicated follow-up because they may encode the missing ingredient, but the best verified candidate remains twisted_torus.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    base_summary = json.loads((DATA_DIR / "pmns_inverse_geometry_scan_summary.json").read_text(encoding="utf-8"))
    profile = base_summary["constraint_profile"]
    old_rows = base_summary["joint_rankings"]

    new_scans = scan_new_topological_families(profile)
    new_best_rows = [bundle["best_record"] for bundle in new_scans.values()]
    new_best_rows.sort(key=lambda row: row["joint_score"], reverse=True)

    all_rows = list(old_rows) + list(new_best_rows)
    all_rows.sort(key=lambda row: row["joint_score"], reverse=True)

    matrix = list(base_summary["geometry_property_matrix"])
    for name, spec in topological_extension_specs().items():
        matrix.append(
            {
                "geometry": name,
                "display_name": spec.display_name,
                "structural_class": "extended_topological_family",
                "properties": dict(spec.properties),
                "notes": list(spec.notes),
            }
        )

    summary = {
        "status": "additive topological extension scan over the existing inverse/direct workflow",
        "baseline_reference": {
            "current_best_geometry": base_summary["top_joint_candidate"]["geometry"],
            "current_best_joint_score": base_summary["top_joint_candidate"]["joint_score"],
        },
        "best_verified_baseline": base_summary["top_joint_candidate"],
        "new_family_rankings": new_best_rows,
        "all_geometry_rankings": all_rows,
        "best_proxy_overall": all_rows[0],
        "geometry_property_matrix_extended": matrix,
        "new_family_scan_details": {
            name: {
                "display_name": bundle["spec"].display_name,
                "description": bundle["spec"].description,
                "best_record": bundle["best_record"],
                "n_records": len(bundle["records"]),
            }
            for name, bundle in new_scans.items()
        },
        "best_new_family": new_best_rows[0],
        "best_overall": base_summary["top_joint_candidate"],
        "conservative_recommendation": {
            "geometry": base_summary["top_joint_candidate"]["geometry"],
            "reason": "extension families are proxy-ranked only and do not yet supersede the verified twisted_torus baseline",
        },
        "notes": [
            "PMNS for the new families is evaluated with an explicit proxy modal map, not a full flavour Hamiltonian re-derivation.",
            "Phi2 bridge for the new families uses a light network/Hessian diagnostic and a proxy micro-closure score only.",
            "If a new family looks promising here, it still deserves its own dedicated flavour and micro-closure modules later.",
        ],
    }

    _write_json(DATA_DIR / "topological_geometry_extension_summary.json", summary)
    _write_text(NOTES_DIR / "topological_geometry_extension_workbook.md", build_note(summary))


if __name__ == "__main__":
    main()
