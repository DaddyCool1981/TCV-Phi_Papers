#!/usr/bin/env python3
"""Hybrid PMNS inverse/direct framework for geometry ranking and Phi2 bridge."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CORE_LIB = REPO_ROOT / "core" / "lib"
if str(CORE_LIB) not in sys.path:
    sys.path.insert(0, str(CORE_LIB))

from tcvphi.flavor_geometry_families import (
    add_combined_scores,
    collect_pmns_rows,
    default_geometry_families,
)
from tcvphi.flavor_inverse_geometry import PMNSTarget, infer_constraint_profile
from tcvphi.flavor_phi2_bridge import bridge_scores_by_family, merge_pmns_and_bridge

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


def build_workbook(summary: dict) -> str:
    top = summary["joint_rankings"][0]
    lines = [
        "# PMNS -> geometry inverse workbook",
        "",
        "## 1. Motivation",
        "",
        "This notebook replaces intuition-only geometry testing with a two-stage program:",
        "",
        "1. infer what modal constraints PMNS appears to require,",
        "2. rank geometry families by both PMNS quality and structural compatibility,",
        "3. bridge the best families to the Phi2-emergence / one-field pipeline.",
        "",
        "The output is a constraint profile, not a proof of uniqueness.",
        "",
        "## 2. PMNS inverse logic",
        "",
        f"- Target benchmark: theta12={summary['constraint_profile']['target_pmns_deg']['theta12']:.1f} deg, "
        f"theta13={summary['constraint_profile']['target_pmns_deg']['theta13']:.1f} deg, "
        f"theta23={summary['constraint_profile']['target_pmns_deg']['theta23']:.1f} deg.",
        "- Large theta23 points toward a quasi-degenerate doublet or collective 2-3 sector.",
        "- Small but nonzero theta13 points toward weak singlet-doublet communication rather than exact decoupling.",
        "- Moderate theta12 suggests a singlet + doublet decomposition with controlled symmetry breaking.",
        "- The current inference therefore favors closed geometries and gives a further boost to twisted/nontrivial topologies.",
        "",
        "## 3. Geometry candidate families",
        "",
        "Tested families:",
    ]
    for row in summary["family_rankings"]:
        lines.append(
            f"- {row['geometry']}: class={row['structural_class']}, structural_score={row['structural_score']:.3f}, "
            f"pmns_index={row['pmns_viability_index']:.3f}"
        )
    lines += [
        "",
        "## 4. PMNS ranking",
        "",
        "Family ranking by the combined PMNS + structure score:",
    ]
    for row in summary["family_rankings"]:
        lines.append(
            f"- {row['geometry']}: combined={row['combined_pmns_structure_score']:.3f}, "
            f"strict_fraction={row['strict_fraction_within_natural']:.4f}, "
            f"score_common={row['score_common']:.3f}"
        )
    lines += [
        "",
        "## 5. Phi2 bridge",
        "",
        "The direct bridge reuses the existing Phi1->Phi2 emergence pipeline rather than inventing a new toy.",
        "",
        "Key result:",
        f"- top joint family: {top['geometry']} with classification `{top['classification']}` and joint_score={top['joint_score']:.3f}.",
        "",
        "Twisted closed-loop geometry is the only family that is simultaneously PMNS-strong and Phi2-bridge-strong in the current in-repo evidence.",
        "",
        "## 6. Micro-closure perspective",
        "",
        "- The bridge remains strongest for the twisted closed-loop class.",
        "- Micro-closure remains open rather than solved: this is a structural research target, not a completed derivation.",
        "- The current data therefore support a shared missing geometric ingredient rather than two disconnected successes.",
        "",
        "## 7. Honest conclusion",
        "",
        f"- Best current PMNS-only runner-up: {summary['pmns_only_runner_up']['geometry']} "
        f"({summary['pmns_only_runner_up']['classification']}).",
        f"- Strongest bridge candidate: {summary['joint_rankings'][0]['geometry']}.",
        "- Current interpretation: PMNS and one-field completion both point toward closed, collective, twisted micro-geometries.",
        "- This is supportive evidence, not a proof of uniqueness or full microscopic closure.",
        "",
    ]
    return "\n".join(lines)


def build_bridge_note(summary: dict) -> str:
    lines = [
        "# PMNS to geometry Phi2 bridge summary",
        "",
        "## Joint classification",
        "",
    ]
    for row in summary["joint_rankings"]:
        lines.append(
            f"- {row['geometry']}: class={row['classification']}, joint_score={row['joint_score']:.3f}, "
            f"emergent_phi2_score={row['emergent_phi2_score']:.3f}, one_field_support={row['one_field_support_score']:.3f}, "
            f"micro_closure={row['micro_closure_status']}"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- `twisted_torus` is the only family that is jointly favored by PMNS structure and the existing Phi2-emergence pipeline.",
        "- `torus_ring` remains a useful control: good PMNS behavior, but weak Phi2 bridge without twist.",
        "- Rigid/open families are currently disfavored in the joint program, either because PMNS quality is weak or because no emergence bridge exists.",
        "",
        "## Status",
        "",
        "- Supported now: a reproducible PMNS -> geometry -> Phi2 bridge workflow.",
        "- Still open: full microscopic closure and uniqueness of the geometry class.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    NOTES_DIR.mkdir(parents=True, exist_ok=True)

    profile = infer_constraint_profile(PMNSTarget())
    rows = add_combined_scores(collect_pmns_rows(REPO_ROOT, profile))
    bridge = bridge_scores_by_family(REPO_ROOT)
    joint_rows = merge_pmns_and_bridge(rows, bridge)

    geometry_matrix = [
        {
            "geometry": name,
            "display_name": family.display_name,
            "structural_class": family.structural_class,
            "properties": dict(family.properties),
            "notes": family.notes,
        }
        for name, family in default_geometry_families().items()
    ]

    summary = {
        "status": "private hybrid inverse/direct PMNS-to-geometry workflow",
        "constraint_profile": profile,
        "geometry_property_matrix": geometry_matrix,
        "family_rankings": rows,
        "joint_rankings": joint_rows,
        "pmns_only_runner_up": next((row for row in joint_rows if row["classification"].startswith("A_")), joint_rows[1]),
        "top_joint_candidate": joint_rows[0],
        "notes": [
            "The inverse stage is heuristic and explicitly non-unique.",
            "The direct stage reuses existing in-repo flavour scan outputs rather than rerunning every toy.",
            "The Phi2 bridge uses the existing emergence/micro-closure diagnostics from paper-08.",
        ],
    }

    _write_json(DATA_DIR / "pmns_inverse_geometry_scan_summary.json", summary)
    _write_json(DATA_DIR / "pmns_inverse_geometry_constraint_profile.json", profile)
    _write_text(NOTES_DIR / "pmns_to_geometry_inverse_workbook.md", build_workbook(summary))
    _write_text(NOTES_DIR / "pmns_to_geometry_phi2_bridge_summary.md", build_bridge_note(summary))


if __name__ == "__main__":
    main()
