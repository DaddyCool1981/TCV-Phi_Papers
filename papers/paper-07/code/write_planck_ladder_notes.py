#!/usr/bin/env python3
"""Write a machine-readable Planck-fit execution ladder for Paper 07."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    out_data = REPO_ROOT / "papers" / "paper-07" / "data"
    out_data.mkdir(parents=True, exist_ok=True)

    checklist = {
        "paper": "paper-07",
        "title": "Planck confrontation ladder",
        "stages": [
            {
                "id": "L1",
                "name": "Baseline technical validation",
                "objective": "Reproduce stable CLASS/CAMB baseline outputs before likelihood runs.",
                "required_artifacts": [
                    "baseline_validation.json",
                    "baseline_pk_comparison.png",
                ],
            },
            {
                "id": "L2",
                "name": "LCDM likelihood run",
                "objective": "Run Planck likelihood in standard LCDM mode to validate inference stack.",
                "required_artifacts": [
                    "lcdm_chain/",
                    "lcdm_posteriors.json",
                    "lcdm_bestfit.json",
                ],
            },
            {
                "id": "L3",
                "name": "Minimal TCV extension",
                "objective": "Activate minimal TCV parameters and compare to LCDM.",
                "required_artifacts": [
                    "tcv_min_chain/",
                    "tcv_min_posteriors.json",
                    "model_comparison.json",
                ],
            },
            {
                "id": "L4",
                "name": "Robustness and publication tables",
                "objective": "Stress priors, solver precision, and convergence diagnostics before PRD submission.",
                "required_artifacts": [
                    "robustness_checks.json",
                    "table_constraints.tex",
                    "table_model_comparison.tex",
                ],
            },
        ],
        "success_criteria": {
            "minimum": "TCV-minimal is not statistically disfavored relative to LCDM.",
            "strong": "TCV-minimal improves fit quality without unacceptable complexity penalty.",
        },
    }

    out_json = out_data / "planck_ladder_checklist.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(checklist, f, indent=2)
    print("[INFO] Wrote:", out_json)


if __name__ == "__main__":
    main()
