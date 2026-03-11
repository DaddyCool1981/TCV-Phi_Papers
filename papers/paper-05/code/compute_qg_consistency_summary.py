#!/usr/bin/env python3
"""Aggregate Paper-05 diagnostics into a single consistency summary."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {"missing": True, "path": str(path)}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    out_data = REPO_ROOT / "papers" / "paper-05" / "data"
    out_data.mkdir(parents=True, exist_ok=True)

    ccphi1_spec = _load_json(out_data / "ccphi1_spectrum.json")
    ccphi1_curve = _load_json(out_data / "ccphi1_curvature_summary.json")
    ccphi1_cg = _load_json(out_data / "ccphi1_coarse_grain.json")
    ccphi2_toy = _load_json(out_data / "ccphi2_flavour_toy.json")

    summary = {
        "paper": "paper-05",
        "scope": "QG microstructure CC-Phi1 + exploratory CC-Phi2 flavour bridge",
        "status_labels": {
            "ccphi1_spectrum": "phenomenological cluster quantization toy",
            "ccphi1_curvature_response": "phenomenological scan",
            "ccphi1_coarse_grain": "toy coarse-graining proxy",
            "ccphi2_flavour_toy": "exploratory flavour toy",
        },
        "artifacts": {
            "ccphi1_spectrum": ccphi1_spec,
            "ccphi1_curvature_response": ccphi1_curve,
            "ccphi1_coarse_grain": ccphi1_cg,
            "ccphi2_flavour_toy": ccphi2_toy,
        },
        "class_camb_note": (
            "No direct CLASS/CAMB call is required by current microstructure "
            "and flavour-toy scripts. Bridge to CLASS/CAMB remains through "
            "primordial/late-time cosmology pipelines from Papers III-IV."
        ),
    }

    out_json = out_data / "paper05_consistency_summary.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("[INFO] Wrote:", out_json)


if __name__ == "__main__":
    main()
