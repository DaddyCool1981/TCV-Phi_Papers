#!/usr/bin/env python3
"""Build conservative strong-field template diagnostics for Paper-04.

This script intentionally produces qualitative templates only.
It does not perform waveform-level inference.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    out_data = REPO_ROOT / "papers" / "paper-04" / "data"
    in_npz = out_data / "bh_profiles.npz"
    if not in_npz.exists():
        raise FileNotFoundError(f"Missing {in_npz}; run compute_bh_profiles.py first.")

    data = np.load(in_npz, allow_pickle=False)
    family_names = data["family_names"]
    m = data["m"]
    r_h = data["horizon_radius"]

    templates: dict[str, dict[str, float | str]] = {}
    for i, family in enumerate(family_names):
        name = str(family)
        m_inf = float(m[i, -1])
        rh = float(r_h[i, 0])
        if np.isfinite(rh) and rh > 0.0:
            compactness = 2.0 * m_inf / rh
        else:
            compactness = None
            rh = None
        templates[name] = {
            "adm_mass": m_inf,
            "horizon_radius": rh,
            "compactness_at_horizon": compactness,
            "note": "Template-level static proxy only; no waveform fit.",
        }

    out_json = out_data / "bh_templates.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(templates, f, indent=2)
    print("[INFO] Wrote:", out_json)


if __name__ == "__main__":
    main()
