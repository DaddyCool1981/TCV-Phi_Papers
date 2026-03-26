#!/usr/bin/env python3
"""Run Phi1->Phi2 emergence scan pipeline (private prep for paper-08)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
mpl_dir = REPO_ROOT / ".tmp_mpl"
mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))

CORE_LIB = REPO_ROOT / "core" / "lib"
if str(CORE_LIB) not in sys.path:
    sys.path.insert(0, str(CORE_LIB))

from tcvphi.phi1_emergence import NetworkConfig  # noqa: E402
from tcvphi.phi1_emergence_diagnostics import (  # noqa: E402
    calibrate_m2_to_uldm,
    classify_hypothesis,
    run_reduced_homogeneous_uldm_test,
    run_emergence_scan,
)


def main() -> None:
    out_data = REPO_ROOT / "papers" / "paper-08" / "data"
    out_data.mkdir(parents=True, exist_ok=True)

    # Priority order requested by user: twisted_ring, ring, scale.
    families = [
        NetworkConfig(family="twisted_ring", n=128),
        NetworkConfig(family="ring", n=128),
        NetworkConfig(family="ladder", n=128),
        # modular kept as secondary comparator.
        NetworkConfig(family="modular", n=128, modular_n1=64, modular_n2=64),
    ]

    scan = run_emergence_scan(
        families=families,
        n_samples_per_family=320,
        seed=20260312,
        disorder_levels=(0.05, 0.10, 0.20),
        soft_thr=0.03,
        gap_thr=0.15,
        support_thr=0.25,
    )
    out_scan = out_data / "phi1_phi2_emergence_scan_summary.json"
    out_scan.write_text(json.dumps(scan, indent=2), encoding="utf-8")
    print("[INFO] Wrote:", out_scan)

    cal = calibrate_m2_to_uldm(
        scan=scan,
        anchor_family="twisted_ring",
        m2_ref_eV=1.0e-22,
        norm_uncert_factor=3.0,
    )
    out_cal = out_data / "phi1_phi2_emergence_calibration_summary.json"
    out_cal.write_text(json.dumps(cal, indent=2), encoding="utf-8")
    print("[INFO] Wrote:", out_cal)

    cls = classify_hypothesis(scan=scan, calibration=cal)
    cls["reduced_homogeneous_test"] = run_reduced_homogeneous_uldm_test()
    out_cls = out_data / "phi1_phi2_emergence_classification.json"
    out_cls.write_text(json.dumps(cls, indent=2), encoding="utf-8")
    print("[INFO] Wrote:", out_cls)


if __name__ == "__main__":
    main()
