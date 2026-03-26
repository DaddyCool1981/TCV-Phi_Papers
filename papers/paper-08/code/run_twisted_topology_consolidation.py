#!/usr/bin/env python3
"""Consolidate the current twisted-topology baseline and main candidate results."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "papers" / "paper-08" / "data"
OUT_PATH = DATA_DIR / "twisted_topology_consolidated_summary.json"


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    headtohead = _load_json(DATA_DIR / "twisted_topology_headtohead_summary.json")
    multi_bridge = _load_json(DATA_DIR / "twisted_multi_ring_full_bridge_summary.json")
    multi_micro = _load_json(DATA_DIR / "twisted_multi_ring_micro_closure_full_summary.json")
    multi_sector = _load_json(
        ROOT / "papers" / "paper-05" / "data" / "twisted_multi_ring_sector_compatibility_summary.json"
    )

    torus = headtohead["twisted_torus"]
    multi = headtohead["twisted_multi_ring"]

    consolidated = {
        "status": "consolidated twisted topology comparison",
        "scope": {
            "historical_baseline": "twisted_torus",
            "current_main_candidate": "twisted_multi_ring",
            "comparison_rule": (
                "Twisted torus is kept as the verified historical baseline. "
                "Twisted multi ring uses the latest dedicated PMNS, full bridge, "
                "sector robustness, and full micro-closure outputs."
            ),
        },
        "twisted_torus": {
            "provenance": {
                "pmns_bridge": str(DATA_DIR / "twisted_topology_headtohead_summary.json"),
                "note": "Best verified historical baseline available in the current consolidated stack.",
            },
            "pmns": torus["pmns"],
            "bridge": torus["bridge"],
            "indices": {
                "flavour_index": torus["flavour_index"],
                "bridge_index": torus["bridge_index"],
                "overall_index": torus["overall_index"],
            },
            "position": "historical_baseline",
        },
        "twisted_multi_ring": {
            "provenance": {
                "pmns_bridge_headtohead": str(DATA_DIR / "twisted_topology_headtohead_summary.json"),
                "full_bridge": str(DATA_DIR / "twisted_multi_ring_full_bridge_summary.json"),
                "sector_compatibility": str(
                    ROOT / "papers" / "paper-05" / "data" / "twisted_multi_ring_sector_compatibility_summary.json"
                ),
                "full_micro_closure": str(DATA_DIR / "twisted_multi_ring_micro_closure_full_summary.json"),
            },
            "pmns": multi["pmns"],
            "bridge_headtohead": multi["bridge"],
            "full_bridge": {
                "joint_fraction": multi_bridge["emergence"]["joint_fraction"],
                "robustness_pass_fraction": multi_bridge["emergence"]["robustness_pass_fraction"],
                "calibrated_m2_eV": multi_bridge["calibrated_m2_eV"],
                "m2_log10_dev_from_1e-22": multi_bridge["m2_log10_dev_from_1e-22"],
                "first_mode_weight_mean": multi_bridge["canonical"]["metrics"]["first_mode_weight_mean"],
                "mode13_weight_mean": multi_bridge["canonical"]["metrics"]["mode13_weight_mean"],
                "homogeneous_slope": multi_bridge["homogeneous"]["diagnostics"]["measured_log_slope_rho_vs_a"],
                "size_scaling_exponent": multi_bridge["size_scaling"]["fit"]["exponent_positive"],
                "checks": multi_bridge["checks"],
                "overall_support": multi_bridge["overall_support"],
            },
            "sector_compatibility": {
                "lepton_strict_fraction": multi_sector["lepton_robustness"]["strict_fraction"],
                "theta23_std_deg": multi_sector["lepton_robustness"]["theta23_std"],
                "quark_theta12_mean_deg": multi_sector["quark_like_compatibility"]["theta12_mean"],
                "quark_theta13_mean_deg": multi_sector["quark_like_compatibility"]["theta13_mean"],
                "quark_theta23_mean_deg": multi_sector["quark_like_compatibility"]["theta23_mean"],
            },
            "micro_closure": {
                "verdict": multi_micro["verdict"],
                "n_all_pass": multi_micro["n_all_pass"],
                "best_record": multi_micro["best_record"],
            },
            "indices": {
                "flavour_index": multi["flavour_index"],
                "bridge_index": multi["bridge_index"],
                "overall_index": multi["overall_index"],
            },
            "position": "current_main_candidate",
        },
        "decision": {
            "winner": "twisted_multi_ring",
            "reason": [
                "better dedicated PMNS benchmark and strict-yield fraction",
                "better charged-lepton robustness",
                "full bridge support now passes all gates",
                "dedicated micro-closure reaches partial_supported with multiple passing records",
            ],
            "recommendation": (
                "Keep twisted_torus as the historical geometry that revealed the topological direction, "
                "but use twisted_multi_ring as the main working candidate for future PMNS and Phi2-emergence work."
            ),
            "paper_positioning": {
                "paper_05": (
                    "Present twisted_torus as the historical trigger, then show that systematic geometry "
                    "testing identifies twisted_multi_ring as the strongest current PMNS candidate."
                ),
                "paper_08_future": (
                    "State that multiple geometries were tested for Phi2 emergence and one-field support, "
                    "with twisted_multi_ring giving the strongest current combined signal."
                ),
            },
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(consolidated, handle, indent=2)

    print(f"[INFO] Wrote: {OUT_PATH}")


if __name__ == "__main__":
    main()
