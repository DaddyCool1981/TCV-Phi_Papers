#!/usr/bin/env python3
"""Focused v2.1 scan near the best v2 twisted_ring region."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
mpl_dir = REPO_ROOT / ".tmp_mpl"
mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))

CORE_LIB = REPO_ROOT / "core" / "lib"
if str(CORE_LIB) not in sys.path:
    sys.path.insert(0, str(CORE_LIB))

from tcvphi.phi1_emergence import NetworkConfig  # noqa: E402
from tcvphi.phi1_emergence_diagnostics import run_emergence_scan  # noqa: E402


def run_two_family_scan(tw_cfg: NetworkConfig, ring_cfg: NetworkConfig, n_samples: int, seed: int) -> dict:
    scan = run_emergence_scan(
        families=[tw_cfg, ring_cfg],
        n_samples_per_family=n_samples,
        seed=seed,
        disorder_levels=(0.05, 0.10, 0.20),
        soft_thr=0.03,
        gap_thr=0.15,
        support_thr=0.25,
    )
    fam = {r["family"]: r for r in scan["family_results"]}
    return {"twisted_ring": fam["twisted_ring"], "ring": fam["ring"]}


def main() -> None:
    out_data = REPO_ROOT / "papers" / "paper-08" / "data"
    out_data.mkdir(parents=True, exist_ok=True)

    # v2 baseline reference for mass calibration.
    base_tw = NetworkConfig(family="twisted_ring", n=128)
    base_ring = NetworkConfig(family="ring", n=128)
    base_run = run_two_family_scan(base_tw, base_ring, n_samples=260, seed=20260312)
    alpha_ref = 1.0e-22 / max(float(base_run["twisted_ring"]["m2_proxy_mean"]), 1.0e-30)

    edge_vals = [0.06, 0.08, 0.10]
    drop_vals = [0.005, 0.015, 0.025]
    noise_vals = [0.01, 0.03, 0.05]
    seeds = [20260350 + i for i in range(6)]

    records = []
    for edge in edge_vals:
        for drop in drop_vals:
            for noise in noise_vals:
                tw_cfg = NetworkConfig(
                    family="twisted_ring",
                    n=128,
                    edge_aniso=edge,
                    drop_prob=drop,
                    onsite_noise=noise,
                    k_link=1.0,
                    eps_pin=1.0e-4,
                )
                tw_joint = []
                tw_rob = []
                tw_m2 = []
                ring_joint = []
                ring_rob = []
                for s in seeds:
                    rr = run_two_family_scan(tw_cfg, base_ring, n_samples=120, seed=s)
                    tw = rr["twisted_ring"]
                    rg = rr["ring"]
                    tw_joint.append(float(tw["joint_fraction"]))
                    tw_rob.append(float(tw["robustness_pass_fraction"]))
                    tw_m2.append(float(alpha_ref * tw["m2_proxy_mean"]))
                    ring_joint.append(float(rg["joint_fraction"]))
                    ring_rob.append(float(rg["robustness_pass_fraction"]))

                tw_joint_arr = np.array(tw_joint, dtype=float)
                tw_rob_arr = np.array(tw_rob, dtype=float)
                tw_m2_arr = np.array(tw_m2, dtype=float)
                ring_joint_arr = np.array(ring_joint, dtype=float)
                ring_rob_arr = np.array(ring_rob, dtype=float)

                rec = {
                    "params": {"edge_aniso": edge, "drop_prob": drop, "onsite_noise": noise},
                    "twisted_joint_mean": float(np.mean(tw_joint_arr)),
                    "twisted_joint_p16_p84": [float(np.percentile(tw_joint_arr, 16)), float(np.percentile(tw_joint_arr, 84))],
                    "twisted_robust_mean": float(np.mean(tw_rob_arr)),
                    "twisted_robust_p16_p84": [float(np.percentile(tw_rob_arr, 16)), float(np.percentile(tw_rob_arr, 84))],
                    "twisted_m2_mean_eV": float(np.mean(tw_m2_arr)),
                    "twisted_m2_p16_p84_eV": [float(np.percentile(tw_m2_arr, 16)), float(np.percentile(tw_m2_arr, 84))],
                    "ring_joint_mean": float(np.mean(ring_joint_arr)),
                    "ring_robust_mean": float(np.mean(ring_rob_arr)),
                }
                log_dev = abs(np.log10(max(rec["twisted_m2_mean_eV"], 1.0e-40) / 1.0e-22))
                rec["criteria"] = {
                    "joint_pass": bool(rec["twisted_joint_mean"] >= 0.35),
                    "robust_pass": bool(rec["twisted_robust_mean"] >= 0.20),
                    "m2_pass": bool(log_dev <= 0.5),
                    "ring_control_passive": bool(rec["ring_joint_mean"] < 0.05 and rec["ring_robust_mean"] < 0.02),
                    "m2_log10_dev": float(log_dev),
                }
                rec["overall_go"] = bool(
                    rec["criteria"]["joint_pass"]
                    and rec["criteria"]["robust_pass"]
                    and rec["criteria"]["m2_pass"]
                    and rec["criteria"]["ring_control_passive"]
                )
                records.append(rec)

    ranked = sorted(
        records,
        key=lambda r: (
            int(r["overall_go"]),
            r["twisted_robust_mean"],
            r["twisted_joint_mean"],
        ),
        reverse=True,
    )
    best = ranked[0]
    go_count = int(np.sum([int(r["overall_go"]) for r in ranked]))

    summary = {
        "v21_settings": {
            "n_samples_per_seed": 120,
            "seeds": seeds,
            "grid_size": len(records),
            "criteria": {
                "twisted_joint_min": 0.35,
                "twisted_robust_min": 0.20,
                "m2_log10_dev_max": 0.5,
                "ring_joint_max": 0.05,
                "ring_robust_max": 0.02,
            },
        },
        "alpha_ref_eV_per_proxy": float(alpha_ref),
        "best_record": best,
        "go_count": go_count,
        "records": ranked,
    }

    out = out_data / "phi1_phi2_emergence_v21_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[INFO] Wrote:", out)


if __name__ == "__main__":
    main()
