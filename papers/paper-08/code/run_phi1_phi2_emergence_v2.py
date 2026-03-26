#!/usr/bin/env python3
"""Focused v2 emergence pipeline: twisted_ring, ring, and scale tests."""

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


def run_two_family_scan(
    twisted_cfg: NetworkConfig,
    ring_cfg: NetworkConfig,
    n_samples: int,
    seed: int,
) -> dict:
    scan = run_emergence_scan(
        families=[twisted_cfg, ring_cfg],
        n_samples_per_family=n_samples,
        seed=seed,
        disorder_levels=(0.05, 0.10, 0.20),
        soft_thr=0.03,
        gap_thr=0.15,
        support_thr=0.25,
    )
    fam = {r["family"]: r for r in scan["family_results"]}
    return {
        "scan": scan,
        "twisted_ring": fam["twisted_ring"],
        "ring": fam["ring"],
    }


def main() -> None:
    out_data = REPO_ROOT / "papers" / "paper-08" / "data"
    out_data.mkdir(parents=True, exist_ok=True)

    baseline_twisted = NetworkConfig(family="twisted_ring", n=128)
    baseline_ring = NetworkConfig(family="ring", n=128)
    baseline_seed = 20260312
    baseline_samples = 320

    baseline = run_two_family_scan(
        twisted_cfg=baseline_twisted,
        ring_cfg=baseline_ring,
        n_samples=baseline_samples,
        seed=baseline_seed,
    )
    alpha_ref = 1.0e-22 / max(float(baseline["twisted_ring"]["m2_proxy_mean"]), 1.0e-30)

    # 1) Local sensitivity around twisted_ring baseline, one-parameter-at-a-time.
    local_points = []
    grid = {
        "edge_aniso": [0.08, 0.12, 0.16],
        "drop_prob": [0.01, 0.03, 0.05],
        "onsite_noise": [0.02, 0.05, 0.08],
        "k_link": [0.8, 1.0, 1.2],
        "eps_pin": [5.0e-5, 1.0e-4, 2.0e-4],
    }

    # Baseline entry.
    base_tw = baseline["twisted_ring"]
    base_ring = baseline["ring"]
    local_points.append(
        {
            "label": "baseline",
            "vary": "none",
            "value": None,
            "twisted_ring": base_tw,
            "ring": base_ring,
            "twisted_m2_eV_calibrated": alpha_ref * float(base_tw["m2_proxy_mean"]),
        }
    )

    par_offset = {
        "edge_aniso": 101,
        "drop_prob": 211,
        "onsite_noise": 307,
        "k_link": 401,
        "eps_pin": 503,
    }
    for par, vals in grid.items():
        for v in vals:
            if np.isclose(v, getattr(baseline_twisted, par)):
                continue
            tw_cfg = NetworkConfig(
                family="twisted_ring",
                n=128,
                edge_aniso=baseline_twisted.edge_aniso,
                drop_prob=baseline_twisted.drop_prob,
                onsite_noise=baseline_twisted.onsite_noise,
                k_link=baseline_twisted.k_link,
                eps_pin=baseline_twisted.eps_pin,
            )
            tw_cfg = NetworkConfig(**{**tw_cfg.__dict__, par: v})
            rr = run_two_family_scan(
                twisted_cfg=tw_cfg,
                ring_cfg=baseline_ring,
                n_samples=240,
                seed=baseline_seed + par_offset[par] + int(10000 * float(v)),
            )
            local_points.append(
                {
                    "label": f"{par}={v}",
                    "vary": par,
                    "value": v,
                    "twisted_ring": rr["twisted_ring"],
                    "ring": rr["ring"],
                    "twisted_m2_eV_calibrated": alpha_ref * float(rr["twisted_ring"]["m2_proxy_mean"]),
                }
            )

    # Pick best local point by robustness then joint.
    ranked_local = sorted(
        local_points,
        key=lambda x: (
            x["twisted_ring"]["robustness_pass_fraction"],
            x["twisted_ring"]["joint_fraction"],
        ),
        reverse=True,
    )
    best_local = ranked_local[0]

    # 2) Scale scan.
    scale_entries = []
    for n in [64, 128, 192, 256]:
        tw_cfg = NetworkConfig(
            family="twisted_ring",
            n=n,
            edge_aniso=best_local["twisted_ring"]["best_sample_by_m2_proxy"]["config"]["edge_aniso"],
            drop_prob=best_local["twisted_ring"]["best_sample_by_m2_proxy"]["config"]["drop_prob"],
            onsite_noise=best_local["twisted_ring"]["best_sample_by_m2_proxy"]["config"]["onsite_noise"],
            k_link=best_local["twisted_ring"]["best_sample_by_m2_proxy"]["config"]["k_link"],
            eps_pin=best_local["twisted_ring"]["best_sample_by_m2_proxy"]["config"]["eps_pin"],
        )
        ring_cfg = NetworkConfig(family="ring", n=n)
        rr = run_two_family_scan(
            twisted_cfg=tw_cfg,
            ring_cfg=ring_cfg,
            n_samples=240,
            seed=baseline_seed + 50 + n,
        )
        scale_entries.append(
            {
                "N": n,
                "twisted_ring": rr["twisted_ring"],
                "ring": rr["ring"],
                "twisted_m2_eV_calibrated": alpha_ref * float(rr["twisted_ring"]["m2_proxy_mean"]),
                "ring_m2_eV_calibrated": alpha_ref * float(rr["ring"]["m2_proxy_mean"]),
            }
        )

    # 3) Multi-seed robustness around best local setting.
    multi_seed = []
    seeds = [20260312 + i for i in range(8)]
    for s in seeds:
        tw_cfg = NetworkConfig(
            family="twisted_ring",
            n=128,
            edge_aniso=best_local["twisted_ring"]["best_sample_by_m2_proxy"]["config"]["edge_aniso"],
            drop_prob=best_local["twisted_ring"]["best_sample_by_m2_proxy"]["config"]["drop_prob"],
            onsite_noise=best_local["twisted_ring"]["best_sample_by_m2_proxy"]["config"]["onsite_noise"],
            k_link=best_local["twisted_ring"]["best_sample_by_m2_proxy"]["config"]["k_link"],
            eps_pin=best_local["twisted_ring"]["best_sample_by_m2_proxy"]["config"]["eps_pin"],
        )
        rr = run_two_family_scan(
            twisted_cfg=tw_cfg,
            ring_cfg=baseline_ring,
            n_samples=220,
            seed=s,
        )
        multi_seed.append(
            {
                "seed": s,
                "twisted_joint": rr["twisted_ring"]["joint_fraction"],
                "twisted_robust": rr["twisted_ring"]["robustness_pass_fraction"],
                "twisted_m2_eV_calibrated": alpha_ref * float(rr["twisted_ring"]["m2_proxy_mean"]),
                "ring_joint": rr["ring"]["joint_fraction"],
                "ring_robust": rr["ring"]["robustness_pass_fraction"],
            }
        )

    tw_joint = np.array([x["twisted_joint"] for x in multi_seed], dtype=float)
    tw_rob = np.array([x["twisted_robust"] for x in multi_seed], dtype=float)
    tw_m2 = np.array([x["twisted_m2_eV_calibrated"] for x in multi_seed], dtype=float)
    ring_joint = np.array([x["ring_joint"] for x in multi_seed], dtype=float)
    ring_rob = np.array([x["ring_robust"] for x in multi_seed], dtype=float)

    stats = {
        "twisted_joint_mean": float(np.mean(tw_joint)),
        "twisted_joint_p16_p84": [float(np.percentile(tw_joint, 16)), float(np.percentile(tw_joint, 84))],
        "twisted_robust_mean": float(np.mean(tw_rob)),
        "twisted_robust_p16_p84": [float(np.percentile(tw_rob, 16)), float(np.percentile(tw_rob, 84))],
        "twisted_m2_mean_eV": float(np.mean(tw_m2)),
        "twisted_m2_p16_p84_eV": [float(np.percentile(tw_m2, 16)), float(np.percentile(tw_m2, 84))],
        "ring_joint_mean": float(np.mean(ring_joint)),
        "ring_robust_mean": float(np.mean(ring_rob)),
    }

    criteria = {
        "target_twisted_robust_min": 0.20,
        "target_twisted_joint_min": 0.35,
        "target_m2_log10_dev_max": 0.5,
    }
    log_dev = abs(np.log10(max(stats["twisted_m2_mean_eV"], 1.0e-40) / 1.0e-22))
    decision = {
        "twisted_robust_pass": bool(stats["twisted_robust_mean"] >= criteria["target_twisted_robust_min"]),
        "twisted_joint_pass": bool(stats["twisted_joint_mean"] >= criteria["target_twisted_joint_min"]),
        "twisted_m2_pass": bool(log_dev <= criteria["target_m2_log10_dev_max"]),
        "ring_control_passive": bool(stats["ring_joint_mean"] < 0.05 and stats["ring_robust_mean"] < 0.02),
        "twisted_m2_log10_dev": float(log_dev),
    }
    decision["overall_go"] = bool(
        decision["twisted_robust_pass"]
        and decision["twisted_joint_pass"]
        and decision["twisted_m2_pass"]
        and decision["ring_control_passive"]
    )

    summary = {
        "v2_settings": {
            "priority": ["twisted_ring", "ring", "scale"],
            "baseline_seed": baseline_seed,
            "baseline_samples": baseline_samples,
            "alpha_ref_eV_per_proxy": float(alpha_ref),
        },
        "baseline": {
            "twisted_ring": baseline["twisted_ring"],
            "ring": baseline["ring"],
        },
        "local_sensitivity": {
            "points": local_points,
            "best_point": best_local,
        },
        "scale_scan": scale_entries,
        "multi_seed": {
            "seeds": seeds,
            "records": multi_seed,
            "stats": stats,
        },
        "success_criteria": criteria,
        "decision": decision,
    }

    out = out_data / "phi1_phi2_emergence_v2_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[INFO] Wrote:", out)


if __name__ == "__main__":
    main()
