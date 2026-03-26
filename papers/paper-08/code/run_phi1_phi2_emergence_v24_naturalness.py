#!/usr/bin/env python3
"""v2.4 naturalness test for emergent m2 scale (non-calibration-first)."""

from __future__ import annotations

import json
import math
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

from tcvphi.phi1_emergence import NetworkConfig, sample_network  # noqa: E402


def estimate_proxy_stats(cfg: NetworkConfig, n_samples: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    vals = []
    joint_hits = 0
    for _ in range(n_samples):
        s = sample_network(cfg, rng)
        lm = s["light_mode"]
        vals.append(float(lm["m2_proxy"]))
        soft = lm["epsilon_soft"] < 0.03
        gap = lm["gap_ratio_12"] < 0.15
        coll = lm["participation_ratio"] > 0.25 * s["N"]
        joint_hits += int(soft and gap and coll)
    arr = np.array(vals, dtype=float)
    return {
        "m2_proxy_mean": float(np.mean(arr)),
        "m2_proxy_p16_p84": [float(np.percentile(arr, 16)), float(np.percentile(arr, 84))],
        "sqrt_proxy_mean": float(np.mean(np.sqrt(np.maximum(arr, 1.0e-30)))),
        "joint_fraction": float(joint_hits / max(n_samples, 1)),
    }


def overlap_fraction_log_uniform(
    s: float,
    lambda_min: float,
    lambda_max: float,
    m_min: float,
    m_max: float,
) -> float:
    """Fraction of log-uniform Lambda prior leading to ULDM mass window.

    Model: m2_pred = Lambda_coh * s, where s is dimensionless suppression.
    """
    if s <= 0.0:
        return 0.0
    lo = m_min / s
    hi = m_max / s
    a = max(lo, lambda_min)
    b = min(hi, lambda_max)
    if b <= a:
        return 0.0
    return float((math.log10(b) - math.log10(a)) / (math.log10(lambda_max) - math.log10(lambda_min)))


def main() -> None:
    out_data = REPO_ROOT / "papers" / "paper-08" / "data"
    out_data.mkdir(parents=True, exist_ok=True)

    v21 = json.loads((out_data / "phi1_phi2_emergence_v21_summary.json").read_text())
    best = v21["best_record"]["params"]

    best_cfg = NetworkConfig(
        family="twisted_ring",
        n=128,
        edge_aniso=float(best["edge_aniso"]),
        drop_prob=float(best["drop_prob"]),
        onsite_noise=float(best["onsite_noise"]),
        k_link=1.0,
        eps_pin=1.0e-4,
    )

    # 1) Structural scale law: N-scaling.
    n_grid = [64, 96, 128, 160, 192, 256]
    scale_rows = []
    for i, n in enumerate(n_grid):
        cfg = NetworkConfig(
            family="twisted_ring",
            n=n,
            edge_aniso=best_cfg.edge_aniso,
            drop_prob=best_cfg.drop_prob,
            onsite_noise=best_cfg.onsite_noise,
            k_link=best_cfg.k_link,
            eps_pin=best_cfg.eps_pin,
        )
        st = estimate_proxy_stats(cfg, n_samples=180, seed=20260500 + i)
        scale_rows.append({"N": n, **st})

    # Fit sqrt(m2_proxy) ~ A N^{-alpha}
    n_arr = np.array([r["N"] for r in scale_rows], dtype=float)
    s_arr = np.array([r["sqrt_proxy_mean"] for r in scale_rows], dtype=float)
    ok = (n_arr > 0.0) & (s_arr > 0.0)
    coef = np.polyfit(np.log(n_arr[ok]), np.log(s_arr[ok]), 1)
    alpha = float(-coef[0])

    # 2) Local sensitivity (finite differences around best point).
    base = estimate_proxy_stats(best_cfg, n_samples=220, seed=20260540)
    s0 = max(base["sqrt_proxy_mean"], 1.0e-30)

    def vary(cfg: NetworkConfig, name: str, factor: float) -> NetworkConfig:
        vals = dict(cfg.__dict__)
        vals[name] = vals[name] * factor
        if name == "n":
            vals[name] = int(round(vals[name]))
            vals[name] = max(vals[name], 32)
            vals[name] += vals[name] % 2
        return NetworkConfig(**vals)

    sens_params = ["edge_aniso", "drop_prob", "onsite_noise", "k_link", "eps_pin", "n"]
    sens_rows = []
    for i, p in enumerate(sens_params):
        c_up = vary(best_cfg, p, 1.10)
        c_dn = vary(best_cfg, p, 0.90)
        s_up = estimate_proxy_stats(c_up, n_samples=140, seed=20260600 + 2 * i)["sqrt_proxy_mean"]
        s_dn = estimate_proxy_stats(c_dn, n_samples=140, seed=20260601 + 2 * i)["sqrt_proxy_mean"]
        # symmetric log-derivative
        num = math.log(max(s_up, 1.0e-30)) - math.log(max(s_dn, 1.0e-30))
        den = math.log(1.10) - math.log(0.90)
        delta = abs(num / den)
        sens_rows.append({"param": p, "delta_log_sensitivity": float(delta), "s_up": float(s_up), "s_dn": float(s_dn)})

    max_delta = float(max(r["delta_log_sensitivity"] for r in sens_rows))

    # 3) UV->IR dimensional budget without using a single-point calibration.
    #    m2_pred = Lambda_coh * sqrt(m2_proxy), Lambda_coh log-uniform prior.
    lambda_min = 1.0e-23
    lambda_max = 1.0e-18
    m_min = 1.0e-23
    m_max = 1.0e-21

    # Evaluate on all GO points from v21 for structural region estimate.
    go_points = [r for r in v21["records"] if r.get("overall_go")]
    overlap_rows = []
    for i, r in enumerate(go_points):
        p = r["params"]
        cfg = NetworkConfig(
            family="twisted_ring",
            n=128,
            edge_aniso=float(p["edge_aniso"]),
            drop_prob=float(p["drop_prob"]),
            onsite_noise=float(p["onsite_noise"]),
            k_link=1.0,
            eps_pin=1.0e-4,
        )
        st = estimate_proxy_stats(cfg, n_samples=110, seed=20260700 + i)
        s = max(st["sqrt_proxy_mean"], 1.0e-30)
        frac = overlap_fraction_log_uniform(s, lambda_min, lambda_max, m_min, m_max)
        overlap_rows.append({
            "params": p,
            "sqrt_proxy_mean": float(s),
            "joint_fraction": float(st["joint_fraction"]),
            "uldm_prior_overlap_fraction": float(frac),
        })

    overlaps = np.array([r["uldm_prior_overlap_fraction"] for r in overlap_rows], dtype=float)
    overlap_summary = {
        "median": float(np.median(overlaps)) if overlaps.size else 0.0,
        "p16_p84": [float(np.percentile(overlaps, 16)), float(np.percentile(overlaps, 84))] if overlaps.size else [0.0, 0.0],
        "fraction_above_0p10": float(np.mean(overlaps >= 0.10)) if overlaps.size else 0.0,
    }

    # Naturalness decision gates (explicit and conservative).
    gates = {
        "alpha_min": 0.50,
        "max_delta_max": 5.0,
        "overlap_median_min": 0.10,
        "overlap_fraction_above_0p10_min": 0.50,
    }
    checks = {
        "size_scaling_present": bool(alpha >= gates["alpha_min"]),
        "no_extreme_fine_tuning": bool(max_delta <= gates["max_delta_max"]),
        "uldm_overlap_median_ok": bool(overlap_summary["median"] >= gates["overlap_median_min"]),
        "uldm_overlap_volume_ok": bool(overlap_summary["fraction_above_0p10"] >= gates["overlap_fraction_above_0p10_min"]),
    }

    if all(checks.values()):
        verdict = "naturalness_supported"
    elif checks["size_scaling_present"] and (checks["uldm_overlap_median_ok"] or checks["uldm_overlap_volume_ok"]):
        verdict = "naturalness_promising_but_not_proven"
    else:
        verdict = "naturalness_not_supported_yet"

    out = {
        "assumptions": {
            "mass_model": "m2_pred = Lambda_coh * sqrt(m2_proxy)",
            "lambda_prior_eV": [lambda_min, lambda_max],
            "uldm_window_eV": [m_min, m_max],
            "best_v21_params": best,
        },
        "size_scaling": {
            "rows": scale_rows,
            "fit_model": "sqrt(m2_proxy) ~ A * N^{-alpha}",
            "alpha": alpha,
        },
        "local_sensitivity": {
            "rows": sens_rows,
            "max_delta_log_sensitivity": max_delta,
            "base_sqrt_proxy": s0,
        },
        "uldm_overlap": {
            "go_points_evaluated": len(overlap_rows),
            "rows": overlap_rows,
            "summary": overlap_summary,
        },
        "gates": gates,
        "checks": checks,
        "verdict": verdict,
    }

    out_path = out_data / "phi1_phi2_emergence_v24_naturalness_summary.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("[INFO] Wrote:", out_path)


if __name__ == "__main__":
    main()
