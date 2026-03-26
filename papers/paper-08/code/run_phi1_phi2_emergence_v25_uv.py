#!/usr/bin/env python3
"""v2.5 UV-compact consistency: analytic scaling vs numeric and UV->IR closure."""

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


def family_stats(cfg: NetworkConfig, n_samples: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    l1, z0, mproxy = [], [], []
    for _ in range(n_samples):
        s = sample_network(cfg, rng)
        lm = s["light_mode"]
        l1.append(float(lm["lambda0"]))
        z0.append(float(lm["z0_proxy"]))
        mproxy.append(float(lm["m2_proxy"]))
    l1a = np.array(l1, dtype=float)
    z0a = np.array(z0, dtype=float)
    ma = np.array(mproxy, dtype=float)
    return {
        "lambda1_mean": float(np.mean(l1a)),
        "lambda1_p16_p84": [float(np.percentile(l1a, 16)), float(np.percentile(l1a, 84))],
        "z0_mean": float(np.mean(z0a)),
        "z0_p16_p84": [float(np.percentile(z0a, 16)), float(np.percentile(z0a, 84))],
        "m2_proxy_mean": float(np.mean(ma)),
        "m2_proxy_p16_p84": [float(np.percentile(ma, 16)), float(np.percentile(ma, 84))],
    }


def fit_power_law(n: np.ndarray, y: np.ndarray) -> dict:
    mask = (n > 0) & (y > 0)
    x = np.log(n[mask])
    z = np.log(y[mask])
    b1, b0 = np.polyfit(x, z, 1)
    y_fit = np.exp(b0 + b1 * x)
    ss_res = float(np.sum((z - (b0 + b1 * x)) ** 2))
    ss_tot = float(np.sum((z - np.mean(z)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1.0e-30)
    return {
        "slope": float(b1),
        "exponent_positive": float(-b1),
        "amplitude": float(np.exp(b0)),
        "r2_logfit": float(r2),
        "n": n[mask].tolist(),
        "y_fit": y_fit.tolist(),
    }


def main() -> None:
    out_data = REPO_ROOT / "papers" / "paper-08" / "data"
    out_data.mkdir(parents=True, exist_ok=True)

    v21 = json.loads((out_data / "phi1_phi2_emergence_v21_summary.json").read_text())
    bp = v21["best_record"]["params"]

    n_grid = [64, 96, 128, 160, 192, 224, 256]
    rows = []
    for i, n in enumerate(n_grid):
        tw_cfg = NetworkConfig(
            family="twisted_ring",
            n=n,
            edge_aniso=float(bp["edge_aniso"]),
            drop_prob=float(bp["drop_prob"]),
            onsite_noise=float(bp["onsite_noise"]),
            k_link=1.0,
            eps_pin=1.0e-4,
        )
        rg_cfg = NetworkConfig(
            family="ring",
            n=n,
            edge_aniso=float(bp["edge_aniso"]),
            drop_prob=float(bp["drop_prob"]),
            onsite_noise=float(bp["onsite_noise"]),
            k_link=1.0,
            eps_pin=1.0e-4,
        )
        tw = family_stats(tw_cfg, n_samples=200, seed=20260800 + i)
        rg = family_stats(rg_cfg, n_samples=200, seed=20260900 + i)
        rows.append({"N": n, "twisted_ring": tw, "ring": rg})

    n_arr = np.array([r["N"] for r in rows], dtype=float)
    lam_tw = np.array([r["twisted_ring"]["lambda1_mean"] for r in rows], dtype=float)
    lam_rg = np.array([r["ring"]["lambda1_mean"] for r in rows], dtype=float)
    z0_tw = np.array([r["twisted_ring"]["z0_mean"] for r in rows], dtype=float)
    z0_rg = np.array([r["ring"]["z0_mean"] for r in rows], dtype=float)

    fit_lam_tw = fit_power_law(n_arr, lam_tw)
    fit_lam_rg = fit_power_law(n_arr, lam_rg)
    fit_z0_tw = fit_power_law(n_arr, z0_tw)
    fit_z0_rg = fit_power_law(n_arr, z0_rg)

    # Compact analytic model (effective):
    # lambda1 ~ c_lambda * N^{-p}, z0 ~ c_z * N^{q},
    # m2_proxy = lambda1/z0 ~ c_m * N^{-(p+q)}
    p_tw = fit_lam_tw["exponent_positive"]
    q_tw = -fit_z0_tw["slope"]  # z0 ~ N^{q}
    p_rg = fit_lam_rg["exponent_positive"]
    q_rg = -fit_z0_rg["slope"]

    # Twist suppression ratio at N=128
    ref_idx = n_grid.index(128)
    twist_lambda_ratio = float(lam_tw[ref_idx] / max(lam_rg[ref_idx], 1.0e-30))
    twist_mproxy_ratio = float(
        rows[ref_idx]["twisted_ring"]["m2_proxy_mean"]
        / max(rows[ref_idx]["ring"]["m2_proxy_mean"], 1.0e-30)
    )
    ratio_vs_n = np.array(
        [
            r["twisted_ring"]["m2_proxy_mean"] / max(r["ring"]["m2_proxy_mean"], 1.0e-30)
            for r in rows
        ],
        dtype=float,
    )
    mask_large = n_arr >= 128
    ratio_large_median = float(np.median(ratio_vs_n[mask_large]))
    z0_tw_large = z0_tw[mask_large]
    z0_rg_large = z0_rg[mask_large]
    z0_plateau_rel_std = float(np.std(z0_tw_large) / max(np.mean(z0_tw_large), 1.0e-30))
    z0_tw_over_ring_large = float(np.mean(z0_tw_large) / max(np.mean(z0_rg_large), 1.0e-30))

    # UV closure test: m2 = Lambda_coh * sqrt(m2_proxy)
    # Derive Lambda_required from m_ref and measured proxy (no fixed calibration injected).
    m_ref = 1.0e-22
    go_points = [r for r in v21["records"] if r.get("overall_go")]
    lambda_required = []
    for i, r in enumerate(go_points):
        cfg = NetworkConfig(
            family="twisted_ring",
            n=128,
            edge_aniso=float(r["params"]["edge_aniso"]),
            drop_prob=float(r["params"]["drop_prob"]),
            onsite_noise=float(r["params"]["onsite_noise"]),
            k_link=1.0,
            eps_pin=1.0e-4,
        )
        st = family_stats(cfg, n_samples=120, seed=20261000 + i)
        s = math.sqrt(max(st["m2_proxy_mean"], 1.0e-30))
        lambda_required.append(m_ref / max(s, 1.0e-30))
    lam_req = np.array(lambda_required, dtype=float)

    uv_consistency = {
        "lambda_required_eV_median": float(np.median(lam_req)),
        "lambda_required_eV_p16_p84": [float(np.percentile(lam_req, 16)), float(np.percentile(lam_req, 84))],
        "log10_width_p84_over_p16": float(np.log10(np.percentile(lam_req, 84) / max(np.percentile(lam_req, 16), 1.0e-40))),
    }

    # Pass/fail table
    checks = {
        "lambda_scaling_power_law_good": bool(fit_lam_tw["r2_logfit"] >= 0.90),
        "z0_collective_plateau_good": bool(z0_plateau_rel_std <= 0.12 and z0_tw_over_ring_large >= 2.0),
        "twist_reduces_soft_mode_scale": bool(ratio_large_median < 0.70),
        "uv_closure_band_not_too_wide": bool(uv_consistency["log10_width_p84_over_p16"] <= 0.7),
    }

    if all(checks.values()):
        verdict = "uv_compact_consistency_supported"
    elif checks["lambda_scaling_power_law_good"] and checks["twist_reduces_soft_mode_scale"]:
        verdict = "uv_compact_consistency_promising"
    else:
        verdict = "uv_compact_consistency_not_supported"

    out = {
        "assumptions": {
            "compact_model": "lambda1~N^-p, z0~N^q, m2=Lambda_coh*sqrt(lambda1/z0)",
            "best_v21_params": bp,
            "m_ref_eV_for_lambda_required": m_ref,
        },
        "scale_rows": rows,
        "fits": {
            "lambda_twisted": fit_lam_tw,
            "lambda_ring": fit_lam_rg,
            "z0_twisted": fit_z0_tw,
            "z0_ring": fit_z0_rg,
            "effective_exponents": {
                "twisted": {"p_lambda": p_tw, "q_z0": q_tw, "p_plus_q": p_tw + q_tw},
                "ring": {"p_lambda": p_rg, "q_z0": q_rg, "p_plus_q": p_rg + q_rg},
            },
        },
        "twist_effect": {
            "lambda_ratio_twisted_over_ring_at_N128": twist_lambda_ratio,
            "m2proxy_ratio_twisted_over_ring_at_N128": twist_mproxy_ratio,
            "m2proxy_ratio_twisted_over_ring_median_Nge128": ratio_large_median,
        },
        "z0_behavior": {
            "z0_twisted_over_ring_mean_Nge128": z0_tw_over_ring_large,
            "z0_twisted_rel_std_Nge128": z0_plateau_rel_std,
            "z0_fit_r2_log_twisted": fit_z0_tw["r2_logfit"],
        },
        "uv_closure": uv_consistency,
        "checks": checks,
        "verdict": verdict,
    }

    out_path = out_data / "phi1_phi2_emergence_v25_uv_summary.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("[INFO] Wrote:", out_path)


if __name__ == "__main__":
    main()
