#!/usr/bin/env python3
"""Heavier non-proxy bridge program for twisted_multi_ring."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
mpl_dir = REPO_ROOT / ".tmp_mpl"
mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CORE_LIB = REPO_ROOT / "core" / "lib"
if str(CORE_LIB) not in sys.path:
    sys.path.insert(0, str(CORE_LIB))

from tcvphi.phi1_emergence_diagnostics import run_strict_homogeneous_uldm_test  # noqa: E402
from tcvphi.phi1_twisted_multi_ring_bridge import (  # noqa: E402
    TwistedMultiRingBridgeConfig,
    canonical_reduction_metrics,
    emergence_family_stats,
    family_stats_over_sizes,
)


DATA_DIR = REPO_ROOT / "papers" / "paper-08" / "data"
FIG_DIR = REPO_ROOT / "papers" / "paper-08" / "figs"


def fit_power_law(n: np.ndarray, y: np.ndarray) -> dict:
    mask = (n > 0) & (y > 0)
    x = np.log(n[mask])
    z = np.log(y[mask])
    b1, b0 = np.polyfit(x, z, 1)
    y_fit = np.exp(b0 + b1 * x)
    ss_res = float(np.sum((z - (b0 + b1 * x)) ** 2))
    ss_tot = float(np.sum((z - np.mean(z)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1.0e-30)
    return {"slope": float(b1), "exponent_positive": float(-b1), "amplitude": float(np.exp(b0)), "r2_logfit": float(r2), "y_fit": y_fit.tolist()}


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    v21 = json.loads((DATA_DIR / "phi1_phi2_emergence_v21_summary.json").read_text(encoding="utf-8"))
    alpha_ref = float(v21["alpha_ref_eV_per_proxy"])

    cfg = TwistedMultiRingBridgeConfig(
        loops=3,
        n_per_loop=32,
        inter_loop=0.13,
        intra_loop=1.0,
        twist_shift=1,
        edge_aniso=0.06,
        onsite_noise=0.03,
        eps_pin=1.0e-4,
        inertia_eta=0.20,
        inertia_twist=0.12,
    )

    emergence = emergence_family_stats(cfg, n_samples=320, seed=20260331)
    canon = canonical_reduction_metrics(cfg, seed=20260332, n_samples=96)
    homo = run_strict_homogeneous_uldm_test(ln_a_min=-18.0, ln_a_max=0.0, n_steps=7000, m_over_h0=500.0)

    n_grid = [16, 24, 32, 40, 48]
    rows = family_stats_over_sizes(cfg, n_grid=n_grid, n_samples=160, seed0=20260340)
    n_arr = np.array([r["n_total"] for r in rows], dtype=float)
    m_arr = np.array([r["m2_proxy_mean"] for r in rows], dtype=float)
    fit = fit_power_law(n_arr, m_arr)

    m2_eV = alpha_ref * float(emergence["m2_proxy_mean"])
    log_dev = abs(np.log10(max(m2_eV, 1.0e-40) / 1.0e-22))

    checks = {
        "joint_pass": bool(emergence["joint_fraction"] >= 0.35),
        "robust_pass": bool(emergence["robustness_pass_fraction"] >= 0.20),
        "m2_pass": bool(log_dev <= 0.5),
        "canonical_pass": bool(canon["pass_flags"]["orthogonality_good"] and canon["pass_flags"]["canonical_norm_good"]),
        "capture_pass": bool(canon["pass_flags"]["mode13_capture_good"]),
        "homogeneous_pass": bool(homo["pass_flags"]["slope_close_to_minus3"] and homo["pass_flags"]["rho_a3_quasi_constant"]),
        "scaling_pass": bool(fit["r2_logfit"] >= 0.85),
    }
    overall = bool(all(checks.values()))

    summary = {
        "status": "heavier twisted_multi_ring non-proxy bridge",
        "config": cfg.__dict__,
        "emergence": emergence,
        "canonical": canon,
        "homogeneous": {
            "diagnostics": homo["diagnostics"],
            "pass_flags": homo["pass_flags"],
        },
        "size_scaling": {
            "rows": rows,
            "fit": fit,
        },
        "calibrated_m2_eV": float(m2_eV),
        "m2_log10_dev_from_1e-22": float(log_dev),
        "checks": checks,
        "overall_support": overall,
        "notes": [
            "This is the stronger bridge layer for twisted_multi_ring.",
            "It is still internal and additive, but it goes well beyond the first dedicated bridge proxy.",
        ],
    }
    out = DATA_DIR / "twisted_multi_ring_full_bridge_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[INFO] Wrote:", out)

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    labels = ["joint", "robust", "canonical", "capture", "homog.", "scaling"]
    vals = [
        float(emergence["joint_fraction"]),
        float(emergence["robustness_pass_fraction"]),
        1.0 if checks["canonical_pass"] else 0.0,
        1.0 if checks["capture_pass"] else 0.0,
        1.0 if checks["homogeneous_pass"] else 0.0,
        min(1.0, max(0.0, fit["r2_logfit"])),
    ]
    ax.bar(np.arange(len(labels)), vals, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"])
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("score / pass")
    ax.set_title("Twisted multi-ring full bridge")
    ax.grid(axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    f1 = FIG_DIR / "twisted_multi_ring_full_bridge_dashboard.png"
    fig.savefig(f1, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("[INFO] Wrote:", f1)


if __name__ == "__main__":
    main()
