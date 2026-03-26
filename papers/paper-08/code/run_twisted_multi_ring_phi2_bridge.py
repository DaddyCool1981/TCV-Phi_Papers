#!/usr/bin/env python3
"""Dedicated Phi2 bridge for the twisted_multi_ring flavour candidate."""

from __future__ import annotations

import json
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

from tcvphi.phi1_twisted_multi_ring_bridge import (  # noqa: E402
    TwistedMultiRingBridgeConfig,
    emergence_family_stats,
    micro_closure_proxy,
)


DATA_DIR = REPO_ROOT / "papers" / "paper-08" / "data"
FIG_DIR = REPO_ROOT / "papers" / "paper-08" / "figs"


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

    stats = emergence_family_stats(cfg, n_samples=260, seed=20260321)
    micro = micro_closure_proxy(cfg, seed=20260322)
    m2_eV = alpha_ref * float(stats["m2_proxy_mean"])
    log_dev = abs(np.log10(max(m2_eV, 1.0e-40) / 1.0e-22))

    checks = {
        "joint_pass": bool(stats["joint_fraction"] >= 0.35),
        "robust_pass": bool(stats["robustness_pass_fraction"] >= 0.20),
        "m2_pass": bool(log_dev <= 0.5),
        "micro_projection_good": bool(micro["checks"]["projection_good"]),
    }
    overall = bool(checks["joint_pass"] and checks["robust_pass"] and checks["m2_pass"])

    summary = {
        "status": "dedicated twisted_multi_ring Phi2 bridge",
        "config": cfg.__dict__,
        "emergence": stats,
        "calibrated_m2_eV": float(m2_eV),
        "m2_log10_dev_from_1e-22": float(log_dev),
        "micro_closure_proxy": micro,
        "checks": checks,
        "overall_bridge_support": overall,
        "notes": [
            "This is the dedicated bridge follow-up for the flavour-leading twisted_multi_ring toy.",
            "Micro-closure here is still a proxy-level generalized-mode check, not a full dedicated closure program.",
        ],
    }
    out = DATA_DIR / "twisted_multi_ring_phi2_bridge_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[INFO] Wrote:", out)

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    labels = ["joint", "robust", "m2 pass", "micro proj."]
    vals = [
        float(stats["joint_fraction"]),
        float(stats["robustness_pass_fraction"]),
        1.0 if checks["m2_pass"] else 0.0,
        1.0 if checks["micro_projection_good"] else 0.0,
    ]
    ax.bar(np.arange(len(labels)), vals, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("score / pass")
    ax.set_title("Twisted multi-ring bridge diagnostics")
    ax.grid(axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    f1 = FIG_DIR / "twisted_multi_ring_phi2_bridge_dashboard.png"
    fig.savefig(f1, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("[INFO] Wrote:", f1)


if __name__ == "__main__":
    main()
