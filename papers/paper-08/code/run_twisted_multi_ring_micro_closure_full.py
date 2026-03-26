#!/usr/bin/env python3
"""More serious dedicated micro-closure attempt for twisted_multi_ring."""

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
    micro_closure_proxy,
)


DATA_DIR = REPO_ROOT / "papers" / "paper-08" / "data"
FIG_DIR = REPO_ROOT / "papers" / "paper-08" / "figs"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    base = TwistedMultiRingBridgeConfig(
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

    eta_grid = [0.10, 0.20, 0.30, 0.40]
    twist_grid = [0.00, 0.08, 0.12, 0.16, 0.20]
    records = []
    for i, eta in enumerate(eta_grid):
        for j, itw in enumerate(twist_grid):
            cfg = TwistedMultiRingBridgeConfig(
                loops=base.loops,
                n_per_loop=base.n_per_loop,
                inter_loop=base.inter_loop,
                intra_loop=base.intra_loop,
                twist_shift=base.twist_shift,
                edge_aniso=base.edge_aniso,
                onsite_noise=base.onsite_noise,
                eps_pin=base.eps_pin,
                inertia_eta=float(eta),
                inertia_twist=float(itw),
            )
            out = micro_closure_proxy(cfg, seed=20260360 + 100 * i + j)
            rec = {
                "inertia_eta": float(eta),
                "inertia_twist": float(itw),
                "orth_err_max": float(out["orth_err_max"]),
                "h_reconstruction_rel_err": float(out["h_reconstruction_rel_err"]),
                "lambda1": float(out["lambda1"]),
                "m0_sq": float(out["effective_mode1"]["m0_sq"]),
                "projection13_err_mean": float(out["projection_error_mode13"]["rel_err_mean"]),
                "checks": out["checks"],
                "all_pass": bool(out["checks"]["diag_exact"] and out["checks"]["projection_good"]),
            }
            records.append(rec)

    best = min(records, key=lambda r: (r["projection13_err_mean"], r["m0_sq"]))
    pass_count = int(sum(int(r["all_pass"]) for r in records))
    summary = {
        "status": "dedicated twisted_multi_ring micro-closure attempt",
        "base_config": base.__dict__,
        "grid": {"inertia_eta": eta_grid, "inertia_twist": twist_grid},
        "records": records,
        "n_all_pass": pass_count,
        "best_record": best,
        "verdict": "micro_closure_partial_supported" if pass_count > 0 else "micro_closure_still_open",
    }
    out = DATA_DIR / "twisted_multi_ring_micro_closure_full_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[INFO] Wrote:", out)

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    xs = [r["inertia_eta"] for r in records]
    ys = [r["inertia_twist"] for r in records]
    cs = [r["projection13_err_mean"] for r in records]
    sc = ax.scatter(xs, ys, c=cs, cmap="viridis", s=70)
    ax.set_xlabel("inertia_eta")
    ax.set_ylabel("inertia_twist")
    ax.set_title("Twisted multi-ring micro-closure grid")
    cb = plt.colorbar(sc)
    cb.set_label("projection13_err_mean")
    ax.grid(ls=":", alpha=0.4)
    fig.tight_layout()
    f1 = FIG_DIR / "twisted_multi_ring_micro_closure_grid.png"
    fig.savefig(f1, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("[INFO] Wrote:", f1)


if __name__ == "__main__":
    main()
