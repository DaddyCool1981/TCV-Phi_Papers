#!/usr/bin/env python3
"""v26: try non-diagonal topology-linked inertia to close micro-closure gap."""

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

from tcvphi.phi1_micro_closure import (  # noqa: E402
    MicroClosureConfig,
    effective_single_mode_params,
    generalized_modes,
    low_mode_projection_error,
    micro_matrices,
    twisted_source_bank,
)


def evaluate(cfg: MicroClosureConfig, seed: int) -> dict:
    mats = micro_matrices(cfg, seed=seed)
    k = np.asarray(mats["K"], dtype=float)
    h = np.asarray(mats["H"], dtype=float)
    gm = generalized_modes(k, h)
    vals = np.asarray(gm["lambda"], dtype=float)
    v = np.asarray(gm["V"], dtype=float)

    v1 = v[:, 1]
    eff1 = effective_single_mode_params(k, h, v1, lambda4=float(mats["lambda4"]))

    if cfg.family == "twisted_ring":
        bank = twisted_source_bank(cfg.n)
    else:
        bank = {"uniform": np.ones(cfg.n, dtype=float)}

    cap13 = {}
    for name, src in bank.items():
        src = src.astype(float)
        src /= max(float(np.sqrt(src.T @ k @ src)), 1.0e-20)
        amps = v.T @ (k @ src)
        p = np.abs(amps) ** 2
        cap13[name] = float(np.sum(p[1:4]) / max(float(np.sum(p)), 1.0e-20))

    best_src = max(cap13, key=cap13.get)
    err3 = low_mode_projection_error(k, h, vals, v, n_keep=4, source=bank[best_src])

    return {
        "orth_err_max": gm["orth_err_max"],
        "h_reconstruction_rel_err": gm["h_reconstruction_rel_err"],
        "lambda1": float(vals[1]),
        "m0_sq": float(eff1["m0_sq"]),
        "capture13_max": float(max(cap13.values())),
        "projection13_err_mean": float(err3["rel_err_mean"]),
    }


def main() -> None:
    out_data = REPO_ROOT / "papers" / "paper-08" / "data"
    out_data.mkdir(parents=True, exist_ok=True)

    v21 = json.loads((out_data / "phi1_phi2_emergence_v21_summary.json").read_text())
    p = v21["best_record"]["params"]

    eta_grid = [0.1, 0.2, 0.3, 0.4]
    twist_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    records = []
    for i, eta in enumerate(eta_grid):
        for j, etw in enumerate(twist_grid):
            tw_cfg = MicroClosureConfig(
                family="twisted_ring",
                n=128,
                edge_aniso=float(p["edge_aniso"]),
                drop_prob=float(p["drop_prob"]),
                onsite_noise=float(p["onsite_noise"]),
                k_link=1.0,
                mu2=1.0e-4,
                lambda4=1.0e-2,
                inertia_eta=float(eta),
                inertia_twist=float(etw),
            )
            rg_cfg = MicroClosureConfig(
                family="ring",
                n=128,
                edge_aniso=float(p["edge_aniso"]),
                drop_prob=float(p["drop_prob"]),
                onsite_noise=float(p["onsite_noise"]),
                k_link=1.0,
                mu2=1.0e-4,
                lambda4=1.0e-2,
                inertia_eta=float(eta),
                inertia_twist=0.0,
            )
            tw = evaluate(tw_cfg, seed=20262000 + 100 * i + j)
            rg = evaluate(rg_cfg, seed=20262500 + 100 * i + j)

            checks = {
                "diag_exact": bool(tw["orth_err_max"] < 1.0e-10 and tw["h_reconstruction_rel_err"] < 1.0e-10),
                "capture_high": bool(tw["capture13_max"] > 0.75),
                "projection_good": bool(tw["projection13_err_mean"] < 0.20),
                "m0sq_lower_than_ring": bool(tw["m0_sq"] < rg["m0_sq"]),
            }
            rec = {
                "inertia_eta": float(eta),
                "inertia_twist": float(etw),
                "twisted": tw,
                "ring": rg,
                "checks": checks,
                "all_pass": bool(all(checks.values())),
                "m0sq_ratio_tw_over_ring": float(tw["m0_sq"] / max(rg["m0_sq"], 1.0e-30)),
            }
            records.append(rec)

    ranked = sorted(records, key=lambda r: (int(r["all_pass"]), -r["m0sq_ratio_tw_over_ring"]) , reverse=True)
    # Better secondary ordering: lower ratio preferred
    ranked = sorted(ranked, key=lambda r: (int(r["all_pass"]), -r["checks"]["capture_high"], -r["checks"]["projection_good"], r["m0sq_ratio_tw_over_ring"]), reverse=True)

    pass_records = [r for r in records if r["all_pass"]]
    best = min(records, key=lambda r: r["m0sq_ratio_tw_over_ring"])

    out = {
        "best_v21_params": p,
        "grid": {"inertia_eta": eta_grid, "inertia_twist": twist_grid},
        "records": records,
        "n_all_pass": int(len(pass_records)),
        "best_ratio_record": best,
        "verdict": "micro_closure_supported_v26" if len(pass_records) > 0 else "micro_closure_still_open_v26",
    }

    out_path = out_data / "phi1_phi2_micro_closure_v26_summary.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("[INFO] Wrote:", out_path)


if __name__ == "__main__":
    main()
