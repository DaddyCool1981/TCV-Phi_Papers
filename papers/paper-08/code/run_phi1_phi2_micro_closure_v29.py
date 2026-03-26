#!/usr/bin/env python3
"""v29: conservative no-fine-tuning test with one mixed twist coupling."""

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

    eff1 = effective_single_mode_params(k, h, v[:, 1], lambda4=float(mats["lambda4"]))

    bank = twisted_source_bank(cfg.n) if cfg.family == "twisted_ring" else {"uniform": np.ones(cfg.n, dtype=float)}
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
        "m0_sq": float(eff1["m0_sq"]),
        "capture13_max": float(max(cap13.values())),
        "projection13_err_mean": float(err3["rel_err_mean"]),
    }


def main() -> None:
    out_data = REPO_ROOT / "papers" / "paper-08" / "data"
    out_data.mkdir(parents=True, exist_ok=True)

    # Baseline from best v28 ratio point (fixed, no broad retuning).
    base = {
        "inertia_eta": 0.2,
        "inertia_twist": 0.0,
        "inertia_loop": 0.2,
        "potential_twist": 0.08,
    }

    v21 = json.loads((out_data / "phi1_phi2_emergence_v21_summary.json").read_text())
    p = v21["best_record"]["params"]

    mix_grid = [-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30]
    # Conservative robustness neighborhood, not a high-dimensional fit.
    mult = [0.9, 1.0, 1.1]

    records = []
    for i, mix in enumerate(mix_grid):
        for j, m_eta in enumerate(mult):
            for kidx, m_loop in enumerate(mult):
                tw_cfg = MicroClosureConfig(
                    family="twisted_ring",
                    n=128,
                    edge_aniso=float(p["edge_aniso"]),
                    drop_prob=float(p["drop_prob"]),
                    onsite_noise=float(p["onsite_noise"]),
                    k_link=1.0,
                    mu2=1.0e-4,
                    lambda4=1.0e-2,
                    inertia_eta=base["inertia_eta"] * m_eta,
                    inertia_twist=base["inertia_twist"],
                    inertia_loop=base["inertia_loop"] * m_loop,
                    potential_twist=base["potential_twist"],
                    mix_twist=float(mix),
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
                    inertia_eta=base["inertia_eta"] * m_eta,
                    inertia_twist=0.0,
                    inertia_loop=0.0,
                    potential_twist=0.0,
                    mix_twist=0.0,
                )

                tw = evaluate(tw_cfg, seed=20267000 + 1000 * i + 100 * j + kidx)
                rg = evaluate(rg_cfg, seed=20268000 + 1000 * i + 100 * j + kidx)

                checks = {
                    "diag_exact": bool(tw["orth_err_max"] < 1.0e-10 and tw["h_reconstruction_rel_err"] < 1.0e-10),
                    "capture_high": bool(tw["capture13_max"] > 0.75),
                    "projection_good": bool(tw["projection13_err_mean"] < 0.20),
                    "m0sq_lower_than_ring": bool(tw["m0_sq"] < rg["m0_sq"]),
                }
                all_pass = bool(all(checks.values()))
                records.append(
                    {
                        "mix_twist": float(mix),
                        "eta_mult": float(m_eta),
                        "loop_mult": float(m_loop),
                        "twisted": tw,
                        "ring": rg,
                        "checks": checks,
                        "all_pass": all_pass,
                        "m0sq_ratio_tw_over_ring": float(tw["m0_sq"] / max(rg["m0_sq"], 1.0e-30)),
                    }
                )

    pass_records = [r for r in records if r["all_pass"]]
    best = min(records, key=lambda r: r["m0sq_ratio_tw_over_ring"])

    # Naturalness criterion: not just one point; require >=20% pass in neighborhood for at least one mix value.
    per_mix = {}
    for mix in mix_grid:
        sub = [r for r in records if abs(r["mix_twist"] - mix) < 1e-12]
        frac = float(np.mean([1.0 if r["all_pass"] else 0.0 for r in sub]))
        per_mix[str(mix)] = {
            "pass_fraction": frac,
            "best_ratio": float(min(r["m0sq_ratio_tw_over_ring"] for r in sub)),
        }
    max_pass_frac = max(v["pass_fraction"] for v in per_mix.values())

    no_finetune_supported = bool(max_pass_frac >= 0.20)
    verdict = "micro_closure_supported_v29" if no_finetune_supported else "micro_closure_still_open_v29"

    out = {
        "base_parameters": base,
        "best_v21_params": p,
        "mix_grid": mix_grid,
        "multiplier_grid": mult,
        "n_records": len(records),
        "n_all_pass": len(pass_records),
        "best_ratio_record": best,
        "per_mix_summary": per_mix,
        "max_pass_fraction_over_mix": max_pass_frac,
        "no_finetune_supported": no_finetune_supported,
        "verdict": verdict,
        "records": records,
    }

    out_path = out_data / "phi1_phi2_micro_closure_v29_summary.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("[INFO] Wrote:", out_path)


if __name__ == "__main__":
    main()
