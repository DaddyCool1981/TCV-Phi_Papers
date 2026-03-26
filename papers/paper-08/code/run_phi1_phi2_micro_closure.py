#!/usr/bin/env python3
"""Run micro-closure UV->IR checks for one-field route."""

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


def evaluate_family(cfg: MicroClosureConfig, seed: int) -> dict:
    mats = micro_matrices(cfg, seed=seed)
    k = np.asarray(mats["K"], dtype=float)
    h = np.asarray(mats["H"], dtype=float)
    gm = generalized_modes(k, h)
    vals = np.asarray(gm["lambda"], dtype=float)
    v = np.asarray(gm["V"], dtype=float)

    # first non-trivial mode
    v1 = v[:, 1]
    eff1 = effective_single_mode_params(k, h, v1, lambda4=float(mats["lambda4"]))

    # capture with source bank
    if cfg.family == "twisted_ring":
        bank = twisted_source_bank(cfg.n)
    else:
        bank = {"uniform": np.ones(cfg.n, dtype=float)}

    cap1 = {}
    cap13 = {}
    for name, src in bank.items():
        src = src.astype(float)
        nrm = float(np.sqrt(src.T @ k @ src))
        src /= max(nrm, 1.0e-20)
        amps = v.T @ (k @ src)
        p = np.abs(amps) ** 2
        cap1[name] = float(p[1] / max(np.sum(p), 1.0e-20))
        cap13[name] = float(np.sum(p[1:4]) / max(np.sum(p), 1.0e-20))

    # low-mode truncation error with best source
    best_src_name = max(cap13, key=cap13.get)
    err1 = low_mode_projection_error(k, h, vals, v, n_keep=2, source=bank[best_src_name])
    err3 = low_mode_projection_error(k, h, vals, v, n_keep=4, source=bank[best_src_name])

    return {
        "family": cfg.family,
        "orth_err_max": gm["orth_err_max"],
        "h_reconstruction_rel_err": gm["h_reconstruction_rel_err"],
        "lambda1": float(vals[1]),
        "effective_mode1": eff1,
        "capture_mode1": cap1,
        "capture_mode13": cap13,
        "best_source": best_src_name,
        "projection_error_mode1": err1,
        "projection_error_mode13": err3,
    }


def main() -> None:
    out_data = REPO_ROOT / "papers" / "paper-08" / "data"
    out_data.mkdir(parents=True, exist_ok=True)

    v21 = json.loads((out_data / "phi1_phi2_emergence_v21_summary.json").read_text())
    p = v21["best_record"]["params"]

    tw_cfg = MicroClosureConfig(
        family="twisted_ring",
        n=128,
        edge_aniso=float(p["edge_aniso"]),
        drop_prob=float(p["drop_prob"]),
        onsite_noise=float(p["onsite_noise"]),
        k_link=1.0,
        mu2=1.0e-4,
        lambda4=1.0e-2,
        inertia_eta=0.2,
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
        inertia_eta=0.2,
    )

    tw = evaluate_family(tw_cfg, seed=20261100)
    rg = evaluate_family(rg_cfg, seed=20261150)

    checks = {
        "exact_generalized_diagonalization": bool(tw["orth_err_max"] < 1e-10 and tw["h_reconstruction_rel_err"] < 1e-10),
        "twisted_capture_mode13_high": bool(max(tw["capture_mode13"].values()) > 0.75),
        "twisted_lowmode_projection_good": bool(tw["projection_error_mode13"]["rel_err_mean"] < 0.20),
        "twisted_vs_ring_m0sq_lower": bool(
            tw["effective_mode1"]["m0_sq"] <= 0.75 * rg["effective_mode1"]["m0_sq"]
        ),
    }

    verdict = "micro_closure_supported" if all(checks.values()) else "micro_closure_partial"

    out = {
        "assumptions": {
            "micro_action": "L=1/2 du^T K du - [1/2 u^T H u + lambda4/4 sum u_i^4]",
            "phi2_emergent_mode": "first non-trivial generalized mode",
            "best_v21_params": p,
        },
        "twisted_ring": tw,
        "ring": rg,
        "checks": checks,
        "verdict": verdict,
    }

    out_path = out_data / "phi1_phi2_micro_closure_summary.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("[INFO] Wrote:", out_path)


if __name__ == "__main__":
    main()
