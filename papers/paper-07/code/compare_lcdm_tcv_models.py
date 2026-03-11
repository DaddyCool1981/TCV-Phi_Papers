#!/usr/bin/env python3
"""Summarize and compare Paper-07 LCDM vs minimal-TCV chains."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from getdist import loadMCSamples

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_chain_stats(root: Path, ignore_rows: float) -> dict:
    samples = loadMCSamples(str(root), no_cache=True, settings={"ignore_rows": ignore_rows})
    stats = samples.getMargeStats()

    def _interval(name: str) -> dict:
        par = stats.parWithName(name)
        lim = par.limits[0]
        return {"mean": float(par.mean), "lower_68": float(lim.lower), "upper_68": float(lim.upper)}

    # Parse columns directly from chain header to avoid index mismatches.
    with open(f"{root}.1.txt", "r", encoding="utf-8") as f:
        header = f.readline().lstrip("#").strip().split()
    if "chi2__CMB" not in header:
        raise RuntimeError(f"chi2__CMB not found in chain header for root {root}")

    chain_data = np.atleast_2d(np.loadtxt(f"{root}.1.txt"))
    idx_chi2 = header.index("chi2__CMB")
    idx_mlp = header.index("minuslogpost") if "minuslogpost" in header else None
    best_idx = int(np.argmin(chain_data[:, idx_chi2]))

    out = {
        "root": str(root),
        "rows": int(chain_data.shape[0]),
        "omega_b": _interval("omega_b"),
        "omega_cdm": _interval("omega_cdm"),
        "h": _interval("h"),
        "tau_reio": _interval("tau_reio"),
        "A_planck": _interval("A_planck"),
        "bestfit": {"chi2__CMB": float(chain_data[best_idx, idx_chi2])},
    }
    if "n_s" in header:
        out["n_s"] = _interval("n_s")
    if "logA" in header:
        out["logA"] = _interval("logA")
    if idx_mlp is not None:
        out["bestfit"]["minuslogpost"] = float(chain_data[best_idx, idx_mlp])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lcdm-root", default="lcdm_planck_chain_classnative_v7_lowl")
    parser.add_argument("--tcv-root", default="tcv_minimal_planck_chain_v1_lowl")
    parser.add_argument("--ignore-rows", type=float, default=0.3)
    parser.add_argument(
        "--n-data",
        type=int,
        default=2500,
        help="Proxy number of effective data points for BIC (report as proxy in paper).",
    )
    args = parser.parse_args()

    data_dir = REPO_ROOT / "papers" / "paper-07" / "data"
    lcdm = _load_chain_stats(data_dir / args.lcdm_root, args.ignore_rows)
    tcv = _load_chain_stats(data_dir / args.tcv_root, args.ignore_rows)

    # k counts sampled cosmological+calibration parameters:
    # LCDM: (logA, n_s, h, omega_b, omega_cdm, tau_reio, A_planck) => 7
    # TCV minimal: (h, omega_b, omega_cdm, tau_reio, A_planck) => 5
    k_lcdm = 7
    k_tcv = 5
    chi2_lcdm = lcdm["bestfit"]["chi2__CMB"]
    chi2_tcv = tcv["bestfit"]["chi2__CMB"]

    aic_lcdm = chi2_lcdm + 2 * k_lcdm
    aic_tcv = chi2_tcv + 2 * k_tcv
    bic_lcdm = chi2_lcdm + k_lcdm * np.log(args.n_data)
    bic_tcv = chi2_tcv + k_tcv * np.log(args.n_data)

    comparison = {
        "lcdm": lcdm,
        "tcv_minimal": tcv,
        "criteria": {
            "n_data_proxy": int(args.n_data),
            "delta_chi2_tcv_minus_lcdm": float(chi2_tcv - chi2_lcdm),
            "delta_AIC_tcv_minus_lcdm": float(aic_tcv - aic_lcdm),
            "delta_BIC_tcv_minus_lcdm": float(bic_tcv - bic_lcdm),
            "note": "AIC/BIC are reported with an explicit n_data proxy.",
        },
    }

    out = data_dir / "model_comparison_lcdm_vs_tcv_minimal.json"
    out.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print("[INFO] Wrote:", out)


if __name__ == "__main__":
    main()
