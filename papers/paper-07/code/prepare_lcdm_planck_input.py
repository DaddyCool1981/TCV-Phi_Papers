#!/usr/bin/env python3
"""Prepare a Cobaya LCDM + Planck input file for Paper 07."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-lowl",
        action="store_true",
        help="Disable low-ell Planck likelihoods (not recommended for final LCDM baseline).",
    )
    args = parser.parse_args()

    out_data = REPO_ROOT / "papers" / "paper-07" / "data"
    out_data.mkdir(parents=True, exist_ok=True)

    packages_path = REPO_ROOT / "papers" / "paper-07" / "external_packages"
    # Use a new output root for low-ell-enabled baseline runs.
    chain_tag = "lcdm_planck_chain_classnative_v7_lowl"
    if args.no_lowl:
        chain_tag = "lcdm_planck_chain_classnative_v7_nolowl"
    chain_prefix = out_data / chain_tag

    likelihood = {
        "planck_2018_highl_plik.TTTEEE_lite_native": None,
        "planck_2018_lensing.native": None,
    }
    if not args.no_lowl:
        likelihood["planck_2018_lowl.TT"] = None
        likelihood["planck_2018_lowl.EE"] = None

    config = {
        "packages_path": str(packages_path),
        "output": str(chain_prefix),
        "debug": False,
        "theory": {
            "classy": {
                "extra_args": {
                    "N_ur": 3.046,
                    "YHe": 0.245,
                    "non_linear": "halofit",
                    "l_max_scalars": 3000,
                }
            }
        },
        "likelihood": likelihood,
        "params": {
            "logA": {
                "prior": {"min": 2.6, "max": 3.6},
                "ref": {"dist": "norm", "loc": 3.044, "scale": 0.01},
                "proposal": 0.005,
                "drop": True,
            },
            "A_s": {"value": "lambda logA: 1e-10*np.exp(logA)", "derived": False},
            "n_s": {"prior": {"min": 0.9, "max": 1.1}, "ref": {"dist": "norm", "loc": 0.965, "scale": 0.004}, "proposal": 0.002},
            # CLASS physical densities (omega_i = Omega_i h^2).
            "h": {"prior": {"min": 0.55, "max": 0.85}, "ref": {"dist": "norm", "loc": 0.674, "scale": 0.01}, "proposal": 0.005},
            "omega_b": {"prior": {"min": 0.020, "max": 0.025}, "ref": {"dist": "norm", "loc": 0.0224, "scale": 0.0002}, "proposal": 0.0001},
            "omega_cdm": {"prior": {"min": 0.10, "max": 0.14}, "ref": {"dist": "norm", "loc": 0.120, "scale": 0.002}, "proposal": 0.001},
            "tau_reio": {"prior": {"min": 0.01, "max": 0.12}, "ref": {"dist": "norm", "loc": 0.054, "scale": 0.007}, "proposal": 0.004},
            # Shared Planck calibration nuisance used by plik+lensing native likelihoods.
            "A_planck": {"prior": {"min": 0.9, "max": 1.1}, "ref": {"dist": "norm", "loc": 1.0, "scale": 0.002}, "proposal": 0.0005},
        },
        "sampler": {
            "mcmc": {
                "burn_in": 0,
                "max_samples": 1500,
                "covmat": "auto",
                "learn_proposal": True,
                "Rminus1_stop": 0.05,
                "Rminus1_cl_stop": 0.2,
            }
        },
    }

    out_yaml = out_data / "lcdm_planck_input.yaml"
    with out_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    out_json = out_data / "lcdm_planck_input_meta.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "input_yaml": str(out_yaml),
                "packages_path": str(packages_path),
                "lowl_enabled": not args.no_lowl,
                "output_root": str(chain_prefix),
                "note": "Prepared for Cobaya LCDM baseline with Planck native likelihoods.",
            },
            f,
            indent=2,
        )

    print("[INFO] Wrote:", out_yaml)
    print("[INFO] Wrote:", out_json)


if __name__ == "__main__":
    main()
