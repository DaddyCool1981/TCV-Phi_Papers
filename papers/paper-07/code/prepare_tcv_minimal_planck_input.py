#!/usr/bin/env python3
"""Prepare Cobaya input for the Paper-07 minimal TCV-vs-LCDM comparison run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_tcv_primordial_targets(path: Path) -> tuple[float, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    a_s = float(data["A_s_from_local_fit"])
    n_s = float(data["n_s_from_local_fit"])
    return a_s, n_s


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-lowl",
        action="store_true",
        help="Disable low-ell Planck likelihoods (kept for quick diagnostics only).",
    )
    args = parser.parse_args()

    out_data = REPO_ROOT / "papers" / "paper-07" / "data"
    out_data.mkdir(parents=True, exist_ok=True)

    prim_obs = REPO_ROOT / "papers" / "paper-03" / "data" / "primordial_observables.json"
    if not prim_obs.exists():
        raise FileNotFoundError(f"Missing primordial observables JSON: {prim_obs}")
    a_s_tcv, n_s_tcv = _load_tcv_primordial_targets(prim_obs)

    packages_path = REPO_ROOT / "papers" / "paper-07" / "external_packages"
    chain_tag = "tcv_minimal_planck_chain_v1_lowl"
    if args.no_lowl:
        chain_tag = "tcv_minimal_planck_chain_v1_nolowl"
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
            # Minimal TCV run: primordial amplitude/tilt fixed from Paper III.
            "A_s": {"value": a_s_tcv},
            "n_s": {"value": n_s_tcv},
            "h": {
                "prior": {"min": 0.55, "max": 0.85},
                "ref": {"dist": "norm", "loc": 0.674, "scale": 0.01},
                "proposal": 0.005,
            },
            "omega_b": {
                "prior": {"min": 0.020, "max": 0.025},
                "ref": {"dist": "norm", "loc": 0.0224, "scale": 0.0002},
                "proposal": 0.0001,
            },
            "omega_cdm": {
                "prior": {"min": 0.10, "max": 0.14},
                "ref": {"dist": "norm", "loc": 0.120, "scale": 0.002},
                "proposal": 0.001,
            },
            "tau_reio": {
                "prior": {"min": 0.01, "max": 0.12},
                "ref": {"dist": "norm", "loc": 0.054, "scale": 0.007},
                "proposal": 0.004,
            },
            "A_planck": {
                "prior": {"min": 0.9, "max": 1.1},
                "ref": {"dist": "norm", "loc": 1.0, "scale": 0.002},
                "proposal": 0.0005,
            },
        },
        "sampler": {
            "mcmc": {
                "burn_in": 0,
                "max_samples": 12000,
                "covmat": "auto",
                "learn_proposal": True,
                "Rminus1_stop": 0.05,
                "Rminus1_cl_stop": 0.2,
            }
        },
    }

    out_yaml = out_data / "tcv_minimal_planck_input.yaml"
    out_yaml.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    out_meta = out_data / "tcv_minimal_planck_input_meta.json"
    out_meta.write_text(
        json.dumps(
            {
                "input_yaml": str(out_yaml),
                "packages_path": str(packages_path),
                "output_root": str(chain_prefix),
                "lowl_enabled": not args.no_lowl,
                "tcv_primordial_source": str(prim_obs),
                "A_s_fixed": a_s_tcv,
                "n_s_fixed": n_s_tcv,
                "note": "Minimal TCV baseline with fixed primordial pair from Paper III.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("[INFO] Wrote:", out_yaml)
    print("[INFO] Wrote:", out_meta)


if __name__ == "__main__":
    main()

