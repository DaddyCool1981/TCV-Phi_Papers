#!/usr/bin/env python3
"""Compute Mukhanov-Sasaki primordial spectrum products for Paper 03."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
CORE_LIB = REPO_ROOT / "core" / "lib"
if str(CORE_LIB) not in sys.path:
    sys.path.insert(0, str(CORE_LIB))

from tcvphi.primordial import write_primordial_table
from tcvphi.scalar_spectrum import (
    CohesiveMixtureParams,
    TCVBounceParams,
    compute_cohesive_mixture_spectrum,
    compute_tcv_bounce_spectrum,
)
from tcvphi.tcv_boltzmann import get_tcv_canonical_params


def _normalize_to_As(k: np.ndarray, p_r: np.ndarray, target_As: float, k_pivot: float) -> tuple[np.ndarray, float]:
    logk = np.log(k)
    logp = np.log(np.maximum(p_r, 1.0e-300))
    p_raw = float(np.exp(np.interp(np.log(k_pivot), logk, logp)))
    scale = target_As / max(p_raw, 1.0e-300)
    return p_r * scale, p_raw


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Paper-03 primordial MS spectrum.")
    parser.add_argument("--model", choices=["cohesive_mix", "tcv_bounce"], default="cohesive_mix")
    parser.add_argument("--target-As", type=float, default=2.1e-9)
    parser.add_argument("--k-pivot", type=float, default=0.05)
    args = parser.parse_args()

    out_figs = REPO_ROOT / "papers" / "paper-03" / "figs"
    out_data = REPO_ROOT / "papers" / "paper-03" / "data"
    out_code_data = REPO_ROOT / "papers" / "paper-03" / "code" / "data"
    out_figs.mkdir(parents=True, exist_ok=True)
    out_data.mkdir(parents=True, exist_ok=True)
    out_code_data.mkdir(parents=True, exist_ok=True)

    model_params: dict[str, float | str]
    if args.model == "tcv_bounce":
        params = TCVBounceParams(
            a_b=1.0e-10,
            eta0=1.0,
            M_pl=1.0,
            Phi0=1.0,
            dPhi1=8.0e-6,
            xi=1.0,
            eta_i=-10.0,
            eta_f=10.0,
            n_eta=4500,
            k_min=1.0e-3,
            k_max=1.0e-1,
            n_k=60,
        )
        k, p_r, n_s = compute_tcv_bounce_spectrum(params)
        model_params = {
            "a_b": params.a_b,
            "eta0": params.eta0,
            "M_pl": params.M_pl,
            "Phi0": params.Phi0,
            "dPhi1": params.dPhi1,
            "xi": params.xi,
            "eta_i": params.eta_i,
            "eta_f": params.eta_f,
            "n_eta": params.n_eta,
            "k_min": params.k_min,
            "k_max": params.k_max,
            "n_k": params.n_k,
        }
    else:
        params = CohesiveMixtureParams(
            w_eff=-0.003,
            eta_i=-1000.0,
            eta_f=-1.0,
            n_eta=20000,
            k_min=1.0e-3,
            k_max=1.0e-1,
            n_k=60,
            dphi_rel=1.0e-5,
            C0=-0.0038,
            eta_c=50.0,
            mode="lorentzian",
        )
        k, p_r, n_s = compute_cohesive_mixture_spectrum(params)
        model_params = {
            "w_eff": params.w_eff,
            "eta_i": params.eta_i,
            "eta_f": params.eta_f,
            "n_eta": params.n_eta,
            "k_min": params.k_min,
            "k_max": params.k_max,
            "n_k": params.n_k,
            "dphi_rel": params.dphi_rel,
            "C0": params.C0,
            "eta_c": params.eta_c,
            "mode": params.mode,
        }

    p_r, A_raw = _normalize_to_As(k, p_r, target_As=args.target_As, k_pivot=args.k_pivot)

    file_code = write_primordial_table(
        out_code_data / "PR_tcvphi.dat",
        k,
        p_r,
        metadata={"model": args.model, "n_s_ms": f"{n_s:.6f}", "A_s_target": args.target_As},
    )
    file_main = write_primordial_table(
        out_data / "PR_tcvphi.dat",
        k,
        p_r,
        metadata={"model": args.model, "n_s_ms": f"{n_s:.6f}", "A_s_target": args.target_As},
    )

    np.savez(
        out_data / "ms_spectrum.npz",
        k=k,
        P_R=p_r,
    )

    summary = {
        "model_used": args.model,
        "model_parameters": model_params,
        "n_s_from_ms": float(n_s),
        "A_s_target": float(args.target_As),
        "A_s_raw_before_rescale": float(A_raw),
        "k_pivot": float(args.k_pivot),
        "canonical_class_camb_params": get_tcv_canonical_params(),
        "primordial_file_code_path": str(file_code),
        "primordial_file_data_path": str(file_main),
        "canonical_reference": "paper-01 canonical cosmology and notation",
    }
    with (out_data / "ms_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plt.figure(figsize=(8, 6))
    plt.loglog(k, p_r, "o-", ms=2.8, lw=1.2)
    plt.xlabel(r"$k$")
    plt.ylabel(r"$\mathcal{P}_{\mathcal{R}}(k)$")
    plt.title(f"Paper-03 primordial MS spectrum ({args.model})")
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.savefig(out_figs / "primordial_ms_spectrum.png", dpi=180, bbox_inches="tight")
    plt.close()

    print("[INFO] Wrote:", file_code)
    print("[INFO] Wrote:", file_main)
    print("[INFO] Wrote:", out_data / "ms_spectrum.npz")
    print("[INFO] Wrote:", out_data / "ms_summary.json")
    print("[INFO] Wrote:", out_figs / "primordial_ms_spectrum.png")
    print("[INFO] n_s (MS) =", f"{n_s:.6f}")


if __name__ == "__main__":
    main()
