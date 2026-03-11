#!/usr/bin/env python3
"""
Paper II/III primordial pipeline:
  1) compute P_R(k) from cohesive bounce effective model
  2) write table PR_tcvphi.dat
  3) compare CLASS P(k) using
     - powerlaw primordial mode
     - external primordial file mode (effective As, ns fit from file)
"""

from __future__ import annotations

import json
import sys
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
CORE_LIB = REPO_ROOT / "core" / "lib"
if str(CORE_LIB) not in sys.path:
    sys.path.insert(0, str(CORE_LIB))

from tcvphi.primordial import (
    write_primordial_table,
)
from tcvphi.scalar_spectrum import (
    TCVBounceParams,
    CohesiveMixtureParams,
    compute_tcv_bounce_spectrum,
    compute_cohesive_mixture_spectrum,
)
from tcvphi.tcv_boltzmann import (
    get_tcv_canonical_params,
    infer_powerlaw_from_primordial_file,
    compute_pk_class,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute primordial spectrum and CLASS bridge outputs.")
    parser.add_argument(
        "--model",
        choices=["cohesive_mix", "tcv_bounce"],
        default="cohesive_mix",
        help="Primordial model used to generate P_R(k).",
    )
    parser.add_argument(
        "--target-As",
        type=float,
        default=2.1e-9,
        help="Target amplitude at pivot k=0.05 used for normalization.",
    )
    args = parser.parse_args()

    out_figs = REPO_ROOT / "papers" / "paper-02" / "figs"
    out_data = REPO_ROOT / "papers" / "paper-02" / "data"
    out_code_data = REPO_ROOT / "papers" / "paper-02" / "code" / "data"
    out_figs.mkdir(parents=True, exist_ok=True)
    out_data.mkdir(parents=True, exist_ok=True)
    out_code_data.mkdir(parents=True, exist_ok=True)

    if args.model == "tcv_bounce":
        params = TCVBounceParams(
            a_b=1.0e-10,
            eta0=1.0,
            M_pl=1.0,
            Phi0=1.0,
            dPhi1=8.0e-6,
            xi=1.0,
            eta_i=-10.0,
            eta_f=+10.0,
            n_eta=4500,
            k_min=1.0e-3,
            k_max=1.0e-1,
            n_k=50,
        )
        k, p_r, n_s = compute_tcv_bounce_spectrum(params)
    else:
        params = CohesiveMixtureParams(
            w_eff=-0.003,
            eta_i=-1000.0,
            eta_f=-1.0,
            n_eta=20000,
            k_min=1.0e-3,
            k_max=1.0e-1,
            n_k=50,
            dphi_rel=1.0e-5,
            C0=-0.0038,
            eta_c=50.0,
            mode="lorentzian",
        )
        k, p_r, n_s = compute_cohesive_mixture_spectrum(params)

    # Normalize amplitude to target A_s at pivot to keep CLASS bridge realistic.
    k_pivot = 0.05
    p_pivot = float(np.interp(np.log(k_pivot), np.log(k), np.log(p_r)))
    A_raw = float(np.exp(p_pivot))
    scale = args.target_As / max(A_raw, 1.0e-300)
    p_r = p_r * scale

    file_code = write_primordial_table(
        out_code_data / "PR_tcvphi.dat",
        k,
        p_r,
        metadata={"model": args.model, "n_s_fit": f"{n_s:.6f}", "A_s_target": args.target_As},
    )
    file_main = write_primordial_table(
        out_data / "PR_tcvphi.dat",
        k,
        p_r,
        metadata={"model": args.model, "n_s_fit": f"{n_s:.6f}", "A_s_target": args.target_As},
    )

    As_eff, ns_eff = infer_powerlaw_from_primordial_file(file_main)

    theta = get_tcv_canonical_params()
    k_pk = np.logspace(-3, 0.7, 320)
    pk_powerlaw = compute_pk_class(theta_tcv=theta, k_vals=k_pk, primordial_mode="powerlaw", z=0.0)
    pk_external = compute_pk_class(
        theta_tcv=theta,
        k_vals=k_pk,
        primordial_mode="external_file",
        primordial_file=file_main,
        z=0.0,
    )
    ratio = pk_external["pk"] / np.maximum(pk_powerlaw["pk"], 1.0e-300)

    np.savez(
        out_data / "primordial_pk_comparison.npz",
        k=pk_powerlaw["k"],
        pk_powerlaw=pk_powerlaw["pk"],
        pk_external=pk_external["pk"],
        ratio=ratio,
        k_primordial=k,
        P_R=p_r,
    )

    summary = {
        "n_s_from_ms": float(n_s),
        "A_s_from_file_fit": float(As_eff),
        "n_s_from_file_fit": float(ns_eff),
        "model_used": args.model,
        "A_s_target": float(args.target_As),
        "A_s_raw_before_rescale": float(A_raw),
        "powerlaw_used_fallback": bool(pk_powerlaw["used_fallback"][0]),
        "external_used_fallback": bool(pk_external["used_fallback"][0]),
        "powerlaw_error": str(pk_powerlaw["error"][0]),
        "external_error": str(pk_external["error"][0]),
        "primordial_file_code_path": str(file_code),
        "primordial_file_data_path": str(file_main),
        "pk_ratio_min": float(np.min(ratio)),
        "pk_ratio_max": float(np.max(ratio)),
        "note": "external_file mode currently maps table -> effective (A_s, n_s) for CLASS.",
    }
    with (out_data / "primordial_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Figure 1: primordial spectrum
    plt.figure(figsize=(8, 6))
    plt.loglog(k, p_r, "o-", ms=3.0, lw=1.2, label=r"$P_{\mathcal{R}}(k)$")
    plt.xlabel(r"$k$")
    plt.ylabel(r"$P_{\mathcal{R}}(k)$")
    plt.title("Cohesive-bounce primordial spectrum")
    plt.grid(True, which="both", ls=":")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_figs / "primordial_spectrum.png", dpi=180, bbox_inches="tight")
    plt.close()

    # Figure 2: matter spectrum comparison
    plt.figure(figsize=(8, 6))
    plt.loglog(pk_powerlaw["k"], pk_powerlaw["pk"], lw=1.8, label="CLASS powerlaw mode")
    plt.loglog(pk_external["k"], pk_external["pk"], lw=1.8, label="CLASS external-file mode")
    plt.xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
    plt.ylabel(r"$P(k)$")
    plt.title("Power-law vs external-primordial mode")
    plt.grid(True, which="both", ls=":")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_figs / "primordial_pk_comparison.png", dpi=180, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 4.8))
    plt.semilogx(pk_powerlaw["k"], ratio, lw=1.8)
    plt.axhline(1.0, ls="--", color="k", alpha=0.6)
    plt.xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
    plt.ylabel(r"$P_{\rm external}/P_{\rm powerlaw}$")
    plt.title("External primordial mode ratio")
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.savefig(out_figs / "primordial_pk_ratio.png", dpi=180, bbox_inches="tight")
    plt.close()

    print("[INFO] Wrote:", file_code)
    print("[INFO] Wrote:", file_main)
    print("[INFO] Wrote:", out_data / "primordial_pk_comparison.npz")
    print("[INFO] Wrote:", out_data / "primordial_summary.json")
    print("[INFO] Wrote:", out_figs / "primordial_spectrum.png")
    print("[INFO] Wrote:", out_figs / "primordial_pk_comparison.png")
    print("[INFO] Wrote:", out_figs / "primordial_pk_ratio.png")
    print("[INFO] model =", args.model)
    print("[INFO] n_s (MS) =", f"{n_s:.6f}")
    print("[INFO] n_s (file fit) =", f"{ns_eff:.6f}")


if __name__ == "__main__":
    main()
