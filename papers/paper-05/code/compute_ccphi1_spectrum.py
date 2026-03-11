#!/usr/bin/env python3
"""Compute baseline CC-Phi1 cluster spectrum for Paper-05."""

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

from tcvphi.ccphi1_micro import CCPhi1Params, coherence_length, mode_energies, mode_frequencies, params_to_dict


def main() -> None:
    out_data = REPO_ROOT / "papers" / "paper-05" / "data"
    out_figs = REPO_ROOT / "papers" / "paper-05" / "figs"
    out_data.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    params = CCPhi1Params(m1=1.0e-8, xi0=0.1, n_modes=3)
    R0 = 0.0
    Xi = mode_energies(R0, params)
    w = mode_frequencies(R0, params)
    gap12 = float(Xi[1] - Xi[0]) if Xi.size >= 2 else None

    summary = {
        "params": params_to_dict(params),
        "R_baseline": R0,
        "coherence_length": coherence_length(R0, params),
        "Xi_modes": [float(x) for x in Xi],
        "E_modes": [float(x) for x in Xi],  # NOTE: compatibility alias for existing readers.
        "omega_modes": [float(x) for x in w],
        "gap_Xi2_minus_Xi1": gap12,
        "gap_E2_minus_E1": gap12,  # NOTE: compatibility alias for existing readers.
        "status": "phenomenological cluster quantization toy aligned with preprint 9",
    }
    out_json = out_data / "ccphi1_spectrum.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("[INFO] Wrote:", out_json)

    n = np.arange(1, Xi.size + 1)
    plt.figure(figsize=(6.5, 4.2))
    plt.plot(n, Xi, "o-", lw=1.6, label=r"$\Xi_n=\omega_n^2$")
    plt.xlabel("mode index n")
    plt.ylabel(r"$\Xi_n$ (Planck units)")
    plt.title("Paper-05 baseline CC-Phi1 mode eigenvalues")
    plt.grid(True, ls=":")
    plt.legend(loc="best")
    plt.tight_layout()
    out_fig = out_figs / "ccphi1_mode_spectrum.png"
    plt.savefig(out_fig, dpi=180, bbox_inches="tight")
    plt.close()
    print("[INFO] Wrote:", out_fig)


if __name__ == "__main__":
    main()
