#!/usr/bin/env python3
"""Exploratory CC-Phi2 flavour toy summary for Paper-05."""

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

from tcvphi.phi2_flavour_toy import (  # noqa: E402
    QuantumClusterPhi2,
    compute_mass_matrix_quantum,
    extract_mixing_angles,
)


def main() -> None:
    out_data = REPO_ROOT / "papers" / "paper-05" / "data"
    out_figs = REPO_ROOT / "papers" / "paper-05" / "figs"
    out_data.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    cluster = QuantumClusterPhi2(m2=0.01, Lc=50.0, N_modes=5)
    m_diag_lep = np.array([5.11e-4, 1.057e-1, 1.777])
    m_diag_nu = np.array([1.0e-3, 8.6e-3, 5.0e-2])

    M_lep = compute_mass_matrix_quantum("lepton", m_diag_lep, cluster, epsilon_f=0.007)
    M_nu = compute_mass_matrix_quantum("neutrino", m_diag_nu, cluster, epsilon_f=0.20, kappa_nu=1.5)

    evals_lep, U_lep = np.linalg.eigh(M_lep)
    evals_nu, U_nu = np.linalg.eigh(M_nu)

    idx_l = np.argsort(evals_lep.real)
    idx_n = np.argsort(evals_nu.real)
    m_phys_lep = evals_lep.real[idx_l]
    m_phys_nu = evals_nu.real[idx_n]
    U_lep = U_lep[:, idx_l]
    U_nu = U_nu[:, idx_n]
    U_pmns = U_lep.conjugate().T @ U_nu

    t12, t13, t23, dcp = extract_mixing_angles(U_pmns)
    rad2deg = 180.0 / np.pi

    summary = {
        "cluster": {"m2": cluster.m2, "Lc": cluster.Lc, "N_modes": cluster.N_modes},
        "masses_input": {"lepton": m_diag_lep.tolist(), "neutrino": m_diag_nu.tolist()},
        "masses_output": {"lepton": m_phys_lep.tolist(), "neutrino": m_phys_nu.tolist()},
        "pmns_angles_deg": {
            "theta12": float(t12 * rad2deg),
            "theta13": float(t13 * rad2deg),
            "theta23": float(t23 * rad2deg),
            "delta_cp": float(dcp * rad2deg),
        },
        "status": "exploratory toy model implemented in-repo for reproducibility",
    }
    out_json = out_data / "ccphi2_flavour_toy.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("[INFO] Wrote:", out_json)

    labels = [r"$\theta_{12}$", r"$\theta_{13}$", r"$\theta_{23}$"]
    vals = [t12 * rad2deg, t13 * rad2deg, t23 * rad2deg]
    plt.figure(figsize=(6.2, 4.0))
    plt.bar(labels, vals, color=["C0", "C1", "C2"])
    plt.ylabel("angle (deg)")
    plt.title("CC-Phi2 flavour toy: PMNS angles")
    plt.grid(axis="y", ls=":", alpha=0.5)
    plt.tight_layout()
    out_fig = out_figs / "ccphi2_flavour_toy_angles.png"
    plt.savefig(out_fig, dpi=180, bbox_inches="tight")
    plt.close()
    print("[INFO] Wrote:", out_fig)


if __name__ == "__main__":
    main()
