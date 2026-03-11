# papers/paper-01/code/test_tcv_background.py

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# NOTE: ensure repo root is on PYTHONPATH when running from repo root.
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT))

from tcvphi.cosmology import (
    get_tcv_canonical_params,
    get_tcv_background,
)


def main():
    # 1) Récupérer les paramètres cosmologiques TCV “effectifs”
    theta = get_tcv_canonical_params()
    print("Paramètres TCV effectifs :", theta)

    # 2) Construire le background H(a) type ΛCDM-like
    bg = get_tcv_background(theta=theta)
    a = bg["a"]          # facteur d'échelle (0 < a ≤ 1)
    H = bg["H"]          # H(a) en unités physiques
    E2 = bg["E2"]        # E(a)^2 = H(a)^2 / H0^2
    Omegas0 = bg["Omegas0"]

    print("\nOmegas0 (aujourd’hui, a=1) :")
    for k, v in Omegas0.items():
        print(f"  Ω_{k} = {v:.5f}")

    # === 2bis) Construire un LCDM "pur" à partir des mêmes Ω0 pour comparer ===
    # On reconstruit E_LCDM^2(a) = Ω_m a^-3 + Ω_r a^-4 + Ω_Λ + Ω_k a^-2
    # en prenant Ω_m = Ω_b + Ω_c si besoin.
    Omega_b = Omegas0.get("b", 0.0)
    Omega_c = Omegas0.get("c", 0.0)
    Omega_m = Omegas0.get("m", Omega_b + Omega_c)
    Omega_r = Omegas0.get("r", 0.0)
    Omega_L = Omegas0.get("L", Omegas0.get("Λ", 0.0))

    if "k" in Omegas0:
        Omega_k = Omegas0["k"]
    else:
        Omega_k = 1.0 - (Omega_m + Omega_r + Omega_L)

    E2_lcdm = (
        Omega_m * a**-3
        + Omega_r * a**-4
        + Omega_L
        + Omega_k * a**-2
    )

    # Ratio H_TCV / H_LCDM = sqrt(E2_TCV / E2_LCDM)
    ratio = np.sqrt(E2 / E2_lcdm)

    print("\nDiagnostic H_TCV / H_LCDM :")
    print(f"  min(H_TCV/H_LCDM) = {ratio.min():.5e}")
    print(f"  max(H_TCV/H_LCDM) = {ratio.max():.5e}")

    # 3) Plot H(a)/H0 vs a (comme avant)
    outdir = Path(__file__).resolve().parents[1] / "figs"
    outdir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.loglog(a, np.sqrt(E2))
    plt.xlabel("a")
    plt.ylabel("H(a)/H0")
    plt.title("Effective TCV-ϕ background (ΛCDM-like)")
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    outfile1 = outdir / "tcv_background_Ha.png"
    plt.savefig(outfile1, dpi=150, bbox_inches="tight")
    print(f"[INFO] Figure background H(a)/H0 enregistrée dans {outfile1}")
    plt.close()

    # 4) Nouveau plot : H_TCV / H_LCDM en fonction de 1/a = 1+z
    x = 1.0 / a  # 1+z
    plt.figure()
    plt.semilogx(x, ratio, label=r"$H_{\rm TCV}/H_{\Lambda{\rm CDM}}$")
    plt.axhline(1.0, ls="--", alpha=0.5, label="ΛCDM")
    plt.xlabel(r"$1/a = 1+z$")
    plt.ylabel(r"$H_{\rm TCV} / H_{\Lambda{\rm CDM}}$")
    plt.title("Relative deviation of expansion rate")
    plt.grid(True, which="both", ls=":")
    plt.legend()
    plt.tight_layout()
    outfile2 = outdir / "tcv_background_H_ratio.png"
    plt.savefig(outfile2, dpi=150, bbox_inches="tight")
    print(f"[INFO] Figure ratio H_TCV/H_LCDM enregistrée dans {outfile2}")
    plt.close()


if __name__ == "__main__":
    main()
