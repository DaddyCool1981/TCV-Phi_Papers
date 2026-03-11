#!/usr/bin/env python
"""
Paper I / TCV-phi :
Test du spectre matière P(k) obtenu avec CLASS pour les paramètres
effectifs TCV-phi, et application d’un modèle de suppression fuzzy
pour le champ cohésif Φ2.

Ce script :
  - lit les paramètres effectifs (H0, Ω_b h^2, Ω_c h^2, A_s, n_s, …)
    via tcv_boltzmann.get_tcv_canonical_params(),
  - appelle CLASS pour obtenir P_cdm(k,z=0),
  - applique un modèle phenomenologique de fuzzy DM (k_J, n, f_coh),
  - produit :
      * une figure P(k) (CDM vs DM cohésive fuzzy),
      * une figure du transfert T^2(k),
      * un fichier ASCII avec k, P_cdm, P_eff, T^2.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# ---------------------------------------------------------------------
# 1. S'assurer que l'on peut importer tcv_boltzmann depuis la racine
# ---------------------------------------------------------------------
# Ce script est dans:  papers/paper-01/code/
# On remonte donc de 3 niveaux pour pointer vers la racine du repo TCV-PHI.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tcv_boltzmann import (  # type: ignore[attr-defined]
    get_tcv_canonical_params,
    get_pk_class,
    get_pk_fuzzy_from_cdm,
)


def main() -> None:
    print("=== TCV-phi : test P(k) CLASS + fuzzy Φ2 ===")

    # -----------------------------------------------------------------
    # 2. Paramètres TCV effectifs (Planck-like) pour CLASS
    # -----------------------------------------------------------------
    theta_tcv = get_tcv_canonical_params()
    print("Paramètres TCV effectifs :", theta_tcv)

    # -----------------------------------------------------------------
    # 3. Grille de k (en h/Mpc)
    #
    # NOTE : on se contente d'une plage sûre pour CLASS. Si CLASS
    # est initialisé avec un k_max ~ 7-8 h/Mpc, on évite k > 7.
    # -----------------------------------------------------------------
    k_min = 1e-4
    k_max_req = 7.0  # h/Mpc (à peu près la borne sûre pour CLASS dans ce setup)
    k_vals = np.logspace(np.log10(k_min), np.log10(k_max_req), 400)

    # -----------------------------------------------------------------
    # 4. P(k) CDM effectif via CLASS
    # -----------------------------------------------------------------
    print("Appel à CLASS pour P(k,z=0)...")
    k_vals_out, pk_cdm = get_pk_class(
        theta_tcv=theta_tcv,
        k_vals=k_vals,
        z=0.0,
    )

    # k_vals_out peut être légèrement différent de k_vals (interne CLASS).
    # On travaille désormais uniquement avec k_vals_out.

    # -----------------------------------------------------------------
    # 5. Modèle fuzzy cohésif pour Φ2
    #
    # Pour m2 ~ 10^-22 eV et f_coh ~ 1, la coupure se situe autour
    # de quelques h/Mpc. Ici on prend kJ ~ 5 h/Mpc et n=4 comme
    # ansatz phenomenologique (Paper I).
    # -----------------------------------------------------------------
    kJ = 5.0   # h/Mpc
    n = 4      # raideur de la coupure
    f_coh = 1.0  # 100% de la DM = Φ2 fuzzy, pour le benchmark

    pk_eff, pk_fuzzy, T2 = get_pk_fuzzy_from_cdm(
        k_vals_out,
        pk_cdm,
        kJ=kJ,
        n=n,
        f_coh=f_coh,
    )

    print(f"Paramètres fuzzy : kJ = {kJ:.2f} h/Mpc, n = {n}, f_coh = {f_coh}")

    # -----------------------------------------------------------------
    # 6. Répertoires de sortie pour PAPER I
    # -----------------------------------------------------------------
    figs_dir = REPO_ROOT / "papers" / "paper-01" / "figs"
    data_dir = REPO_ROOT / "papers" / "paper-01" / "data"
    figs_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # 7. Figure P(k)
    # -----------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    plt.loglog(k_vals_out, pk_cdm, label="Effective CDM (TCV-φ)")
    plt.loglog(k_vals_out, pk_eff, label="Cohesive fuzzy DM (Φ2, f_coh=1)")
    plt.loglog(k_vals_out, pk_fuzzy, "--", label="Fuzzy Φ2 only")
    plt.axvline(kJ, linestyle=":", label=r"$k_J$")

    plt.xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
    plt.ylabel(r"$P(k)$")
    plt.title("TCV-φ matter power spectrum (CLASS) + fuzzy Φ2 suppression")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    outfile_pk = figs_dir / "tcvphi_pk_fuzzy.png"
    plt.tight_layout()
    plt.savefig(outfile_pk, dpi=150)
    plt.close()
    print(f"[INFO] Figure P(k) enregistrée dans {outfile_pk}")

    # -----------------------------------------------------------------
    # 8. Figure du transfert T^2(k) = P_eff / P_cdm
    # -----------------------------------------------------------------
    # On utilise T2 renvoyé par get_pk_fuzzy_from_cdm, qui devrait être
    # exactement ce ratio (ou très proche).
    plt.figure(figsize=(8, 6))
    plt.loglog(k_vals_out, T2, label=r"$T^2(k) = P_{\rm eff}/P_{\rm cdm}$")
    plt.axvline(kJ, linestyle=":", label=r"$k_J$")

    plt.xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
    plt.ylabel(r"$T^2(k)$")
    plt.title("Cohesive fuzzy transfer $T^2(k)$ for Φ2")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    outfile_T2 = figs_dir / "tcvphi_transfer_T2.png"
    plt.tight_layout()
    plt.savefig(outfile_T2, dpi=150)
    plt.close()
    print(f"[INFO] Figure T^2(k) enregistrée dans {outfile_T2}")

    # -----------------------------------------------------------------
    # 9. Sauvegarde ASCII des données pour le papier
    # -----------------------------------------------------------------
    data = np.column_stack([k_vals_out, pk_cdm, pk_eff, T2])
    header = (
        "# TCV-phi : P(k) et transfert fuzzy pour PAPER I\n"
        "# colonnes : k[h/Mpc]   P_cdm(k)   P_eff(k)   T2(k)=P_eff/P_cdm\n"
        f"# Paramètres TCV : {theta_tcv}\n"
        f"# Fuzzy : kJ={kJ} h/Mpc, n={n}, f_coh={f_coh}\n"
    )
    outfile_dat = data_dir / "tcvphi_pk_fuzzy.dat"
    np.savetxt(outfile_dat, data, header=header)
    print(f"[INFO] Données P(k) enregistrées dans {outfile_dat}")

    # -----------------------------------------------------------------
    # 10. Petit résumé numérique
    # -----------------------------------------------------------------
    for kval in [0.1, 1.0, 5.0]:
        # interpolation log-log grossière
        pk_c = np.interp(
            np.log10(kval),
            np.log10(k_vals_out),
            np.log10(pk_cdm)
        )
        pk_e = np.interp(
            np.log10(kval),
            np.log10(k_vals_out),
            np.log10(pk_eff)
        )
        print(
            f"k = {kval:.2f} h/Mpc : "
            f"P_cdm ~ 10^{pk_c:.2f},  P_eff ~ 10^{pk_e:.2f}"
        )

    print("✅ test_tcv_pk terminé.")


if __name__ == "__main__":
    main()
