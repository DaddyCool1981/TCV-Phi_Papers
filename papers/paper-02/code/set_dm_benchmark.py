# file: code/set_dm_benchmark.py

"""
Fixe la DM cohésive (Phi2) dans le benchmark TCV-phi en choisissant φ2_0
pour reproduire une Ω_dm h^2 cible à partir de m2 (ULDM).

Idée :
  - On prend le benchmark global (make_default_benchmark).
  - On lit m2 (en GeV) dans les micro-paramètres.
  - On choisit une valeur cible pour Ω_dm h^2 (par défaut 0.12, DM totale).
  - On utilise une formule standard de misalignment ULDM pour déduire φ2_0.
  - On calcule la Ω_phi h^2 correspondante et la fraction f_dm_coh = Ω_phi/0.12.
  - On met à jour le benchmark en mémoire (cosmo.phi2_0, cosmo.Omega_dm_h2, cosmo.f_dm_coh)
    et on imprime un résumé.

Ce script ne modifie pas les fichiers sources : il sert à documenter
numériquement le mapping {m2, Ω_dm h^2_target} -> φ2_0.
"""

from __future__ import annotations

import os
import sys
from math import sqrt
from typing import Tuple

# --- Rendre tcvphi importable quand on lance depuis code/ ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tcvphi.params import make_default_benchmark, TCVPhiBenchmark


def omega_phi_h2_from_misalignment(phi2_0: float, m2_GeV: float) -> float:
    """
    Approximation standard ULDM (ordre de grandeur) :

      Ω_φ h^2 ≃ 0.1 * (φ2_0 / 1e17 GeV)^2 * (m2 / 10^-22 eV)^(1/2)

    où :
      - m2_GeV est en GeV.
      - 10^-22 eV ≃ 10^-31 GeV.

    On l'utilise comme mapping simple entre {m2, φ2_0} et Ω_φ h^2.
    """
    m2_over_1e_22eV = m2_GeV / 1.0e-31
    return 0.1 * (phi2_0 / 1.0e17) ** 2 * (m2_over_1e_22eV ** 0.5)


def phi2_0_for_target_Omega(
    m2_GeV: float,
    Omega_target: float = 0.12,
) -> float:
    """
    Inversion de la formule ULDM pour obtenir φ2_0
    à partir d'une Ω_target (par défaut 0.12, DM totale).

    On résout :

      Ω_target = 0.1 * (φ2_0 / 1e17)^2 * (m2 / 10^-22 eV)^(1/2)

      => φ2_0 = 1e17 * sqrt( Ω_target / (0.1 * (m2/10^-22 eV)^(1/2)) )

    Tout en GeV.
    """
    m2_over_1e_22eV = m2_GeV / 1.0e-31
    denom = 0.1 * (m2_over_1e_22eV ** 0.5)
    if denom <= 0.0:
        raise ValueError("m2_GeV doit être positif pour la formule ULDM.")
    phi2_0 = 1.0e17 * sqrt(Omega_target / denom)
    return phi2_0


def build_dm_benchmark(
    Omega_target: float = 0.12,
    f_dm_total: float = 0.12,
) -> Tuple[TCVPhiBenchmark, float, float]:
    """
    Construit un benchmark mis à jour où Phi2 explique une fraction
    f_dm = Omega_target / f_dm_total de la DM.

      - Omega_target : Ω_phi h^2 visée pour le champ Phi2.
                       par défaut 0.12 (toute la DM).
      - f_dm_total   : Ω_dm h^2 totale observée (par défaut 0.12).

    Retourne (benchmark_modifié, phi2_0, f_dm_coh).
    """
    bench = make_default_benchmark()
    micro = bench.micro
    cosmo = bench.cosmo

    m2 = micro.m2  # en GeV

    phi2_0 = phi2_0_for_target_Omega(m2_GeV=m2, Omega_target=Omega_target)
    Omega_phi_h2 = omega_phi_h2_from_misalignment(phi2_0, m2_GeV=m2)
    f_dm_coh = Omega_phi_h2 / f_dm_total if f_dm_total > 0.0 else 0.0

    # On met à jour les champs dérivés du benchmark (en mémoire uniquement)
    cosmo.phi2_0 = phi2_0
    cosmo.Omega_dm_h2 = Omega_phi_h2
    cosmo.f_dm_coh = f_dm_coh

    return bench, phi2_0, f_dm_coh


def main() -> None:
    # Cibles par défaut : Phi2 = 100% de la DM (Ω_dm h^2 ~ 0.12).
    Omega_dm_total = 0.12
    Omega_target = 0.12  # on peut modifier ici si on veut < 100% DM

    bench, phi2_0, f_dm_coh = build_dm_benchmark(
        Omega_target=Omega_target,
        f_dm_total=Omega_dm_total,
    )

    micro = bench.micro
    cosmo = bench.cosmo

    print("=== TCV-phi : réglage benchmark DM cohésive (Phi2) ===\n")
    print(f"Benchmark de départ : {bench.name}")
    print("\n--- Paramètres micro pertinents ---")
    print(f"  m2 (Phi2, ULDM)          = {micro.m2:.3e} GeV")
    print("  (≈ 10^-22 eV si m2 ≈ 1e-31 GeV)")
    print()
    print("--- Cible DM ---")
    print(f"  Ω_dm,total h^2 (observé) ≈ {Omega_dm_total:.3f}")
    print(f"  Ω_phi,target h^2         = {Omega_target:.3f}")
    print(f"  => fraction de DM visée  = Ω_phi/Ω_dm,total ≈ {Omega_target / Omega_dm_total:.3f}")
    print()

    print("--- Résultat du mapping ULDM ---")
    print(f"  φ2_0 requis       ≈ {phi2_0:.3e} GeV")
    print(f"  Ω_phi h^2 obtenu  ≈ {cosmo.Omega_dm_h2:.3e}")
    print(f"  fraction f_dm_coh ≈ {cosmo.f_dm_coh:.3f}")
    print()
    print("Interprétation :")
    print("  - Si f_dm_coh ≈ 1, Phi2 peut en principe expliquer ~100% de la DM.")
    print("  - Si f_dm_coh ≪ 1, ce choix m2 + φ2_0 ne fournit qu'une fraction")
    print("    de la DM, et il faudrait soit augmenter φ2_0, soit ajouter")
    print("    d'autres composantes de matière noire.")
    print()
    print("Mise à jour interne du benchmark :")
    print(f"  cosmo.phi2_0      = {cosmo.phi2_0:.3e} GeV")
    print(f"  cosmo.Omega_dm_h2 = {cosmo.Omega_dm_h2:.3e}")
    print(f"  cosmo.f_dm_coh    = {cosmo.f_dm_coh:.3f}")
    print()
    print("⚠️  Cette mise à jour n'est faite qu'en mémoire (aucun fichier n'est modifié).")
    print("    Pour figer ce benchmark, reporte φ2_0 et Ω_dm h^2 dans tcvphi/params.py")
    print("    ou dans un fichier de configuration dédié.")
    print()
    print("✅ Script set_dm_benchmark.py terminé.")


if __name__ == "__main__":
    main()