"""
test_potential_properties.py

Petits tests de cohérence pour le potentiel cohésif TCV-phi.

- Vérifie que le minimum global est bien autour de (phi1 = phi0, phi2 = 0)
- Vérifie que V_tot(min) ~ 0
- Vérifie que les dérivées premières au minimum sont ~ 0
- Estime numériquement la masse effective de Phi1 au minimum et la compare à m1

Ce n'est pas une "preuve" mathématique, mais un sanity check numérique
pour aider à fixer/contrôler les paramètres de l'EFT.
"""

from __future__ import annotations

import math
import os
import sys

# Ajoute le répertoire parent (TCV-PHI) au PYTHONPATH quand on lance depuis code/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tcvphi.params import make_default_benchmark
from tcvphi.potential import CohesivePotential


def safe_call_dV_dphi1(pot: CohesivePotential, phi1: float, phi2: float) -> float:
    """
    Dérivée de V_tot par rapport à phi1.

    Ici, on suppose que la seule dépendance en phi1 vient de V1(phi1),
    comme dans CohesivePotential.dV1_dphi1.
    Le second argument phi2 est ignoré.
    """
    if hasattr(pot, "dV1_dphi1"):
        return pot.dV1_dphi1(phi1)  # type: ignore[attr-defined]
    raise AttributeError("CohesivePotential n'a pas de méthode dV1_dphi1")


def safe_call_dV_dphi2(pot: CohesivePotential, phi1: float, phi2: float) -> float:
    """
    Dérivée de V_tot par rapport à phi2.

    Ici, on suppose que la seule dépendance en phi2 vient de V2(phi2),
    comme dans CohesivePotential.dV2_dphi2.
    Le premier argument phi1 est ignoré.
    """
    if hasattr(pot, "dV2_dphi2"):
        return pot.dV2_dphi2(phi2)  # type: ignore[attr-defined]
    raise AttributeError("CohesivePotential n'a pas de méthode dV2_dphi2")

def main() -> None:
    # --- 1. Construction des paramètres et du potentiel ---
    benchmark = make_default_benchmark()
    micro = benchmark.micro  # <- c'est lui qui contient m1, phi0, etc.
    pot = CohesivePotential(micro)

    # On essaie de récupérer la valeur attendue du minimum de Phi1
    phi1_min_guess = 0.0
    for name in ("phi0", "Phi0", "phi1_min", "phi1_vev"):
        if hasattr(micro, name):
            phi1_min_guess = getattr(micro, name)
            break

    phi2_min_guess = 0.0

    print("=== TCV-phi : tests de cohérence du potentiel cohésif ===")
    print("Benchmark complet :", benchmark)
    print("Micro-paramètres  :", micro)
    print(f"Guess pour le minimum : phi1 ≈ {phi1_min_guess:.3e}, phi2 ≈ {phi2_min_guess:.3e}")
    print()

    # --- 2. Valeur et dérivées au minimum supposé ---
    V_min = pot.V_tot(phi1_min_guess, phi2_min_guess)
    dV1 = safe_call_dV_dphi1(pot, phi1_min_guess, phi2_min_guess)
    dV2 = safe_call_dV_dphi2(pot, phi1_min_guess, phi2_min_guess)

    print(f"V_tot(min_guess)      = {V_min:.6e}")
    print(f"dV/dphi1(min_guess)   = {dV1:.6e}")
    print(f"dV/dphi2(min_guess)   = {dV2:.6e}")
    print()

    # --- 3. Masse effective de Phi1 au minimum (Hessien numérique) ---
    if phi1_min_guess != 0.0:
        dphi = 1e-3 * abs(phi1_min_guess)
    else:
        dphi = 1e-3

    def V1_slice(phi1: float) -> float:
        return pot.V_tot(phi1, phi2_min_guess)

    V_plus = V1_slice(phi1_min_guess + dphi)
    V_cen = V1_slice(phi1_min_guess)
    V_minus = V1_slice(phi1_min_guess - dphi)

    m_eff1_sq_num = (V_plus - 2.0 * V_cen + V_minus) / (dphi ** 2)

    # m1 vient des micro-paramètres
    m1_sq = getattr(micro, "m1", 1.0) ** 2

    print("Masse effective numérique de Phi1 au minimum :")
    print(f"  m_eff1^2 (num)      = {m_eff1_sq_num:.6e}")
    print(f"  m1^2 (paramètre)    = {m1_sq:.6e}")
    if m1_sq != 0.0:
        print(f"  ratio m_eff1^2/m1^2 = {m_eff1_sq_num / m1_sq:.6e}")
    print()

    # --- 4. Assertions "douces" de cohérence ---
    phi_scale = max(abs(phi1_min_guess), 1.0)
    V_scale = m1_sq * phi_scale ** 2

    assert abs(V_min) < 1e-6 * V_scale, (
        "V_tot au minimum supposé n'est pas ~0 à 10^-6 près de l'échelle m1^2 * phi1^2. "
        "Vérifie la normalisation du potentiel / du vacuum."
    )

    grad_scale = m1_sq * phi_scale
    assert abs(dV1) < 1e-6 * grad_scale, (
        "dV/dphi1 au minimum supposé n'est pas ~0. "
        "Peut indiquer que le guess de phi1_min est mauvais, ou un bug dans dV_dphi1."
    )
    assert abs(dV2) < 1e-6 * grad_scale, (
        "dV/dphi2 au minimum supposé n'est pas ~0. "
        "Si Phi2=0 est bien le minimum, il faut vérifier dV_dphi2."
    )

    assert m_eff1_sq_num > 0.0, (
        "La masse effective de Phi1 au minimum est négative : "
        "le point n'est pas un minimum stable."
    )

    print("✅ Tous les tests de cohérence de base sont passés.")
    print("   (V(min) ≈ 0, gradient ≈ 0, m_eff1^2 > 0 et ~ m1^2)")


if __name__ == "__main__":
    main()