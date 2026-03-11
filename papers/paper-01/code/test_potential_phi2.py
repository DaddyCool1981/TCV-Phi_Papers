"""
test_potential_phi2.py

Petits tests de cohérence pour le potentiel de la composante cohésive légère Phi2.

Objectifs :
- Vérifier que le minimum global en Phi2 est bien autour de phi2 = 0 (pour phi1 fixé à phi0)
- Vérifier que dV/dphi2(phi2=0) ~ 0
- Estimer numériquement la masse effective de Phi2 au minimum et la comparer au paramètre m2

Ce n'est pas une "preuve" mathématique, mais un sanity check numérique
pour aider à fixer/contrôler les paramètres de l'EFT pour le secteur DM cohésif.
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


def main() -> None:
    # --- 1. Construction des paramètres et du potentiel ---
    benchmark = make_default_benchmark()
    micro = benchmark.micro  # contient m1, phi0, m2, etc.
    pot = CohesivePotential(micro)

    # On fixe Phi1 à son minimum attendu (phi0)
    phi1_min = getattr(micro, "phi0", 0.0)
    # Pour Phi2, on s'attend à un minimum en 0
    phi2_min_guess = 0.0

    print("=== TCV-phi : tests de cohérence du potentiel pour Phi2 (DM cohésive) ===")
    print("Benchmark complet :", benchmark)
    print("Micro-paramètres  :", micro)
    print(f"Phi1 fixé au minimum attendu : phi1 = {phi1_min:.3e}")
    print(f"Guess pour le minimum de Phi2 : phi2 ≈ {phi2_min_guess:.3e}")
    print()

    # --- 2. Valeur et dérivée au minimum supposé en Phi2 ---
    V_min_phi2 = pot.V_tot(phi1_min, phi2_min_guess)

    # dV/dphi2 : on utilise la méthode analytique si disponible
    if hasattr(pot, "dV2_dphi2"):
        dV2 = pot.dV2_dphi2(phi2_min_guess)  # type: ignore[attr-defined]
    else:
        # Fallback : dérivée numérique si jamais la méthode n'existait pas
        dphi = 1.0
        def V_slice(phi2: float) -> float:
            return pot.V_tot(phi1_min, phi2)
        dV2 = (V_slice(phi2_min_guess + dphi) - V_slice(phi2_min_guess - dphi)) / (2.0 * dphi)

    print(f"V_tot(phi1_min, phi2_min_guess) = {V_min_phi2:.6e}")
    print(f"dV/dphi2(min_guess)            = {dV2:.6e}")
    print()

    # --- 3. Masse effective de Phi2 au minimum (Hessien numérique) ---
    # On fait une coupe en phi2, en gardant phi1 = phi1_min fixé
    # On prend un pas dphi2 "raisonnable" : 1 unité (par exemple 1 GeV)
    dphi2 = 1.0

    def V2_slice(phi2: float) -> float:
        return pot.V_tot(phi1_min, phi2)

    V_plus = V2_slice(phi2_min_guess + dphi2)
    V_cen = V2_slice(phi2_min_guess)
    V_minus = V2_slice(phi2_min_guess - dphi2)

    m_eff2_sq_num = (V_plus - 2.0 * V_cen + V_minus) / (dphi2 ** 2)

    # m2 vient des micro-paramètres (masse de la DM cohésive)
    m2 = getattr(micro, "m2", 0.0)
    m2_sq = m2 ** 2

    print("Masse effective numérique de Phi2 au minimum :")
    print(f"  m_eff2^2 (num)      = {m_eff2_sq_num:.6e}")
    print(f"  m2^2 (paramètre)    = {m2_sq:.6e}")
    if m2_sq != 0.0:
        print(f"  ratio m_eff2^2/m2^2 = {m_eff2_sq_num / m2_sq:.6e}")
    print()

    # --- 4. Assertions "douces" de cohérence ---
    # Échelle naturelle pour les variations en Phi2 : on prend une échelle de 1 (si phi2_min = 0)
    phi2_scale = max(abs(phi2_min_guess), 1.0)

    # Échelle d'énergie associée à Phi2 : m2^2 * phi2^2
    # (ici purement indicative, m2 est très petit mais ce n'est pas grave pour le test)
    V_scale_phi2 = max(m2_sq * (phi2_scale ** 2), 1e-100)

    # On ne demande pas que V_tot soit ~0 à cette échelle (la valeur de V1 domine),
    # ce qui nous intéresse ici est surtout la dérivée et la masse effective de Phi2.
    grad_scale_phi2 = max(m2_sq * phi2_scale, 1e-100)

    # dV/dphi2 au minimum doit être ~0
    assert abs(dV2) < 1e-6 * grad_scale_phi2, (
        "dV/dphi2 au minimum supposé n'est pas ~0. "
        "Si Phi2=0 est bien le minimum, cela peut indiquer un bug dans dV2_dphi2 "
        "ou un mauvais choix de paramètres."
    )

    # La masse effective doit être positive (minimum, pas maximum)
    assert m_eff2_sq_num > 0.0, (
        "La masse effective de Phi2 au minimum est négative : "
        "le point n'est pas un minimum stable dans la direction Phi2."
    )

    # Et cohérente avec m2^2 à un facteur O(1) près (on laisse une marge large car les échelles sont minuscules)
    if m2_sq > 0.0:
        ratio = m_eff2_sq_num / m2_sq
        assert 0.1 < ratio < 10.0, (
            f"m_eff2^2 (numérique) s'écarte trop de m2^2. "
            f"Ratio = {ratio:.3e} ; vérifier la normalisation du potentiel V2."
        )

    print("✅ Tests de cohérence pour Phi2 passés.")
    print("   (dV/dphi2(min) ≈ 0, m_eff2^2 > 0 et ~ m2^2)")


if __name__ == "__main__":
    main()