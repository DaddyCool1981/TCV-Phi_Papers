"""
print_benchmark_params.py

Petit utilitaire pour afficher proprement tous les paramètres
du benchmark TCV-phi (micro + dérivés cosmologiques + BH).

Usage :
    cd TCV-PHI/code
    python print_benchmark_params.py
"""

from __future__ import annotations

import os
import sys
from typing import Any

# --- 0. Rendre le package tcvphi importable quand on lance depuis code/ ---

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)  # répertoire TCV-PHI
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tcvphi.params import make_default_benchmark  # type: ignore[import]


# ---------- Helpers de formatage ----------

def fmt_value(val: Any) -> str:
    """
    Format un scalaire de manière lisible :
    - float/int : notation scientifique si |val| > 1e4 ou < 1e-3
    - autrement : format "normal"
    - autres types : str(val)
    """
    if isinstance(val, (int, float)):
        if val == 0:
            return "0"
        abs_val = abs(val)
        if abs_val >= 1e4 or abs_val <= 1e-3:
            return f"{val:.3e}"
        else:
            return f"{val:.6g}"
    return str(val)


def print_section(title: str) -> None:
    print()
    print("=" * (len(title) + 4))
    print(f"  {title}")
    print("=" * (len(title) + 4))


def print_param_dict(label: str, data: dict[str, Any]) -> None:
    print_section(label)
    # Alignement des noms de paramètres
    if not data:
        print("  (aucun paramètre)")
        return

    max_key_len = max(len(k) for k in data.keys())

    for k in sorted(data.keys()):
        v = data[k]
        print(f"  {k:<{max_key_len}} = {fmt_value(v)}")


# ---------- Script principal ----------

def main() -> None:
    benchmark = make_default_benchmark()

    print("=== TCV-phi : benchmark global des paramètres EFT ===")
    print()
    print(f"Nom du benchmark : {getattr(benchmark, 'name', 'N/A')}")
    print()

    # 1) Paramètres micro (ceux de l'EFT de base : Φ1, Φ2, couplage non minimal, etc.)
    micro = benchmark.micro
    micro_dict = {
        # On explicite l'ordre pour que ça ressemble à une future "Table 1"
        "m1"      : getattr(micro, "m1", None),
        "phi0"    : getattr(micro, "phi0", None),
        "lambda1" : getattr(micro, "lambda1", None),
        "kappa"   : getattr(micro, "kappa", None),
        "xi0"     : getattr(micro, "xi0", None),
        "m2"      : getattr(micro, "m2", None),
        "Lambda_B": getattr(micro, "Lambda_B", None),
        "rho_vac" : getattr(micro, "rho_vac", None),
        "M_Pl"    : getattr(micro, "M_Pl", None),
    }
    print_param_dict("Paramètres microscopiques de l'EFT (Φ1, Φ2, gravité)", micro_dict)

    # 2) Paramètres cosmologiques dérivés (bounce, primordial, DM, baryogénèse...)
    cosmo = benchmark.cosmo
    cosmo_dict = {
        # Bounce / background
        "a_b"                  : getattr(cosmo, "a_b", None),
        "eta0"                 : getattr(cosmo, "eta0", None),
        "DeltaPhi1_over_Phi0"  : getattr(cosmo, "DeltaPhi1_over_Phi0", None),
        "C0"                   : getattr(cosmo, "C0", None),
        # Primordial scalar/tensor
        "n_s"                  : getattr(cosmo, "n_s", None),
        "A_s"                  : getattr(cosmo, "A_s", None),
        "alpha_s"              : getattr(cosmo, "alpha_s", None),
        "r"                    : getattr(cosmo, "r", None),
        # Baryogénèse
        "eta_B"                : getattr(cosmo, "eta_B", None),
        # DM cohésive
        "phi2_0"               : getattr(cosmo, "phi2_0", None),
        "f_dm_coh"             : getattr(cosmo, "f_dm_coh", None),
        "Omega_dm_h2"          : getattr(cosmo, "Omega_dm_h2", None),
    }
    print_param_dict("Paramètres cosmologiques dérivés (bounce, primordial, DM, baryogénèse)", cosmo_dict)

    # 3) Infos sur les solutions de trous noirs cohésifs (si déjà présentes dans le benchmark)
    bh_solutions = getattr(benchmark, "bh_solutions", {})
    print_section("Solutions de trous noirs cohésifs (si connues)")
    if not bh_solutions:
        print("  Aucune solution stockée pour l'instant (dict vide).")
    else:
        for key, sol in bh_solutions.items():
            print(f"  - {key}: {sol}")

    # 4) Petit résumé calculé
    print()
    print("Résumé rapide :")
    m1 = micro_dict["m1"]
    phi0 = micro_dict["phi0"]
    m2 = micro_dict["m2"]
    n_s = cosmo_dict["n_s"]
    A_s = cosmo_dict["A_s"]
    r_ = cosmo_dict["r"]

    if isinstance(m1, (int, float)) and isinstance(phi0, (int, float)):
        print(f"  Échelle structurelle : m1 ≈ {fmt_value(m1)}  (GeV)")
        print(f"  Valeur d'équilibre Φ1 : phi0 ≈ {fmt_value(phi0)}  (GeV)")
    if isinstance(m2, (int, float)):
        print(f"  Masse DM cohésive Φ2 : m2 ≈ {fmt_value(m2)}  (GeV)")
    if isinstance(n_s, (int, float)) and isinstance(A_s, (int, float)):
        print(f"  Spectre scalaire primordial : n_s ≈ {fmt_value(n_s)},  A_s ≈ {fmt_value(A_s)}")
    if isinstance(r_, (int, float)):
        print(f"  Rapport tensor/scalaire : r ≈ {fmt_value(r_)}")

    print()
    print("✅ Impression des paramètres benchmark TCV-phi terminée.")


if __name__ == "__main__":
    main()