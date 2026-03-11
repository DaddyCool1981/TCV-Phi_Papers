import numpy as np
from classy import Class

# =============================
# 1. Paramètres TCV-Φ "canoniques"
# =============================

def get_tcv_canonical_params():
    """
    Paramètres cosmologiques effectifs correspondant
    au point canonique TCV-Φ (à raffiner si besoin).

    TCV-Φ minimal :
      - Φ1 fixe (ns, As, nrun, r)
      - Φ2 fixe Ω_DM (ici vue comme CDM effective)
    """
    params = {
        "H0": 67.4,
        "ombh2": 0.0224,
        "omch2": 0.120,
        "As": 2.1e-9,
        "ns": 0.964,
        #"nrun": -3e-4,
        "r": 0.0,       # r~10^-7 négligeable pour TT/TE/EE
        "tau": 0.054,
        "YHe": 0.245,   # fixé (BBN-compatible)
    }
    return params


# =============================
# 2. Mapping TCV → params CLASS
# =============================

def tcv_to_class_params(theta_tcv=None, want_pk=True):
    """
    Convertit un set de paramètres TCV-Φ (optionnel)
    en dictionnaire de paramètres CLASS.

    Pour l'instant, si theta_tcv est None, on prend
    le point canonique TCV-Φ (Paper VI).
    """
    if theta_tcv is None:
        theta_tcv = get_tcv_canonical_params()

    h = theta_tcv["H0"] / 100.0

    params_class = {
        "h": h,
        "omega_b": theta_tcv["ombh2"] / h**2,
        "omega_cdm": theta_tcv["omch2"] / h**2,
        "A_s": theta_tcv["As"],
        "n_s": theta_tcv["ns"],
    #    "n_run": theta_tcv["nrun"],
        "tau_reio": theta_tcv["tau"],
        "YHe": theta_tcv["YHe"],
    }

    if want_pk:
        params_class["output"] = "mPk"
        # bornes de k raisonnables
        params_class["P_k_max_h/Mpc"] = 10.0

    return params_class


# =============================
# 3. Wrapper CLASS pour P(k)
# =============================

def get_pk_class(theta_tcv=None, k_vals=None, z=0.0):
    """
    Renvoie P(k,z) pour un set TCV-Φ donné,
    calculé avec CLASS.

    k_vals : array de k en h/Mpc
    """
    if k_vals is None:
        # par défaut, fenêtre large mais sûre
        k_vals = np.logspace(-4, 0, 300)  # max ~ 1 h/Mpc

    params_class = tcv_to_class_params(theta_tcv, want_pk=True)

    cosmo = Class()
    cosmo.set(params_class)
    cosmo.compute()

    pk = np.array([cosmo.pk(float(k), float(z)) for k in k_vals])

    cosmo.struct_cleanup()
    cosmo.empty()

    return k_vals, pk


# =============================
# 4. Modèle phénoménologique de Φ2 fuzzy cohésive
# =============================

def suppression_fuzzy(k, kJ, n=4):
    """
    Facteur de suppression phenomenologique pour la
    composante fuzzy DM cohésive Φ2.

    kJ : échelle de cohésion/Jeans (h/Mpc)
    n  : raideur de la coupure.
    """
    return 1.0 / (1.0 + (k / kJ)**(2 * n))


def get_pk_fuzzy_from_cdm(k_vals, pk_cdm, kJ, n=4, f_coh=1.0):
    """
    Construit un spectre de matière effectif pour une
    DM partiellement (ou totalement) cohésive/fuzzy.

    f_coh : fraction de la DM totale portée par Φ2 cohésive.

    - Si f_coh = 1 : toute la DM = Φ2 fuzzy cohésive.
    - Si 0 < f_coh < 1 : mélange Φ2 fuzzy + CDM standard.
    """
    T2 = suppression_fuzzy(k_vals, kJ, n=n)

    # Spectre de Φ2 fuzzy seule
    pk_fuzzy = pk_cdm * T2

    # Mélange quadratique des composantes de DM :
    pk_eff = (f_coh**2) * pk_fuzzy + (1.0 - f_coh)**2 * pk_cdm

    return pk_eff, pk_fuzzy, T2