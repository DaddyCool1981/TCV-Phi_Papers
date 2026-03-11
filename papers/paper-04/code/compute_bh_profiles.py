#!/usr/bin/env python3
"""Generate Paper-04 static cohesive profile families."""

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

from tcvphi.bh_static import (  # noqa: E402
    StaticBHParams,
    params_to_dict,
    solve_static_ivp,
)


def main() -> None:
    out_data = REPO_ROOT / "papers" / "paper-04" / "data"
    out_figs = REPO_ROOT / "papers" / "paper-04" / "figs"
    out_data.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    # Family scan: progressively denser cohesive closure.
    families = [
        ("regular_core", 1.0e-26, 1.0),
        ("compact_core", 1.0e-24, 1.0),
        ("dense_core", 1.0e-22, 1.0),
    ]
    stacked: dict[str, list[np.ndarray]] = {
        "r": [],
        "m": [],
        "phi_metric": [],
        "phi1": [],
        "f_metric": [],
        "rho": [],
        "pr": [],
        "pt": [],
        "ricci_scalar": [],
        "ricci_sq": [],
        "kretschmann_proxy": [],
        "adm_mass": [],
        "horizon_radius": [],
        "solver_success": [],
    }
    names: list[str] = []
    metadata: dict[str, dict[str, object]] = {}

    for name, rho_cc0, phi1_center in families:
        params = StaticBHParams(
            rho_cc0=rho_cc0,
            alpha_cc=0.0,
            beta_cc=0.0,
            xi0=0.0,
            r_max=1.0e4,
            r_points=1800,
        )
        prof, sol = solve_static_ivp(phi1_center=phi1_center, params=params, method="RK45")
        if not sol.success:
            print(f"[WARN] {name}: solver failed ({sol.message})")
            continue
        names.append(name)
        metadata[name] = {
            "params": params_to_dict(params),
            "phi1_center": float(phi1_center),
            "solver_message": sol.message,
            "n_steps": int(sol.t.size),
        }
        for key in stacked:
            stacked[key].append(np.asarray(prof[key]))

        print(
            f"[INFO] {name}: M_adm={float(prof['adm_mass'][0]):.3e}, "
            f"r_h={float(prof['horizon_radius'][0]):.3e}"
        )

    if not names:
        raise RuntimeError("No converged family; adjust benchmark controls.")

    arrays = {
        "family_names": np.asarray(names),
        "r_eval": stacked["r"][0],
    }
    for key, values in stacked.items():
        arrays[key] = np.stack(values, axis=0)

    np.savez(out_data / "bh_profiles.npz", **arrays)
    with (out_data / "bh_profiles_meta.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print("[INFO] Wrote:", out_data / "bh_profiles.npz")
    print("[INFO] Wrote:", out_data / "bh_profiles_meta.json")

    # Figure 1: metric and mass profiles.
    r = arrays["r"]
    f_metric = arrays["f_metric"]
    m = arrays["m"]

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for i, label in enumerate(names):
        ax[0].plot(r[i], f_metric[i], lw=1.4, label=label)
        ax[1].plot(r[i], m[i], lw=1.4, label=label)
    ax[0].set_xscale("log")
    ax[1].set_xscale("log")
    ax[0].set_xlabel(r"$r$")
    ax[1].set_xlabel(r"$r$")
    ax[0].set_ylabel(r"$F(r)=1-2m(r)/r$")
    ax[1].set_ylabel(r"$m(r)$")
    ax[0].grid(True, which="both", ls=":")
    ax[1].grid(True, which="both", ls=":")
    ax[0].legend()
    ax[1].legend()
    fig.suptitle("Paper-04 static cohesive profile families")
    fig.tight_layout()
    fig.savefig(out_figs / "bh_metric_profiles.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("[INFO] Wrote:", out_figs / "bh_metric_profiles.png")


if __name__ == "__main__":
    main()
