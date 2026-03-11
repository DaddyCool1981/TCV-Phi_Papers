#!/usr/bin/env python3
"""Build photon-ring and echo template observables for Paper-04 Kerr-like branch."""

from __future__ import annotations

import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
mpl_dir = REPO_ROOT / ".tmp_mpl"
mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _photon_ring_radius(r: np.ndarray, f_metric: np.ndarray) -> float | None:
    fp = np.gradient(f_metric, r, edge_order=2)
    eq = 2.0 * f_metric - r * fp
    idx = np.where(np.sign(eq[:-1]) != np.sign(eq[1:]))[0]
    if idx.size == 0:
        return None
    j = int(idx[-1])  # outermost branch
    x0, x1 = r[j], r[j + 1]
    y0, y1 = eq[j], eq[j + 1]
    if abs(y1 - y0) < 1.0e-20:
        return float(x0)
    return float(x0 - y0 * (x1 - x0) / (y1 - y0))


def _echo_template(
    r: np.ndarray,
    f_metric: np.ndarray,
    r_core: float,
    r_ph: float,
) -> tuple[float | None, float | None]:
    mask = (r >= r_core) & (r <= r_ph)
    if np.count_nonzero(mask) < 5:
        return None, None
    rr = r[mask]
    ff = np.maximum(f_metric[mask], 1.0e-8)
    dt = 2.0 * float(np.trapezoid(1.0 / ff, rr))
    if dt <= 0.0:
        return None, None
    f_echo = float(np.pi / dt)
    return dt, f_echo


def main() -> None:
    out_data = REPO_ROOT / "papers" / "paper-04" / "data"
    out_figs = REPO_ROOT / "papers" / "paper-04" / "figs"

    profiles_npz = out_data / "bh_profiles.npz"
    kerr_npz = out_data / "bh_kerr_slow.npz"
    if not profiles_npz.exists():
        raise FileNotFoundError(f"Missing {profiles_npz}; run compute_bh_profiles.py first.")
    if not kerr_npz.exists():
        raise FileNotFoundError(f"Missing {kerr_npz}; run compute_bh_kerr_slow.py first.")

    pdat = np.load(profiles_npz, allow_pickle=False)
    kdat = np.load(kerr_npz, allow_pickle=False)

    families = [str(x) for x in pdat["family_names"]]
    family_used = str(kdat["family"][0])
    if family_used in families:
        i = families.index(family_used)
    else:
        i = len(families) - 1

    r = pdat["r"][i]
    f_metric = pdat["f_metric"][i]
    m = pdat["m"][i]
    a_values = kdat["a_over_m"]
    omega = kdat["omega"]

    m_adm = float(m[-1])
    r_core = float(r[np.argmin(np.abs(r - r[0]))])
    r_ph = _photon_ring_radius(r, f_metric)
    r_ph_schw = 3.0 * m_adm

    if r_ph is None and r_ph_schw > 0.0:
        # Conservative fallback: cohesive shift template from effective coupling.
        xi0 = 0.1
        phi1_ref = float(np.interp(float(kdat["r_ref"][0]), r, pdat["phi1"][i]))
        r_ph = r_ph_schw / (1.0 + xi0 * phi1_ref**2)
        photon_model = "effective_coupling_template"
    else:
        photon_model = "direct_profile_root" if r_ph is not None else "unresolved"

    photon_shift = (r_ph - r_ph_schw) / r_ph_schw if (r_ph is not None and r_ph_schw > 0.0) else None

    # Echo template on Schwarzschild-like exterior with cohesive core proxy.
    dt_echo, f_echo = (None, None)
    if r_ph is not None:
        r_core_template = 2.0 * m_adm * 1.02
        rr = np.geomspace(r_core_template, max(r_ph, r_core_template * 1.001), 800)
        f_schw = 1.0 - 2.0 * m_adm / np.maximum(rr, 1.0e-14)
        dt_echo, f_echo = _echo_template(rr, f_schw, r_core=r_core_template, r_ph=r_ph)

    # Eikonal QNM template from photon ring.
    qnm_template = []
    if r_ph is not None:
        omega_ph = float(np.sqrt(max(np.interp(r_ph, r, f_metric), 1.0e-14)) / r_ph)
        for a in a_values:
            # Conservative linear-in-spin template around static branch.
            omega_r = omega_ph * (1.0 + 0.6 * float(a))
            qnm_template.append(float(omega_r))
    else:
        qnm_template = [None for _ in a_values]

    summary = {
        "family_used": family_used,
        "m_adm": m_adm,
        "r_core_proxy": r_core,
        "r_photon": r_ph,
        "r_photon_schwarzschild": r_ph_schw,
        "photon_ring_relative_shift": photon_shift,
        "photon_ring_model": photon_model,
        "delta_t_echo_template": dt_echo,
        "f_echo_template": f_echo,
        "a_over_m_values": [float(x) for x in a_values],
        "omega_r_eikonal_template": qnm_template,
        "note": "Photon-ring and echo outputs are template-level diagnostics in the current slow-rotation branch.",
    }

    with (out_data / "bh_kerr_observables.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("[INFO] Wrote:", out_data / "bh_kerr_observables.json")

    # Figure: photon-ring proxy.
    plt.figure(figsize=(6.8, 4.6))
    fp = np.gradient(f_metric, r, edge_order=2)
    eq = 2.0 * f_metric - r * fp
    plt.plot(r, eq, lw=1.6, label=r"$2F-rF'$")
    if r_ph is not None:
        plt.axvline(r_ph, ls="--", color="C3", label=rf"$r_{{\rm ph}}={r_ph:.3e}$")
    plt.axhline(0.0, color="k", lw=0.9)
    plt.xscale("log")
    plt.xlabel(r"$r$")
    plt.ylabel(r"$2F-rF'$")
    plt.title("Photon-ring condition (template)")
    plt.grid(True, which="both", ls=":")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_figs / "bh_kerr_photon_ring.png", dpi=180, bbox_inches="tight")
    plt.close()
    print("[INFO] Wrote:", out_figs / "bh_kerr_photon_ring.png")

    # Figure: echo + qnm template versus spin.
    fig, ax1 = plt.subplots(figsize=(7.2, 4.6))
    x = np.asarray(a_values, dtype=float)
    y_qnm = np.array([np.nan if y is None else y for y in qnm_template], dtype=float)
    ax1.plot(x, y_qnm, "o-", lw=1.6, label=r"$\omega_R$ template")
    ax1.set_xlabel(r"$a/M$")
    ax1.set_ylabel(r"$\omega_R$ template")
    ax1.grid(True, ls=":")
    if f_echo is not None:
        ax2 = ax1.twinx()
        ax2.plot(x, np.full_like(x, f_echo), "s--", color="C3", lw=1.4, label=r"$f_{\rm echo}$ template")
        ax2.set_ylabel(r"$f_{\rm echo}$ template")
    fig.tight_layout()
    fig.savefig(out_figs / "bh_kerr_echo_template.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("[INFO] Wrote:", out_figs / "bh_kerr_echo_template.png")


if __name__ == "__main__":
    main()
