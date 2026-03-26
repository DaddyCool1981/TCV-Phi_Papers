#!/usr/bin/env python3
"""Robustness checks for twisted-multi-ring geometry across fermion sectors."""

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

from tcvphi.flavor_twisted_multi_ring import (  # noqa: E402
    TwistedMultiRingParams,
    build_effective_charged_lepton_matrix,
    build_effective_neutrino_matrix,
    build_twisted_multi_ring_hamiltonian,
    diagonalize,
    extract_pmns_angles_deg,
    pmns_from_effective_matrices,
)


def _load_best_candidate(path: Path) -> tuple[TwistedMultiRingParams, dict]:
    d = json.loads(path.read_text(encoding="utf-8"))
    best = d["best_candidates"][0]
    p = TwistedMultiRingParams(**best["params"])
    c = best["controls"]
    return p, c


def _strict_ok(a: dict) -> bool:
    return 40.0 <= a["theta23"] <= 50.0 and 2.0 <= a["theta13"] <= 12.0 and 10.0 <= a["theta12"] <= 40.0


def main() -> None:
    out_data = REPO_ROOT / "papers" / "paper-05" / "data"
    out_figs = REPO_ROOT / "papers" / "paper-05" / "figs"
    out_data.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    summary_path = out_data / "twisted_multi_ring_flavour_toy_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError("Run compute_twisted_multi_ring_flavor_toy.py first.")

    p_best, c_best = _load_best_candidate(summary_path)
    h = build_twisted_multi_ring_hamiltonian(p_best)
    vals, vecs = diagonalize(h)

    rng = np.random.default_rng(20260315)

    n_lepton = 3000
    lepton_angles = []
    lepton_ok = 0
    for _ in range(n_lepton):
        bridge = float(np.clip(c_best["bridge"] * rng.uniform(0.9, 1.1), 0.5, 3.0))
        m_nu = build_effective_neutrino_matrix(vals, vecs, bridge=bridge)

        eps_l = float(np.clip(c_best["epsilon_l"] * rng.uniform(0.5, 2.0), 5e-4, 2e-2))
        anc_l = float(np.clip(c_best["anchor"] * rng.uniform(0.5, 1.8), 0.3, 3.0))
        m_l = build_effective_charged_lepton_matrix(vals, epsilon_l=eps_l, anchor=anc_l)

        u = pmns_from_effective_matrices(m_l, m_nu)
        a = extract_pmns_angles_deg(u)
        lepton_angles.append(a)
        lepton_ok += int(_strict_ok(a))

    n_quark = 3000
    q12, q13, q23 = [], [], []
    for _ in range(n_quark):
        eps_u = float(rng.uniform(2e-5, 5e-4))
        eps_d = float(rng.uniform(2e-5, 8e-4))
        anc_u = float(rng.uniform(0.5, 2.0))
        anc_d = float(rng.uniform(0.5, 2.0))

        m_u = build_effective_charged_lepton_matrix(
            vals,
            epsilon_l=eps_u,
            anchor=anc_u,
            hierarchy=(2.2e-3, 1.27, 173.0),
        )
        m_d = build_effective_charged_lepton_matrix(
            vals,
            epsilon_l=eps_d,
            anchor=anc_d,
            hierarchy=(4.7e-3, 9.6e-2, 4.18),
        )
        _, uu = np.linalg.eigh(m_u)
        _, ud = np.linalg.eigh(m_d)
        v_ckm = uu.conjugate().T @ ud
        a = extract_pmns_angles_deg(v_ckm)
        q12.append(float(a["theta12"]))
        q13.append(float(a["theta13"]))
        q23.append(float(a["theta23"]))

    l12 = np.array([a["theta12"] for a in lepton_angles], dtype=float)
    l13 = np.array([a["theta13"] for a in lepton_angles], dtype=float)
    l23 = np.array([a["theta23"] for a in lepton_angles], dtype=float)

    summary = {
        "status": "twisted-multi-ring sector compatibility check (exploratory)",
        "reference_candidate": {"params": p_best.__dict__, "controls": c_best},
        "lepton_robustness": {
            "n_samples": int(n_lepton),
            "n_strict_ok": int(lepton_ok),
            "strict_fraction": float(lepton_ok / n_lepton),
            "theta12_mean": float(np.mean(l12)),
            "theta13_mean": float(np.mean(l13)),
            "theta23_mean": float(np.mean(l23)),
            "theta12_std": float(np.std(l12)),
            "theta13_std": float(np.std(l13)),
            "theta23_std": float(np.std(l23)),
        },
        "quark_like_compatibility": {
            "n_samples": int(n_quark),
            "theta12_mean": float(np.mean(q12)),
            "theta13_mean": float(np.mean(q13)),
            "theta23_mean": float(np.mean(q23)),
            "theta12_p95": float(np.percentile(q12, 95)),
            "theta13_p95": float(np.percentile(q13, 95)),
            "theta23_p95": float(np.percentile(q23, 95)),
            "note": "Small angles indicate compatibility with a weak-mixing quark-like sector in this toy layer.",
        },
        "notes": [
            "This does not prove full SM flavour unification.",
            "It checks whether twisted-multi-ring neutrino success is stable against charged-lepton perturbations.",
            "And whether a small-mixing quark-like toy can coexist on the same geometry.",
        ],
    }

    out_json = out_data / "twisted_multi_ring_sector_compatibility_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[INFO] Wrote:", out_json)

    plt.figure(figsize=(6.2, 4.2))
    plt.scatter(l12, l13, s=10, alpha=0.25)
    plt.xlabel(r"$\theta_{12}$ (deg)")
    plt.ylabel(r"$\theta_{13}$ (deg)")
    plt.title("Twisted-multi-ring lepton robustness cloud")
    plt.grid(ls=":", alpha=0.4)
    plt.tight_layout()
    f1 = out_figs / "twisted_multi_ring_lepton_robustness_cloud.png"
    plt.savefig(f1, dpi=180, bbox_inches="tight")
    plt.close()
    print("[INFO] Wrote:", f1)

    plt.figure(figsize=(7.0, 4.2))
    bins = np.linspace(0.0, max(1.0, np.percentile(q12 + q13 + q23, 99)), 30)
    plt.hist(q12, bins=bins, alpha=0.5, label=r"$\theta_{12}^{q}$")
    plt.hist(q13, bins=bins, alpha=0.5, label=r"$\theta_{13}^{q}$")
    plt.hist(q23, bins=bins, alpha=0.5, label=r"$\theta_{23}^{q}$")
    plt.xlabel("angle (deg)")
    plt.ylabel("count")
    plt.title("Twisted-multi-ring quark-like mixing toy")
    plt.grid(ls=":", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    f2 = out_figs / "twisted_multi_ring_quark_like_hist.png"
    plt.savefig(f2, dpi=180, bbox_inches="tight")
    plt.close()
    print("[INFO] Wrote:", f2)


if __name__ == "__main__":
    main()
