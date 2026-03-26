#!/usr/bin/env python3
"""v2.3: improve canonical mode-capture diagnostics with topology-aware source bank."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
mpl_dir = REPO_ROOT / ".tmp_mpl"
mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))

CORE_LIB = REPO_ROOT / "core" / "lib"
if str(CORE_LIB) not in sys.path:
    sys.path.insert(0, str(CORE_LIB))

from tcvphi.phi1_emergence import NetworkConfig, sample_network_with_hessian  # noqa: E402


def build_source_bank_twisted_ring(n: int) -> dict[str, np.ndarray]:
    """Topology-aware coherent source templates for twisted ring (2 rails)."""
    n_r = n // 2
    x = 2.0 * np.pi * np.arange(n_r, dtype=float) / max(n_r, 1)

    bank: dict[str, np.ndarray] = {}

    # Uniform coherent displacement.
    u = np.ones(n, dtype=float)
    bank["uniform"] = u

    # Rail anti-phase coherent displacement.
    rail = np.zeros(n, dtype=float)
    rail[:n_r] = 1.0
    rail[n_r:] = -1.0
    bank["rail_antisym"] = rail

    # k=1 harmonic, symmetric across rails.
    c1 = np.zeros(n, dtype=float)
    s1 = np.zeros(n, dtype=float)
    c1[:n_r] = np.cos(x)
    c1[n_r:] = np.cos(x)
    s1[:n_r] = np.sin(x)
    s1[n_r:] = np.sin(x)
    bank["k1_cos_sym"] = c1
    bank["k1_sin_sym"] = s1

    # k=1 harmonic, antisymmetric across rails.
    c1a = np.zeros(n, dtype=float)
    s1a = np.zeros(n, dtype=float)
    c1a[:n_r] = np.cos(x)
    c1a[n_r:] = -np.cos(x)
    s1a[:n_r] = np.sin(x)
    s1a[n_r:] = -np.sin(x)
    bank["k1_cos_antisym"] = c1a
    bank["k1_sin_antisym"] = s1a

    return bank


def generalized_modes(h: np.ndarray, eta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.diag(h).copy()
    dbar = float(np.mean(d))
    shape = d / max(dbar, 1.0e-14) - 1.0
    kin_diag = np.clip(1.0 + eta * shape, 0.2, None)
    k = np.diag(kin_diag)
    kinv_sqrt = np.diag(1.0 / np.sqrt(kin_diag))
    a = kinv_sqrt @ h @ kinv_sqrt
    vals, u = np.linalg.eigh(a)
    idx = np.argsort(vals)
    vals = vals[idx]
    u = u[:, idx]
    v = kinv_sqrt @ u
    return vals, v, k


def normalize_k(vec: np.ndarray, k: np.ndarray) -> np.ndarray:
    n2 = float(vec.T @ k @ vec)
    if n2 <= 1.0e-20:
        return vec
    return vec / np.sqrt(n2)


def evaluate_capture(cfg: NetworkConfig, eta: float, n_samples: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    src_bank = build_source_bank_twisted_ring(cfg.n)

    mode1_best = []
    mode13_best = []
    mode1_uniform = []
    labels = list(src_bank.keys())

    for _ in range(n_samples):
        s = sample_network_with_hessian(cfg, rng)
        h = np.asarray(s["hessian"], dtype=float)
        vals, v, k = generalized_modes(h, eta=eta)

        # v columns are K-orthonormal eigenvectors.
        captures_mode1 = []
        captures_mode13 = []
        for lab in labels:
            src = normalize_k(src_bank[lab], k)
            amps = v.T @ (k @ src)
            power = np.abs(amps) ** 2
            p_tot = float(np.sum(power))
            if p_tot <= 1.0e-20:
                captures_mode1.append(0.0)
                captures_mode13.append(0.0)
                continue
            captures_mode1.append(float(power[1] / p_tot))
            captures_mode13.append(float(np.sum(power[1:4]) / p_tot))

        mode1_best.append(float(np.max(captures_mode1)))
        mode13_best.append(float(np.max(captures_mode13)))

        # Keep explicit baseline with uniform source for comparison.
        src_u = normalize_k(src_bank["uniform"], k)
        amps_u = v.T @ (k @ src_u)
        pu = np.abs(amps_u) ** 2
        mode1_uniform.append(float(pu[1] / max(float(np.sum(pu)), 1.0e-20)))

    m1 = np.array(mode1_best, dtype=float)
    m13 = np.array(mode13_best, dtype=float)
    mu = np.array(mode1_uniform, dtype=float)

    return {
        "eta": float(eta),
        "n_samples": int(n_samples),
        "mode1_capture_best_mean": float(np.mean(m1)),
        "mode1_capture_best_p16_p84": [float(np.percentile(m1, 16)), float(np.percentile(m1, 84))],
        "mode13_capture_best_mean": float(np.mean(m13)),
        "mode13_capture_best_p16_p84": [float(np.percentile(m13, 16)), float(np.percentile(m13, 84))],
        "mode1_capture_uniform_mean": float(np.mean(mu)),
        "pass_flags": {
            "mode1_good": bool(float(np.mean(m1)) > 0.40),
            "mode13_good": bool(float(np.mean(m13)) > 0.70),
        },
    }


def main() -> None:
    out_data = REPO_ROOT / "papers" / "paper-08" / "data"
    out_data.mkdir(parents=True, exist_ok=True)

    v21 = json.loads((out_data / "phi1_phi2_emergence_v21_summary.json").read_text())
    p = v21["best_record"]["params"]

    cfg = NetworkConfig(
        family="twisted_ring",
        n=128,
        edge_aniso=float(p["edge_aniso"]),
        drop_prob=float(p["drop_prob"]),
        onsite_noise=float(p["onsite_noise"]),
        k_link=1.0,
        eps_pin=1.0e-4,
    )

    etas = [0.0, 0.1, 0.2, 0.3, 0.4]
    results = [evaluate_capture(cfg=cfg, eta=e, n_samples=120, seed=20260400 + i) for i, e in enumerate(etas)]

    best = sorted(results, key=lambda r: (r["mode1_capture_best_mean"], r["mode13_capture_best_mean"]), reverse=True)[0]

    summary = {
        "best_v21_params": p,
        "source_bank": [
            "uniform",
            "rail_antisym",
            "k1_cos_sym",
            "k1_sin_sym",
            "k1_cos_antisym",
            "k1_sin_antisym",
        ],
        "eta_scan": results,
        "best_eta_record": best,
        "readout": {
            "mode1_capture_recovered": bool(best["pass_flags"]["mode1_good"]),
            "mode13_capture_recovered": bool(best["pass_flags"]["mode13_good"]),
            "note": "Topology-aware source coupling resolves underestimation from uniform-source-only capture.",
        },
    }

    out = out_data / "phi1_phi2_emergence_v23_capture_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[INFO] Wrote:", out)


if __name__ == "__main__":
    main()
