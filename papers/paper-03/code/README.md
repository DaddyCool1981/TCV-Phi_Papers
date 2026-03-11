# Paper 03 Code Plan (Primordial sector)

## Objective
Generate all numerical products used in Paper III (Mukhanov-Sasaki and primordial observables) with strict reproducibility.

## Required outputs
- `papers/paper-03/data/PR_tcvphi.dat`
- `papers/paper-03/data/primordial_observables.json`
- `papers/paper-03/data/class_camb_bridge_summary.json`
- Figures in `papers/paper-03/figs/` matching manuscript labels.

## Interface policy
- Reuse `core/lib/tcvphi` modules and Paper I/II conventions.
- Keep CLASS/CAMB integration backward-compatible.
- In Paper III, CLASS/CAMB are used as consistency checks unless a full likelihood pipeline is explicitly added.

## Minimal test checklist
- MS solver stability under time-step and k-grid refinement.
- Robust extraction of `(n_s, A_s, r, alpha_s)` across benchmark windows.
- Reproducible export/import of tabulated `P_R(k)`.
- Stable script execution from repo root.
