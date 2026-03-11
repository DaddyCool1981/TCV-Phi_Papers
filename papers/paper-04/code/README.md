# Paper 04 Code Plan (BH sector)

## Objective
Generate reproducible static strong-field outputs for cohesive non-singular BH benchmark families.

## Canonical script split
- `compute_bh_profiles.py`: solve radial ODE system for `{Phi(r), m(r), Phi1(r), Phi2(r)}` and save raw profiles.
- `compute_bh_diagnostics.py`: evaluate regularity invariants, matching residuals, and convergence metrics.
- `compute_bh_templates.py` (optional): produce conservative signature templates (ringdown/shadow proxies) from solved static backgrounds.
- `compute_bh_direction_scan.py` (next): compare center->outward vs exterior->inward integration behavior at matched parameters.
- `compute_bh_kerr_slow.py` (next): solve slow-rotation frame-dragging function `omega(r)` on top of static profiles.
- `compute_bh_kerr_observables.py` (next): extract photon-ring and echo-template diagnostics in the slow-rotation window.
- `compute_bh_stress_summary.py` (next): aggregate branch, asymptotic, and slow-rotation diagnostics into one stress-test JSON.

## Mandatory outputs
- `papers/paper-04/data/bh_profiles.npz`
- `papers/paper-04/data/bh_diagnostics.json`
- `papers/paper-04/data/bh_convergence.json`
- `papers/paper-04/data/bh_direction_scan.json`
- `papers/paper-04/data/bh_kerr_slow.npz`
- `papers/paper-04/data/bh_kerr_observables.json`
- `papers/paper-04/data/bh_stress_summary.json`
- `papers/paper-04/figs/bh_metric_profiles.png`
- `papers/paper-04/figs/bh_curvature_regularization.png`
- `papers/paper-04/figs/bh_matching_residuals.png`
- `papers/paper-04/figs/bh_direction_dependence.png`
- `papers/paper-04/figs/bh_kerr_frame_dragging.png`
- `papers/paper-04/figs/bh_kerr_photon_ring.png`
- `papers/paper-04/figs/bh_kerr_echo_template.png`

## Validation gates (must pass before claims)
1. Finite central invariants (`R`, `R_{mu nu}R^{mu nu}`, `K`) on the innermost grid.
2. Stable diagnostics under at least two radial-grid refinements.
3. Explicit large-radius window with small Schwarzschild residuals.
4. One script-to-figure mapping documented in script headers.
5. Slow-rotation outputs restricted to stated `a/M` validity window.

## Immediate TODO (agreed next pass)
1. Implement direction-dependence scan with same benchmark controls.
2. Reuse legacy Kerr/echo scripts from `/Users/cyrille/Perso/Mes projets/TCV/REPO/TCV-Phi/code/` into Paper-04 wrappers.
3. Export a compact JSON table for paper inclusion: `{branch, solver_success, r_H, photon_ring_shift, f_echo}`.
