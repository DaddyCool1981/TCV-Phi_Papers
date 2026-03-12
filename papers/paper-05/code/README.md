# Paper 05 Code Plan

## Canonical inputs
- `latex/6_Cohesive Microphysics and Scalar Perturbations in the TCV.tex`
- `latex/9_Cohesive_Quantum_Gravity_CCphi1.tex`
- `latex/10_From Quantum CC-Phi2 to Flavour.tex`

## Reused code candidates
- `code/paper10_phi2_cluster.py`
- `code/paper10_quantum_phi2_leptons.py`
- `TCV-PHI/core/lib/tcvphi/phi2_from_phi1_complex.py`

## Target scripts (next pass)
1. `compute_ccphi1_spectrum.py`
2. `compute_ccphi1_curvature_response.py`
3. `compute_ccphi1_coarse_grain.py`
4. `compute_ccphi2_flavour_toy.py` (explicitly exploratory)
5. `compute_qg_consistency_summary.py`
6. `compute_tetrahedral_flavor_toy.py` (exploratory tetrahedral geometry test)
7. `plot_tetrahedral_scan.py` (optional plotting helper)
8. `compute_tetrahedral_targeted_scan.py` (targeted geometry+response scan)
9. `compute_tetrahedral_star_toy.py` (exploratory tetrahedral-star toy)
10. `plot_tetrahedral_star_scan.py` (optional tetrahedral-star plotting helper)
11. `compute_torus_flavor_toy.py` (exploratory torus-ring toy)
12. `compute_flavour_geometry_comparison.py` (cross-geometry comparison)
13. `compute_spherical_flavor_toy.py` (exploratory spherical-shell toy)
14. `compute_twisted_torus_flavor_toy.py` (exploratory twisted torus / Mobius toy)
15. `compute_twisted_torus_sector_compatibility.py` (charged-lepton + quark-like compatibility check)

## Mandatory outputs
- `papers/paper-05/data/ccphi1_spectrum.json`
- `papers/paper-05/data/ccphi1_curvature_scan.npz`
- `papers/paper-05/data/ccphi1_coarse_grain.json`
- `papers/paper-05/data/ccphi2_flavour_toy.json`
- `papers/paper-05/data/paper05_consistency_summary.json`
- `papers/paper-05/data/tetrahedral_flavour_toy_summary.json`
- `papers/paper-05/data/tetrahedral_flavour_scan.npz`
- `papers/paper-05/data/tetrahedral_targeted_scan_summary.json`
- `papers/paper-05/data/tetrahedral_star_toy_summary.json`
- `papers/paper-05/data/tetrahedral_star_scan.npz`
- `papers/paper-05/data/tetrahedral_star_targeted_scan_summary.json`
- `papers/paper-05/data/tetrahedral_star_targeted_scan_pheno_v2_summary.json`
- `papers/paper-05/data/torus_flavour_toy_summary.json`
- `papers/paper-05/data/flavour_geometry_comparison_summary.json`
- `papers/paper-05/data/spherical_flavour_toy_summary.json`
- `papers/paper-05/data/twisted_torus_flavour_toy_summary.json`
- `papers/paper-05/data/twisted_torus_sector_compatibility_summary.json`
