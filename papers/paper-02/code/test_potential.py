
import os
import sys

# Ajoute le répertoire parent (TCV-PHI) au PYTHONPATH quand on lance depuis code/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tcvphi.params import make_default_benchmark
from tcvphi.potential import CohesivePotential

bench = make_default_benchmark()
pot = CohesivePotential(bench.micro)

phi1_eq = bench.micro.phi0
phi2_eq = 0.0

print("V_tot(min) =", pot.V_tot(phi1_eq, phi2_eq))
print("m_eff^2 at minimum =", pot.mass_matrix_at_minimum())
