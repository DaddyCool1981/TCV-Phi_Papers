from __future__ import annotations

from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
SM_FIGS = ROOT / "papers" / "sm-exploration" / "figs"
OUT = ROOT / "papers" / "paper-09" / "figs"

PROMOTED = {
    "collective_particle_patch_scales_and_mass.png": "paper-09-fig-01-mesoscopic-scale-hierarchy.png",
    "qsp_interpretation_results.png": "paper-09-fig-02-qsp-identity-diagnostics.png",
    "qsp_collision_results.png": "paper-09-fig-03-collision-channels.png",
    "qsp_attractor_reduction_results.png": "paper-09-fig-04-attractor-reduction.png",
    "qsp_basis_sector_results.png": "paper-09-fig-05-sector-projection.png",
    "qsp_effective_family_reading_results.png": "paper-09-fig-06-effective-family-reading.png",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for src_name, dst_name in PROMOTED.items():
        src = SM_FIGS / src_name
        dst = OUT / dst_name
        shutil.copy2(src, dst)
        print(f"promoted {src.name} -> {dst.name}")


if __name__ == "__main__":
    main()
