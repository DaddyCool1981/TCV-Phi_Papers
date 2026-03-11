#!/usr/bin/env python3
"""Install Planck native Cobaya likelihood data into paper-07 local packages path."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    out_data = REPO_ROOT / "papers" / "paper-07" / "data"
    out_data.mkdir(parents=True, exist_ok=True)

    packages = REPO_ROOT / "papers" / "paper-07" / "external_packages"
    packages.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("HOME", str(REPO_ROOT))
    env.setdefault("COBAYA_PACKAGES_PATH", str(packages))

    cmd = [
        "cobaya-install",
        "-p",
        str(packages),
        "--no-set-global",
        "--skip-global",
        "planck_2018_highl_plik.TTTEEE_lite_native",
        "planck_2018_lensing.native",
    ]

    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    status = {
        "command": cmd,
        "returncode": result.returncode,
        "packages_path": str(packages),
        "stdout_tail": "\n".join(result.stdout.splitlines()[-40:]),
        "stderr_tail": "\n".join(result.stderr.splitlines()[-40:]),
    }

    out_json = out_data / "planck_install_status.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)
    print("[INFO] Wrote:", out_json)

    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
