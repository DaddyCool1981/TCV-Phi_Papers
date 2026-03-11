#!/usr/bin/env python3
"""Run (or dry-run) Cobaya LCDM+Planck baseline for Paper 07."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]


def _write_status(payload: dict) -> None:
    out = REPO_ROOT / "papers" / "paper-07" / "data" / "lcdm_planck_run_status.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print("[INFO] Wrote:", out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Only validate input and environment.")
    args = parser.parse_args()

    data_dir = REPO_ROOT / "papers" / "paper-07" / "data"
    input_yaml = data_dir / "lcdm_planck_input.yaml"
    if not input_yaml.exists():
        raise FileNotFoundError(f"Missing input YAML: {input_yaml}")

    with input_yaml.open("r", encoding="utf-8") as f:
        info = yaml.safe_load(f)

    # Cobaya on macOS defaults to ~/Library/Application Support/cobaya.
    # In this repo workflow we force HOME to a writable workspace path.
    os.environ.setdefault("HOME", str(REPO_ROOT))
    os.environ.setdefault("COBAYA_PACKAGES_PATH", str(REPO_ROOT / "papers" / "paper-07" / "external_packages"))

    if args.dry_run:
        _write_status(
            {
                "dry_run": True,
                "input_yaml": str(input_yaml),
                "packages_path": os.environ.get("COBAYA_PACKAGES_PATH"),
                "status": "validated_input_only",
                "next": "Run without --dry-run once Planck likelihood data is installed.",
            }
        )
        return

    try:
        from cobaya.run import run

        updated_info, sampler = run(info)
        _write_status(
            {
                "dry_run": False,
                "status": "completed",
                "packages_path": os.environ.get("COBAYA_PACKAGES_PATH"),
                "sampler": str(type(sampler)),
                "updated_info_keys": sorted(list(updated_info.keys())),
            }
        )
    except Exception as exc:
        _write_status(
            {
                "dry_run": False,
                "status": "failed",
                "error": str(exc),
                "packages_path": os.environ.get("COBAYA_PACKAGES_PATH"),
                "hint": (
                    "Install Planck native likelihood data under papers/paper-07/external_packages "
                    "and rerun."
                ),
            }
        )
        raise


if __name__ == "__main__":
    main()
