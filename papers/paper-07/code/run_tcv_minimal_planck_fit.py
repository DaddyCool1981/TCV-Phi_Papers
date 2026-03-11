#!/usr/bin/env python3
"""Run Cobaya for the Paper-07 minimal TCV Planck setup."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]


def _write_status(payload: dict) -> None:
    out = REPO_ROOT / "papers" / "paper-07" / "data" / "tcv_minimal_planck_run_status.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("[INFO] Wrote:", out)


def _read_last_progress_row(prefix: Path) -> dict | None:
    progress = prefix.with_suffix(".progress")
    if not progress.exists():
        return None
    last = None
    for line in progress.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 5:
            continue
        last = {
            "N": float(parts[0]),
            "timestamp": parts[1],
            "acceptance_rate": float(parts[2]),
            "Rminus1": float(parts[3]),
            "Rminus1_cl": None if parts[4] == "NaN" else float(parts[4]),
        }
    return last


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Validate input/environment only.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing chain products.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing chain products.")
    parser.add_argument(
        "--allow-changes",
        action="store_true",
        help="Allow compatible changes when resuming (passed through to cobaya.run).",
    )
    args = parser.parse_args()

    data_dir = REPO_ROOT / "papers" / "paper-07" / "data"
    input_yaml = data_dir / "tcv_minimal_planck_input.yaml"
    if not input_yaml.exists():
        raise FileNotFoundError(f"Missing input YAML: {input_yaml}")

    info = yaml.safe_load(input_yaml.read_text(encoding="utf-8"))

    os.environ.setdefault("HOME", str(REPO_ROOT))
    os.environ.setdefault("COBAYA_PACKAGES_PATH", str(REPO_ROOT / "papers" / "paper-07" / "external_packages"))

    if args.dry_run:
        _write_status(
            {
                "dry_run": True,
                "input_yaml": str(input_yaml),
                "packages_path": os.environ.get("COBAYA_PACKAGES_PATH"),
                "status": "validated_input_only",
            }
        )
        return

    try:
        from cobaya.run import run

        updated_info, sampler = run(
            info,
            resume=args.resume,
            force=args.force,
            allow_changes=args.allow_changes,
        )
        output_root = Path(str(info.get("output", "")))
        _write_status(
            {
                "dry_run": False,
                "status": "completed",
                "packages_path": os.environ.get("COBAYA_PACKAGES_PATH"),
                "output_root": str(output_root),
                "resume": args.resume,
                "force": args.force,
                "allow_changes": args.allow_changes,
                "sampler": str(type(sampler)),
                "updated_info_keys": sorted(list(updated_info.keys())),
                "convergence_last_progress_row": _read_last_progress_row(output_root),
            }
        )
    except Exception as exc:
        _write_status(
            {
                "dry_run": False,
                "status": "failed",
                "error": str(exc),
                "packages_path": os.environ.get("COBAYA_PACKAGES_PATH"),
            }
        )
        raise


if __name__ == "__main__":
    main()
