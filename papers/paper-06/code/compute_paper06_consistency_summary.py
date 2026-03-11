#!/usr/bin/env python3
"""Merge Paper-06 artifacts into one consistency summary."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load(path: Path):
    if not path.exists():
        return {"missing": True, "path": str(path)}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    out_data = REPO_ROOT / "papers" / "paper-06" / "data"
    out_data.mkdir(parents=True, exist_ok=True)

    comparison = _load(out_data / "series_comparison.json")
    falsification = _load(out_data / "falsification_table.json")

    summary = {
        "paper": "paper-06",
        "scope": "critical comparison and falsification roadmap",
        "comparison": comparison,
        "falsification_table": falsification,
        "note": "This file tracks explicit exclusion logic and source-linked diagnostics.",
    }

    out_json = out_data / "paper06_consistency_summary.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("[INFO] Wrote:", out_json)


if __name__ == "__main__":
    main()
