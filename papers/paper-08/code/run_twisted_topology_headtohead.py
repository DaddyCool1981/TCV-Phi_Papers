#!/usr/bin/env python3
"""Consolidated head-to-head comparison: twisted_torus vs twisted_multi_ring."""

from __future__ import annotations

import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
mpl_dir = REPO_ROOT / ".tmp_mpl"
mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DATA_DIR = REPO_ROOT / "papers" / "paper-08" / "data"
FIG_DIR = REPO_ROOT / "papers" / "paper-08" / "figs"
P5_DATA = REPO_ROOT / "papers" / "paper-05" / "data"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _pmns_score(angles: dict) -> float:
    d12 = (float(angles["theta12"]) - 33.0) / 10.0
    d13 = (float(angles["theta13"]) - 8.6) / 4.0
    d23 = (float(angles["theta23"]) - 45.0) / 7.0
    return float(np.sqrt(d12 * d12 + d13 * d13 + d23 * d23))


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("[INFO] Wrote:", path)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    tw_fl = _load(P5_DATA / "twisted_torus_flavour_toy_summary.json")
    mr_fl = _load(P5_DATA / "twisted_multi_ring_flavour_toy_summary.json")
    tw_br_v21 = _load(DATA_DIR / "phi1_phi2_emergence_v21_summary.json")
    tw_br_v29 = _load(DATA_DIR / "phi1_phi2_emergence_v29_readiness_update.json")
    mr_br = _load(DATA_DIR / "twisted_multi_ring_phi2_bridge_summary.json")

    tw_best_fl = tw_fl["best_candidates"][0]
    mr_best_fl = mr_fl["best_candidates"][0]
    tw_best_br = tw_br_v21["best_record"]

    tw = {
        "geometry": "twisted_torus",
        "pmns": {
            "angles_deg": tw_best_fl["angles_deg"],
            "score_common": _pmns_score(tw_best_fl["angles_deg"]),
            "strict_fraction_within_natural": float(tw_fl["scan_summary"]["fractions"]["strict_within_natural"]),
        },
        "bridge": {
            "joint_fraction": float(tw_best_br["twisted_joint_mean"]),
            "robustness_pass_fraction": float(tw_best_br["twisted_robust_mean"]),
            "m2_eV": float(tw_best_br["twisted_m2_mean_eV"]),
            "m2_log10_dev": float(tw_best_br["criteria"]["m2_log10_dev"]),
            "micro_status": "structured_but_open",
            "micro_projection_good": True,
        },
    }
    mr = {
        "geometry": "twisted_multi_ring",
        "pmns": {
            "angles_deg": mr_best_fl["angles_deg"],
            "score_common": _pmns_score(mr_best_fl["angles_deg"]),
            "strict_fraction_within_natural": float(mr_fl["scan_summary"]["fractions"]["strict_within_natural"]),
        },
        "bridge": {
            "joint_fraction": float(mr_br["emergence"]["joint_fraction"]),
            "robustness_pass_fraction": float(mr_br["emergence"]["robustness_pass_fraction"]),
            "m2_eV": float(mr_br["calibrated_m2_eV"]),
            "m2_log10_dev": float(mr_br["m2_log10_dev_from_1e-22"]),
            "micro_status": "good_proxy",
            "micro_projection_good": bool(mr_br["checks"]["micro_projection_good"]),
        },
    }

    # Simple normalized dashboard.
    def flavour_index(block: dict) -> float:
        score_q = float(np.clip((2.7 - block["pmns"]["score_common"]) / (2.7 - 0.235), 0.0, 1.0))
        strict_q = float(np.clip(block["pmns"]["strict_fraction_within_natural"] / 0.05, 0.0, 1.0))
        return float(0.7 * score_q + 0.3 * strict_q)

    def bridge_index(block: dict) -> float:
        joint_q = float(np.clip(block["bridge"]["joint_fraction"], 0.0, 1.0))
        robust_q = float(np.clip(block["bridge"]["robustness_pass_fraction"], 0.0, 1.0))
        m2_q = float(np.clip(1.0 - block["bridge"]["m2_log10_dev"] / 0.5, 0.0, 1.0))
        micro_q = 1.0 if block["bridge"]["micro_projection_good"] else 0.0
        return float(0.35 * joint_q + 0.25 * robust_q + 0.20 * m2_q + 0.20 * micro_q)

    tw["flavour_index"] = flavour_index(tw)
    mr["flavour_index"] = flavour_index(mr)
    tw["bridge_index"] = bridge_index(tw)
    mr["bridge_index"] = bridge_index(mr)
    tw["overall_index"] = float(0.5 * tw["flavour_index"] + 0.5 * tw["bridge_index"])
    mr["overall_index"] = float(0.5 * mr["flavour_index"] + 0.5 * mr["bridge_index"])

    winner = mr if mr["overall_index"] > tw["overall_index"] else tw
    summary = {
        "status": "consolidated twisted-topology head-to-head",
        "twisted_torus": tw,
        "twisted_multi_ring": mr,
        "winner": {
            "geometry": winner["geometry"],
            "reason": "higher combined flavour + bridge index",
            "overall_index": winner["overall_index"],
        },
        "notes": [
            "The twisted_torus bridge uses the established v21/v29 in-repo diagnostics.",
            "The twisted_multi_ring bridge is the new dedicated additive bridge.",
            "This comparison is the current best internal decision layer for choosing the lead twisted topology.",
        ],
    }
    out = DATA_DIR / "twisted_topology_headtohead_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[INFO] Wrote:", out)

    names = ["twisted_torus", "twisted_multi_ring"]
    flavour = [tw["flavour_index"], mr["flavour_index"]]
    bridge = [tw["bridge_index"], mr["bridge_index"]]
    overall = [tw["overall_index"], mr["overall_index"]]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(len(names))
    w = 0.24
    ax.bar(x - w, flavour, width=w, label="Flavour")
    ax.bar(x, bridge, width=w, label="Bridge")
    ax.bar(x + w, overall, width=w, label="Overall")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("index (0-1)")
    ax.set_title("Twisted topology head-to-head")
    ax.grid(axis="y", ls=":", alpha=0.4)
    ax.legend(frameon=False)
    _save(fig, FIG_DIR / "twisted_topology_headtohead.png")


if __name__ == "__main__":
    main()
