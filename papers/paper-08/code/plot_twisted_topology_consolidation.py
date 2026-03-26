#!/usr/bin/env python3
"""Plot a compact consolidated dashboard for twisted_torus vs twisted_multi_ring."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = ROOT / "papers" / "paper-08" / "data" / "twisted_topology_consolidated_summary.json"
FIG_PATH = ROOT / "papers" / "paper-08" / "figs" / "twisted_topology_consolidated_dashboard.png"


def main() -> None:
    data = json.load(DATA_PATH.open("r", encoding="utf-8"))

    torus = data["twisted_torus"]
    multi = data["twisted_multi_ring"]

    labels = ["flavour idx", "bridge idx", "overall idx"]
    torus_scores = [
        torus["indices"]["flavour_index"],
        torus["indices"]["bridge_index"],
        torus["indices"]["overall_index"],
    ]
    multi_scores = [
        multi["indices"]["flavour_index"],
        multi["indices"]["bridge_index"],
        multi["indices"]["overall_index"],
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    x = np.arange(len(labels))
    width = 0.34
    axes[0].bar(x - width / 2, torus_scores, width=width, label="twisted_torus", color="#5b7c99")
    axes[0].bar(x + width / 2, multi_scores, width=width, label="twisted_multi_ring", color="#c96f3b")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=10)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("normalized score")
    axes[0].set_title("Consolidated indices")
    axes[0].legend(frameon=False)

    metric_labels = ["strict PMNS", "bridge robust", "mode13", "micro pass count / 20"]
    torus_metrics = [
        torus["pmns"]["strict_fraction_within_natural"],
        torus["bridge"]["robustness_pass_fraction"],
        0.0,
        0.0,
    ]
    multi_metrics = [
        multi["pmns"]["strict_fraction_within_natural"],
        multi["full_bridge"]["robustness_pass_fraction"],
        multi["full_bridge"]["mode13_weight_mean"],
        multi["micro_closure"]["n_all_pass"] / 20.0,
    ]
    x2 = np.arange(len(metric_labels))
    axes[1].bar(x2 - width / 2, torus_metrics, width=width, label="twisted_torus", color="#5b7c99")
    axes[1].bar(x2 + width / 2, multi_metrics, width=width, label="twisted_multi_ring", color="#c96f3b")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(metric_labels, rotation=18, ha="right")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_title("Operational metrics")

    fig.suptitle("Twisted topology consolidation")
    fig.tight_layout()

    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=180, bbox_inches="tight")
    print(f"[INFO] Wrote: {FIG_PATH}")


if __name__ == "__main__":
    main()
