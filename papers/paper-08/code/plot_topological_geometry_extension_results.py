#!/usr/bin/env python3
"""Plot the topological geometry extension scan results."""

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


def _load() -> dict:
    return json.loads((DATA_DIR / "topological_geometry_extension_summary.json").read_text(encoding="utf-8"))


def _save(fig: plt.Figure, name: str) -> None:
    path = FIG_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("[INFO] Wrote:", path)


def plot_scores(summary: dict) -> None:
    rows = summary["all_geometry_rankings"]
    names = [row["geometry"] for row in rows]
    comb = [float(row["combined_pmns_structure_score"]) for row in rows]
    phi2 = [float(row["emergent_phi2_score"]) for row in rows]
    onef = [float(row["one_field_support_score"]) for row in rows]

    x = np.arange(len(names))
    w = 0.26
    fig, ax = plt.subplots(figsize=(11.2, 4.8))
    ax.bar(x - w, comb, width=w, label="PMNS + structure")
    ax.bar(x, phi2, width=w, label="Emergent Phi2")
    ax.bar(x + w, onef, width=w, label="One-field support")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=28, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("score (0-1)")
    ax.set_title("Old + new geometries: PMNS / Phi2 / one-field comparison")
    ax.legend(frameon=False)
    ax.grid(axis="y", ls=":", alpha=0.4)
    _save(fig, "topological_extension_scores.png")


def plot_joint(summary: dict) -> None:
    rows = summary["all_geometry_rankings"]
    names = [row["geometry"] for row in rows]
    vals = [float(row["joint_score"]) for row in rows]
    colors = ["#C44E52" if row["geometry"] == summary["best_overall"]["geometry"] else "#4C72B0" for row in rows]

    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    ax.bar(np.arange(len(names)), vals, color=colors)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=28, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("joint score")
    ax.set_title("Joint PMNS + Phi2 + one-field ranking")
    ax.grid(axis="y", ls=":", alpha=0.4)
    _save(fig, "topological_extension_joint_ranking.png")


def plot_classification(summary: dict) -> None:
    rows = summary["all_geometry_rankings"]
    class_map = {
        "E_strong_one_field_candidate": 5,
        "D_good_on_PMNS_emergent_Phi2_and_partial_micro_closure": 4,
        "C_good_on_both_PMNS_and_emergent_Phi2": 3,
        "B_Phi2_emergence_good_only": 2,
        "A_PMNS_good_only": 1,
        "F_weak_or_disfavored": 0,
    }
    vals = [class_map[row["classification"]] for row in rows]
    names = [row["geometry"] for row in rows]
    labels = ["F", "A", "B", "C", "D", "E"]

    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    ax.bar(np.arange(len(names)), vals, color="#55A868")
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=28, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.5, 5.5)
    ax.set_title("Joint classification across old + new geometries")
    ax.grid(axis="y", ls=":", alpha=0.4)
    _save(fig, "topological_extension_classification.png")


def plot_property_matrix(summary: dict) -> None:
    rows = summary["geometry_property_matrix_extended"]
    props = list(rows[0]["properties"].keys())
    arr = np.array([[float(row["properties"][prop]) for prop in props] for row in rows], dtype=float)
    names = [row["geometry"] for row in rows]

    fig, ax = plt.subplots(figsize=(11.0, 5.4))
    im = ax.imshow(arr, cmap="viridis", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(props)))
    ax.set_xticklabels([p.replace("requires_", "") for p in props], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names)
    ax.set_title("Extended geometry-property matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, "topological_extension_property_matrix.png")


def plot_new_family_ranking(summary: dict) -> None:
    rows = summary["new_family_rankings"]
    names = [row["geometry"] for row in rows]
    vals = [float(row["joint_score"]) for row in rows]

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.bar(np.arange(len(names)), vals, color="#8172B2")
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("joint score")
    ax.set_title("Topological extension families only")
    ax.grid(axis="y", ls=":", alpha=0.4)
    _save(fig, "topological_extension_new_family_ranking.png")


def main() -> None:
    summary = _load()
    plot_scores(summary)
    plot_joint(summary)
    plot_classification(summary)
    plot_property_matrix(summary)
    plot_new_family_ranking(summary)


if __name__ == "__main__":
    main()
