#!/usr/bin/env python3
"""Plot diagnostics for the PMNS inverse/direct geometry workflow."""

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


def _load_summary() -> dict:
    path = DATA_DIR / "pmns_inverse_geometry_scan_summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("[INFO] Wrote:", path)


def plot_constraints(summary: dict) -> None:
    rows = summary["constraint_profile"]["constraint_table"]
    names = [row["constraint"].replace("requires_", "") for row in rows]
    vals = [float(row["weight"]) for row in rows]

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    y = np.arange(len(names))
    ax.barh(y, vals, color="#4C72B0")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("inverse constraint weight")
    ax.set_title("PMNS-inferred structural constraints")
    ax.grid(axis="x", ls=":", alpha=0.4)
    _save(fig, FIG_DIR / "pmns_inverse_constraints.png")


def plot_geometry_matrix(summary: dict) -> None:
    matrix = summary["geometry_property_matrix"]
    props = list(matrix[0]["properties"].keys())
    geoms = [row["geometry"] for row in matrix]
    arr = np.array([[float(row["properties"][prop]) for prop in props] for row in matrix], dtype=float)

    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    im = ax.imshow(arr, cmap="viridis", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(props)))
    ax.set_xticklabels([p.replace("requires_", "") for p in props], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(geoms)))
    ax.set_yticklabels(geoms)
    ax.set_title("Geometry-property matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="property strength")
    _save(fig, FIG_DIR / "pmns_geometry_property_matrix.png")


def plot_scores(summary: dict) -> None:
    rows = summary["family_rankings"]
    names = [row["geometry"] for row in rows]
    pmns = [float(row["pmns_viability_index"]) for row in rows]
    structural = [float(row["structural_score"]) for row in rows]
    combined = [float(row["combined_pmns_structure_score"]) for row in rows]

    x = np.arange(len(names))
    w = 0.25
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    ax.bar(x - w, pmns, width=w, label="PMNS")
    ax.bar(x, structural, width=w, label="Structure")
    ax.bar(x + w, combined, width=w, label="Combined")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("score (0-1)")
    ax.set_title("PMNS / structural / combined scores")
    ax.legend(frameon=False)
    ax.grid(axis="y", ls=":", alpha=0.4)
    _save(fig, FIG_DIR / "pmns_inverse_scores.png")


def plot_bridge(summary: dict) -> None:
    rows = summary["joint_rankings"]
    names = [row["geometry"] for row in rows]
    phi2 = [float(row["emergent_phi2_score"]) for row in rows]
    one_field = [float(row["one_field_support_score"]) for row in rows]
    joint = [float(row["joint_score"]) for row in rows]

    x = np.arange(len(names))
    w = 0.25
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    ax.bar(x - w, phi2, width=w, label="Emergent Phi2")
    ax.bar(x, one_field, width=w, label="One-field support")
    ax.bar(x + w, joint, width=w, label="Joint score")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("score (0-1)")
    ax.set_title("Phi2 bridge and joint ranking")
    ax.legend(frameon=False)
    ax.grid(axis="y", ls=":", alpha=0.4)
    _save(fig, FIG_DIR / "pmns_phi2_bridge_scores.png")


def plot_classification(summary: dict) -> None:
    rows = summary["joint_rankings"]
    class_map = {
        "E_strong_one_field_candidate": 5,
        "D_good_on_PMNS_emergent_Phi2_and_partial_micro_closure": 4,
        "C_good_on_both_PMNS_and_emergent_Phi2": 3,
        "B_Phi2_emergence_good_only": 2,
        "A_PMNS_good_only": 1,
        "F_weak_or_disfavored": 0,
    }
    names = [row["geometry"] for row in rows]
    vals = [class_map[row["classification"]] for row in rows]
    labels = ["F", "A", "B", "C", "D", "E"]

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.bar(np.arange(len(names)), vals, color="#C44E52")
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.5, 5.5)
    ax.set_ylabel("classification band")
    ax.set_title("Joint PMNS + Phi2 + micro-closure classification")
    ax.grid(axis="y", ls=":", alpha=0.4)
    _save(fig, FIG_DIR / "pmns_joint_classification.png")


def main() -> None:
    summary = _load_summary()
    plot_constraints(summary)
    plot_geometry_matrix(summary)
    plot_scores(summary)
    plot_bridge(summary)
    plot_classification(summary)


if __name__ == "__main__":
    main()
