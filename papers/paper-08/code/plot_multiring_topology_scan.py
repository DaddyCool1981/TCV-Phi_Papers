#!/usr/bin/env python3
"""Plot additive topology scan on fixed twisted-multi-ring geometry."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = ROOT / 'papers' / 'paper-08' / 'data' / 'multiring_topology_scan_summary.json'
FIG_DIR = ROOT / 'papers' / 'paper-08' / 'figs'


def _save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'[INFO] Wrote: {path}')


def main() -> None:
    data = json.loads(DATA_PATH.read_text(encoding='utf-8'))
    rows = data['results']
    labels = [r['label'] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].bar(np.arange(len(rows)), [r['flavour_index'] for r in rows], color='#4c78a8')
    axes[0].axhline(data['baseline']['flavour_index'], color='black', linestyle='--', linewidth=1)
    axes[0].set_xticks(np.arange(len(rows)))
    axes[0].set_xticklabels(labels, rotation=20, ha='right')
    axes[0].set_title('Topology ranking by PMNS index')
    axes[0].grid(axis='y', alpha=0.25)

    axes[1].bar(np.arange(len(rows)), [r['bridge_index'] for r in rows], color='#f58518')
    axes[1].axhline(data['baseline']['bridge_index'], color='black', linestyle='--', linewidth=1)
    axes[1].set_xticks(np.arange(len(rows)))
    axes[1].set_xticklabels(labels, rotation=20, ha='right')
    axes[1].set_title('Topology ranking by bridge index')
    axes[1].grid(axis='y', alpha=0.25)

    axes[2].bar(np.arange(len(rows)), [r['overall_index'] for r in rows], color='#54a24b')
    axes[2].set_xticks(np.arange(len(rows)))
    axes[2].set_xticklabels(labels, rotation=20, ha='right')
    axes[2].set_title('Topology ranking by joint index')
    axes[2].grid(axis='y', alpha=0.25)
    fig.tight_layout()
    _save(fig, 'multiring_topology_rankings.png')

    fig, ax = plt.subplots(figsize=(9, 4.8))
    heat = np.array([
        [r['flavour_index'], r['bridge_index'], r.get('micro_closure_index') or 0.0, r['overall_index']]
        for r in rows
    ])
    im = ax.imshow(heat, aspect='auto', cmap='YlGnBu', vmin=0, vmax=max(1.0, np.max(heat)))
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(['PMNS', 'Bridge', 'Closure', 'Overall'])
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(labels)
    ax.set_title('Topology vs PMNS / bridge / closure metrics')
    fig.colorbar(im, ax=ax)
    _save(fig, 'multiring_topology_heatmap.png')

    fig, ax = plt.subplots(figsize=(9, 4.8))
    closure_vals = [r.get('micro_closure_index') if r.get('micro_closure_index') is not None else 0.0 for r in rows]
    scatter = ax.scatter([r['flavour_index'] for r in rows], [r['bridge_index'] for r in rows], c=closure_vals, cmap='plasma', s=90)
    for i, r in enumerate(rows):
        ax.text(r['flavour_index'] + 0.004, r['bridge_index'] + 0.004, str(i + 1), fontsize=8)
    ax.set_xlabel('flavour index')
    ax.set_ylabel('bridge index')
    ax.set_title('Topology descriptors vs micro-closure status')
    ax.grid(alpha=0.25)
    fig.colorbar(scatter, ax=ax, label='micro-closure index')
    _save(fig, 'multiring_topology_closure_scatter.png')


if __name__ == '__main__':
    main()
