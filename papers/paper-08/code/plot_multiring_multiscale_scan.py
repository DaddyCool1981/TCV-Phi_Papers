#!/usr/bin/env python3
"""Plot multiscale/coarse-graining scan."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = ROOT / 'papers' / 'paper-08' / 'data' / 'multiring_multiscale_scan_summary.json'
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
    x = np.arange(len(rows))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(x - 0.18, [r['flavour_index'] for r in rows], width=0.18, label='PMNS', color='#4c78a8')
    axes[0].bar(x, [r['bridge_index'] for r in rows], width=0.18, label='Bridge', color='#f58518')
    axes[0].bar(x + 0.18, [r.get('micro_closure_index') or 0.0 for r in rows], width=0.18, label='Closure', color='#54a24b')
    axes[0].axhline(data['baseline']['flavour_index'], color='#4c78a8', linestyle='--', linewidth=1)
    axes[0].axhline(data['baseline']['bridge_index'], color='#f58518', linestyle='--', linewidth=1)
    axes[0].axhline(data['baseline']['micro_closure_index'], color='#54a24b', linestyle='--', linewidth=1)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha='right')
    axes[0].set_title('Multiscale metrics vs baseline')
    axes[0].grid(axis='y', alpha=0.25)
    axes[0].legend(frameon=False)

    heat = np.array([[r['flavour_index'], r['bridge_index'], r.get('micro_closure_index') or 0.0, r['overall_index']] for r in rows])
    im = axes[1].imshow(heat, aspect='auto', cmap='YlGnBu', vmin=0, vmax=max(1.0, np.max(heat)))
    axes[1].set_xticks(np.arange(4))
    axes[1].set_xticklabels(['PMNS', 'Bridge', 'Closure', 'Overall'])
    axes[1].set_yticks(np.arange(len(rows)))
    axes[1].set_yticklabels(labels)
    axes[1].set_title('Multiscale heatmap')
    fig.colorbar(im, ax=axes[1])
    fig.tight_layout()
    _save(fig, 'multiring_multiscale_summary.png')


if __name__ == '__main__':
    main()
