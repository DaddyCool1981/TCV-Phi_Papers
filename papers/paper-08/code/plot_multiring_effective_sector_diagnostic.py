#!/usr/bin/env python3
"""Plot effective-sector closure diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = ROOT / 'papers' / 'paper-08' / 'data' / 'multiring_effective_sector_diagnostic_summary.json'
FIG_DIR = ROOT / 'papers' / 'paper-08' / 'figs'


def _save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'[INFO] Wrote: {path}')


def main() -> None:
    data = json.loads(DATA_PATH.read_text(encoding='utf-8'))
    rows = data['rows']
    n_keep = [r['n_keep'] for r in rows]
    coherent = [r['coherent_rel_err_mean'] for r in rows]
    local = [r['local_rel_err_mean'] for r in rows]
    contrast = [r['contrast_rel_err_mean'] for r in rows]
    twistpair = [r['twistpair_rel_err_mean'] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(n_keep, coherent, marker='o', label='coherent global twist', color='#4c78a8')
    axes[0].plot(n_keep, local, marker='o', label='local loop modes', color='#54a24b')
    axes[0].plot(n_keep, twistpair, marker='o', label='twist-pair modes', color='#f58518')
    axes[0].plot(n_keep, contrast, marker='o', label='contrast modes', color='#e45756')
    axes[0].axhline(0.10, color='black', linestyle='--', linewidth=1)
    axes[0].set_xlabel('n_keep')
    axes[0].set_ylabel('mean relative projection error')
    axes[0].set_title('Closure quality vs retained subspace size')
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)

    heat = np.array([[r['coherent_rel_err_mean'], r['local_rel_err_mean'], r['twistpair_rel_err_mean'], r['contrast_rel_err_mean']] for r in rows])
    im = axes[1].imshow(heat, aspect='auto', cmap='magma_r')
    axes[1].set_xticks(np.arange(4))
    axes[1].set_xticklabels(['coherent', 'local', 'twistpair', 'contrast'])
    axes[1].set_yticks(np.arange(len(rows)))
    axes[1].set_yticklabels([str(x) for x in n_keep])
    axes[1].set_title('Sector-wise closure error heatmap')
    fig.colorbar(im, ax=axes[1])
    fig.tight_layout()
    _save(fig, 'multiring_effective_sector_closure.png')


if __name__ == '__main__':
    main()
