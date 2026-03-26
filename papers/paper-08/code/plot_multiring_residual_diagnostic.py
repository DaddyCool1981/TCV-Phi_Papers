#!/usr/bin/env python3
"""Plot residual non-closure diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = ROOT / 'papers' / 'paper-08' / 'data' / 'multiring_residual_diagnostic_summary.json'
FIG_DIR = ROOT / 'papers' / 'paper-08' / 'figs'


def _save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'[INFO] Wrote: {path}')


def main() -> None:
    data = json.loads(DATA_PATH.read_text(encoding='utf-8'))
    lambdas = data['residual_summary']['first_omitted_lambdas']
    cumul = data['residual_summary']['omitted_cumulative_fraction_first8']
    blocks = np.array(data['ring_block_residual'])
    best = data['source_overlap_summary']['best_low3_sources']
    worst = data['source_overlap_summary']['worst_tail_sources']

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(np.arange(len(lambdas)), lambdas, color='#4c78a8')
    axes[0].set_xlabel('omitted mode offset')
    axes[0].set_ylabel(r'$\lambda_n$')
    axes[0].set_title('First omitted generalized eigenvalues')
    axes[0].grid(axis='y', alpha=0.25)

    axes[1].plot(np.arange(1, len(cumul) + 1), cumul, marker='o', color='#f58518')
    axes[1].set_xlabel('number of omitted modes included')
    axes[1].set_ylabel('cumulative omitted lambda fraction')
    axes[1].set_title('Omitted spectrum accumulation')
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, 'multiring_residual_spectrum.png')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    im = axes[0].imshow(blocks, cmap='magma')
    axes[0].set_xticks(np.arange(blocks.shape[1]))
    axes[0].set_yticks(np.arange(blocks.shape[0]))
    axes[0].set_title('Residual block norm by ring pair')
    fig.colorbar(im, ax=axes[0])

    names = [name for name, _ in best[:4]] + [name for name, _ in worst[:4]]
    low3 = [entry['frac_low3'] for _, entry in best[:4]] + [entry['frac_low3'] for _, entry in worst[:4]]
    tail = [entry['frac_tail'] for _, entry in best[:4]] + [entry['frac_tail'] for _, entry in worst[:4]]
    x = np.arange(len(names))
    axes[1].bar(x - 0.16, low3, width=0.32, label='low3', color='#54a24b')
    axes[1].bar(x + 0.16, tail, width=0.32, label='tail', color='#e45756')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=25, ha='right')
    axes[1].set_ylim(0, 1)
    axes[1].set_title('Source overlap: low subspace vs tail')
    axes[1].legend(frameon=False)
    axes[1].grid(axis='y', alpha=0.25)
    fig.tight_layout()
    _save(fig, 'multiring_residual_blocks_and_sources.png')


if __name__ == '__main__':
    main()
