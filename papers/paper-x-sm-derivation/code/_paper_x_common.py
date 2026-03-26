from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STAGE_DATA = ROOT / 'papers' / 'sm-stage3-exploration' / 'data'
PAPER_ROOT = ROOT / 'papers' / 'paper-x-sm-derivation'
PAPER_DATA = PAPER_ROOT / 'data'
PAPER_TABLES = PAPER_ROOT / 'tables'


def load_stage(name: str) -> dict:
    return json.loads((STAGE_DATA / name).read_text(encoding='utf-8'))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding='utf-8')


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join(['---'] * len(headers)) + ' |',
    ]
    for row in rows:
        lines.append('| ' + ' | '.join(row) + ' |')
    return '\n'.join(lines) + '\n'
