#!/usr/bin/env python3
from __future__ import annotations

import csv
from _paper_x_common import PAPER_DATA, PAPER_TABLES, load_stage, md_table, write_json, write_text


def main() -> None:
    src = load_stage('stage3_to_stage29_prepaper_assessment.json')
    hierarchy = src['claim_hierarchy']
    rows = []
    csv_rows = []
    for level in ['Level_A', 'Level_B', 'Level_C']:
        for claim in hierarchy[level]:
            rows.append([level.replace('_', ' '), claim, 'allowed'])
            csv_rows.append({'level': level, 'claim': claim, 'status': 'allowed'})
    for claim in src['blocked_claims']:
        rows.append(['Blocked', claim, 'blocked'])
        csv_rows.append({'level': 'Blocked', 'claim': claim, 'status': 'blocked'})

    payload = {
        'status': 'paper_x_claim_hierarchy_export',
        'source': 'stage3_to_stage29_prepaper_assessment.json',
        'row_count': len(rows),
    }
    write_json(PAPER_DATA / 'paper_x_claim_hierarchy.json', payload)
    write_text(PAPER_TABLES / 'claim-hierarchy.md', md_table(['Tier', 'Claim', 'Status'], rows))
    with (PAPER_TABLES / 'claim-hierarchy.csv').open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=['level', 'claim', 'status'])
        writer.writeheader()
        writer.writerows(csv_rows)


if __name__ == '__main__':
    main()
