#!/usr/bin/env python3
from __future__ import annotations

import csv
from _paper_x_common import PAPER_DATA, PAPER_TABLES, load_stage, md_table, write_json, write_text


def main() -> None:
    stage25 = load_stage('stage25_full_sm_derivation_assessment.json')
    stage26 = load_stage('stage26_final_assessment.json')
    stage27 = load_stage('stage27_hard_prediction_assessment.json')
    stage28 = load_stage('stage28_generation_route_assessment.json')
    stage29 = load_stage('stage29_unification_route_assessment.json')

    rows = [
        ['Stage 25', 'Windowed SM crossing', stage25['allowed_claim'], 'window-tested'],
        ['Stage 26', 'Anomaly dynamics', stage26['allowed_claim'], 'guarded theorem-style'],
        ['Stage 27', 'Hard predictions', 'Lambda_TCV, m_Phi2, and conditional EW anchor', 'candidate'],
        ['Stage 28', 'Generation routes', 'Three-generation, CKM-like, and neutrino-spectrum routes', 'candidate'],
        ['Stage 29', 'Unification route', 'Gauge-coupling meeting route and conditional 3/8 path', 'candidate'],
    ]
    payload = {
        'status': 'paper_x_result_summary_export',
        'sources': [
            'stage25_full_sm_derivation_assessment.json',
            'stage26_final_assessment.json',
            'stage27_hard_prediction_assessment.json',
            'stage28_generation_route_assessment.json',
            'stage29_unification_route_assessment.json',
        ],
        'row_count': len(rows),
    }
    write_json(PAPER_DATA / 'paper_x_result_summary.json', payload)
    write_text(PAPER_TABLES / 'result-summary.md', md_table(['Stage', 'Package', 'Claim-safe summary', 'Status'], rows))
    with (PAPER_TABLES / 'result-summary.csv').open('w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['stage', 'package', 'claim_safe_summary', 'status'])
        writer.writerows(rows)


if __name__ == '__main__':
    main()
