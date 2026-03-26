#!/usr/bin/env python3
from __future__ import annotations

import csv
from _paper_x_common import PAPER_DATA, PAPER_TABLES, load_stage, md_table, write_json, write_text


def main() -> None:
    stage27 = load_stage('stage27_hard_prediction_assessment.json')
    stage28 = load_stage('stage28_generation_route_assessment.json')
    stage29 = load_stage('stage29_unification_route_assessment.json')

    rows = [
        ['Lambda_TCV', f"{stage27['predictions']['Lambda_TCV_GeV']:.6e} GeV", 'candidate calibration', 'benchmark-dependent'],
        ['m_Phi2', f"{stage27['predictions']['m_Phi2_eV']:.3e} eV", 'candidate prediction', 'anchor-dependent normalization'],
        ['sin^2(theta_W)', f"{stage27['predictions']['sin2_theta_w_at_Lambda_TCV']:.3f} +/- {stage27['predictions']['sin2_theta_w_error']:.3f}", 'external matching anchor', 'not internally derived'],
        ['Generation family', 'muon/strange anchor', 'candidate route', 'not theorem-level'],
        ['CKM-like route', f"hierarchy margin = {stage28['headline_structure']['triplet_hierarchy_margin']:.6f}", 'candidate route', 'not precision CKM'],
        ['Neutrino route', '2nd-gen neutrino-like', 'candidate route', 'not quantitative spectrum'],
        ['Unification route', str(stage29['headline_structure']['high_scale_target_window']), 'candidate route', 'no RG closure'],
    ]
    payload = {
        'status': 'paper_x_prediction_route_export',
        'sources': [
            'stage27_hard_prediction_assessment.json',
            'stage28_generation_route_assessment.json',
            'stage29_unification_route_assessment.json',
        ],
        'row_count': len(rows),
    }
    write_json(PAPER_DATA / 'paper_x_prediction_routes.json', payload)
    write_text(PAPER_TABLES / 'prediction-routes.md', md_table(['Item', 'Value', 'Category', 'Boundary'], rows))
    with (PAPER_TABLES / 'prediction-routes.csv').open('w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['item', 'value', 'category', 'boundary'])
        writer.writerows(rows)


if __name__ == '__main__':
    main()
