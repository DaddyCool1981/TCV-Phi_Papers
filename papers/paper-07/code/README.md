# Paper 07 code

## Scripts
- `run_baseline_planck_check.py`
  - Validates baseline CLASS bridge consistency before likelihood runs.
  - Produces one JSON summary and one P(k) comparison figure.

- `write_planck_ladder_notes.py`
  - Writes a machine-readable checklist for the Planck fit ladder.

- `prepare_lcdm_planck_input.py`
  - Writes a Cobaya LCDM+Planck input YAML in `papers/paper-07/data/`.
  - By default includes low-$\ell$ likelihoods (`lowl.TT`, `lowl.EE`).

- `install_planck_native_data.py`
  - Installs Planck native likelihood data into `papers/paper-07/external_packages`.

- `run_lcdm_planck_fit.py`
  - Runs Cobaya with the generated YAML and writes a status JSON, including
    explicit failure diagnostics when external Planck likelihood data is missing.

- `prepare_tcv_minimal_planck_input.py`
  - Writes a Cobaya minimal-TCV input YAML with fixed primordial pair
    $(A_s,n_s)$ imported from `papers/paper-03/data/primordial_observables.json`.

- `run_tcv_minimal_planck_fit.py`
  - Runs Cobaya for the minimal-TCV setup and writes a dedicated status JSON.

- `compare_lcdm_tcv_models.py`
  - Builds a JSON comparison report (`\Delta\chi^2`, AIC, BIC proxy) from
  converged LCDM and minimal-TCV chains.

- `compute_model_robustness_summary.py`
  - Builds a lightweight robustness JSON (BIC-sign scan vs `N_data` proxy and
    final convergence rows from both chain progress files).

## Usage

```bash
source .venv/bin/activate
python papers/paper-07/code/run_baseline_planck_check.py
python papers/paper-07/code/write_planck_ladder_notes.py
python papers/paper-07/code/prepare_lcdm_planck_input.py
python papers/paper-07/code/install_planck_native_data.py
python papers/paper-07/code/run_lcdm_planck_fit.py --dry-run
python papers/paper-07/code/prepare_tcv_minimal_planck_input.py
python papers/paper-07/code/run_tcv_minimal_planck_fit.py --dry-run
python papers/paper-07/code/compare_lcdm_tcv_models.py
python papers/paper-07/code/compute_model_robustness_summary.py
```
