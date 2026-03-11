# TCV-Φ Papers (Public Series)

This repository hosts the public paper series of the **TCV-Φ (Théorie du Vide Cohésif)** program.

## Scope

The goal of this repository is to provide:
- journal-style manuscripts (PRD/JCAP format),
- reproducible figures and key numerical outputs,
- a transparent progression from EFT setup to data confrontation.

This is a **research program** with explicit status labelling:
- derived EFT results,
- phenomenological ansätze,
- deferred or exploratory components.

---

## Paper Series

- **TCV-Φ I**: EFT definition, late-time \(\Lambda\)CDM-like background, ULDM-inspired sector.
- **TCV-Φ II**: cohesive bounce/reheating/baryogenesis setup and computational backbone.
- **TCV-Φ III**: primordial scalar perturbations and CLASS bridge workflow.
- **TCV-Φ IV**: cohesive non-singular compact-object sector (static + slow-rotation templates).
- **TCV-Φ V**: cohesive microstructure and exploratory quantum-sector proxies.
- **TCV-Φ VI**: cross-paper consistency, falsification windows, risk map.
- **TCV-Φ VII**: first Planck-likelihood confrontation (\(\Lambda\)CDM baseline + minimal TCV comparison).

---

## Repository Layout

```text
papers/
  paper-01/
  paper-02/
  paper-03/
  paper-04/
  paper-05/
  paper-06/
  paper-07/
Each paper folder typically contains:

tex/ : LaTeX source,
figs/ : final figures used in manuscript,
data/ : summarized outputs used in tables/diagnostics,
code/ : scripts to regenerate key results.
Reproducibility Notes
Numerical runs were performed with Python + CLASS/Cobaya workflows.
Some long MCMC chains are not stored in full in this public repo; compact summaries are included where relevant.
For exact environment recreation, see paper-specific code/README.md files.
Citation
If you use this repository, please cite the relevant paper(s) in the TCV-Φ series.
A CITATION.cff file will be maintained at repository root.

Status
This repository is an actively maintained research release.
For final publication metadata (arXiv/DOI), entries will be updated paper by paper as they become available.