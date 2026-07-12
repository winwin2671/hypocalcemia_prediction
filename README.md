# Preoperative prediction of postoperative hypocalcemia after endoscopic thyroidectomy

Reproducible analysis for the **ICTMHS 2026** paper: *Development and internal validation of a preoperative prediction model for postoperative endoscopic thyroidectomy hypocalcemia.*

> ⚠️ **Data notice:** `Thyroid_Sx_research_(ReadyToClean).csv` is patient-level clinical data. Do not keep it in a public repository — see [Data & privacy](#data--privacy).

## Study design

| Item | Value |
|---|---|
| Design | Retrospective cohort, single centre |
| N | 78 patients |
| Outcome | Postoperative hypocalcemia = **postoperative serum calcium < 2.1 mmol/L** (10 events, 12.8%) |
| Preoperative predictors | Pre-op PTH, pre-op calcium, pre-op vitamin D |
| Candidate models | Standard logistic regression · **Firth** logistic regression · **LASSO** logistic regression |
| Internal validation | Bootstrap (B = 1000), Harrell optimism correction |

Events-per-variable (3 predictors) ≈ 3.3 — a small-sample / rare-event regime where penalised methods (Firth, LASSO) are specifically indicated over plain maximum-likelihood logistic regression.

## Headline result

After bootstrap optimism correction, all three models fell from an apparent AUC ≈ 0.64 to **≈ 0.53 (95% CI crossing 0.5)** — none discriminates better than chance once overfitting is removed. **Firth logistic regression was selected** as the final model (the preferred estimator for small-sample rare-event logistic regression, with the tightest coefficient CIs). Pre-op calcium was the strongest predictor in every model.

This is a transparent, hypothesis-generating *limited* result — the appropriate posture for a small preliminary cohort, and stronger scientifically than an overfit AUC that would not replicate. Full interpretation, calibration, decision-curve analysis, and line-by-line manuscript-update guidance are written into the notebook.

## Repository contents

| Path | Description |
|---|---|
| `model_analysis.ipynb` | **Main deliverable** — the full 7-step analysis (code + per-step "learnings & verdict" markdown for the clinical team) |
| `Thyroid_Sx_research_(ReadyToClean).csv` | Raw dataset (patient-level — see Data notice) |
| `old_manuscript_TOETVA_2026.pdf` | Current manuscript draft (scope changed TOETVA → endoscopic; many `[PENDING]` fields) |
| `figures/` | Publication figures: ROC (`fig1_roc`), calibration (`fig2_calibration`), decision-curve analysis (`fig3_dca`) |
| `pyproject.toml`, `uv.lock` | Environment (Python ≥ 3.13, managed with [uv](https://docs.astral.sh/uv/)) |

Files beginning with `_` (`_eda.py`, `_analysis_dev.py`, `_build_notebook.py`, `_sota_metrics.py`, `_test_sota.py`, `_manuscript_text.txt`) are development scaffolding used to build the notebook; they are not needed to read or rerun the analysis.

## How to run

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.13.

```bash
uv sync                       # install dependencies
uv run jupyter lab            # open and run model_analysis.ipynb
```

Re-execute the whole notebook headless (regenerates all outputs + figures):

```bash
uv run --with nbconvert --with ipykernel \
  jupyter nbconvert --to notebook --execute --inplace model_analysis.ipynb
```

All randomness is seeded (`SEED = 42`), so results are deterministic and reproducible.

## Methods & tooling

- **Firth logistic regression** implemented dependency-free (Firth 1993; Heinze & Schemper 2002) — the `firthlogist` package is incompatible with numpy ≥ 2.
- **Optimism-corrected AUC** via a hand-coded Harrell/Steyerberg bootstrap; cross-checked with **DeLong** AUC 95% CIs (Sun & Xu 2014), an independent asymptotic method. Generic bootstrap libraries don't implement the fit-on-resample / test-on-original optimism step, and there is no mature Python equivalent of R's `rms::validate()`.
- **Calibration:** binned observed-vs-predicted, Brier score, and the Integrated Calibration Index (Austin 2020).
- **Decision-curve analysis** via [`dcurves`](https://github.com/pypi-release/dcurves).
- A gradient-boosting comparator is shown only as a memorisation check (it hits apparent AUC = 1.0), empirically justifying the regression-only design at EPV ≈ 3.3.

Key references: van der Ploeg 2014 (EPV / data-hungry models); Christodoulou 2019 (ML vs logistic for clinical prediction); Harrell, *Regression Modeling Strategies*; Steyerberg, *Clinical Prediction Models* (2nd ed.); TRIPOD+AI (2024).

## Data & privacy

The CSV contains patient-level clinical variables (demographics, operative details, labs, surgery dates). Even de-identified, this can be re-identifiable. **Do not keep it in a public repository.** Make the repo private and/or remove the file from git history, and confirm what your ethics/IRB approval permits regarding data sharing before any publication or data deposit.

## Status

- ✅ Analysis complete — `model_analysis.ipynb` executes end-to-end with 0 errors.
- ⏳ Manuscript `[PENDING]` fields still to be filled (abstract numbers, inclusion/exclusion criteria, institution, dates, ethics statement).
- 🔜 Recommended future work: larger multicentre cohort, inclusion of surgical-extent variables (which vary little in this mostly-unilateral-lobectomy cohort), and external validation.
