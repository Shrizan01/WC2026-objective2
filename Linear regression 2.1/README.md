# HIT140 — Linear regression 2.1

Predict match goal difference using 104 rows and eight pre-match predictors. This is a working analysis with unresolved source checks, not a submission-ready claim of verified data. See [data definitions and limitations](DATA_NOTES.md).

## Run

From the repository root:

```bash
cd "Linear regression 2.1"
bash setup.sh
source .venv/bin/activate
python src/verify_lr21.py
python src/run_analysis.py
```

The setup creates an environment and rebuilds the dataset. The analysis runs independently of the current working directory. Alternatively, open `notebooks/01_lr21_model_comparison.ipynb` from this folder or its notebooks folder; it calls the same script.

## Files

- `data/raw/`: supplied historical matches, rankings and fixture list.
- `data/processed/`: 104-row modelling dataset.
- `data/source_manifest.json`: input fingerprints and unresolved provenance.
- `src/build_lr21_refined.py`: feature construction.
- `src/verify_lr21.py`: internal consistency checks; exits unsuccessfully if a check fails. These do not verify external facts.
- `src/run_analysis.py`: EDA, chronological model comparison, tuning, diagnostics and figures.
- `notebooks/01_lr21_model_comparison.ipynb`: executable interface to the same analysis.
- `figures/`: reproducible figures, per-fold metrics, date splits, predictions, model summary and development OLS output.

## Evaluation

Four expanding date blocks within the 72 group-stage matches compare a mean baseline, OLS, Ridge, Lasso, Huber and random forest. Ridge/Lasso tuning uses chronological inner validation with scaling inside each fit. The lowest development MAE among the linear candidates determines the selected model. The 32 knockout matches are a **previously inspected retrospective evaluation**. See `figures/results_summary.json` for the selected model and limitations; `table_model_comparison.csv` reports development-fold averages, not knockout scores.

The old random-fold analysis is retained in Git history, not as competing current results. The obsolete Lasso-path figure was removed because that analysis is no longer run. Stage is constant in development; source gaps and the capped rest variable remain limitations.

## Assessment completion

- Resolve source URLs, ranking date, score conventions and missing historical matches.
- Agree and document variable overlap with the Regression 2.2 author.
- Interpret actual results and limitations in your own report prose; do not claim significance from small metric differences.
- Post real decisions, reviews and contribution evidence in the lecturer's Teams space. Link GitHub work there; GitHub does not replace Teams evidence.
- The group submission also needs its four-page Word report, repository URL text file, Turnitin report and signed AI-use declaration. These are separate from this analysis folder.

## Documentation for your contribution

- [Report outline and results guide](docs/REPORT_OUTLINE_AND_RESULTS.md) — evidence and prompts for writing your own section.
- [Contribution and review record](docs/CONTRIBUTION_AND_REVIEW.md) — complete with actual contributions and peer feedback.
- [Teams update draft](docs/TEAMS_UPDATE_DRAFT.md) — adapt and post yourself; not sent automatically.
