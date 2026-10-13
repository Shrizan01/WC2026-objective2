# Regression 2.1: report planning and results guide

This is a planning aid, not assessed report prose. Write your final section in your own words and explain every method you include. The four-page limit applies to the whole group report, including names and IDs. Agree space with the group; around one page for 2.1 is a planning suggestion, not an assessment rule.

## Suggested section outline

### 1. Question and data
- What is predicted? Team A goals minus Team B goals in each actual WC 2026 match.
- State 104 match rows, eight predictors, and that IDs/team names/date are metadata.
- Explain how this complements 2.2: match score difference versus a team's goal count.
- Briefly justify strength (rank/Elo), form (previous wins/tournament result points), experience, host status, rest and stage.
- Link exact definitions to `../DATA_NOTES.md`; acknowledge that source verification is still pending.

### 2. Method and EDA-driven decisions
- Explain the correlation figure: ranking and Elo capture related strength information. Describe the observed pattern rather than asserting one causes the other.
- Explain why Ridge is tested: shrinking coefficients can reduce instability when predictors overlap and the training sample is small.
- Explain chronological validation: earlier dates train the model, later dates evaluate it; same-day games stay together.
- Four development folds use 12/28/40/54 training rows and 16/12/14/18 validation rows. These are small samples; avoid overconfident conclusions.
- Ridge/Lasso alpha is tuned within three chronological inner folds. Scaling is fitted inside each training split.
- Stage is constant in development, so it cannot have a learned effect. Discuss this weakness explicitly.

### 3. Model comparison and recommendation

Development results: equal-weight averages over four chronological group-stage validation folds. Lower MAE/RMSE is better; higher R² is better. Mean fold R² is not a single pooled R².

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Mean baseline | 1.829 | 2.246 | -0.170 |
| OLS | 1.844 | 2.253 | -0.332 |
| Ridge | 1.503 | 1.831 | 0.222 |
| Lasso | 1.635 | 2.022 | 0.061 |
| Huber | 2.111 | 2.697 | -0.959 |
| Random forest benchmark | 1.471 | 1.819 | 0.233 |

- Ridge is selected among the linear candidates by lowest development MAE.
- Its MAE is approximately 18.5% lower than OLS and 17.8% lower than the mean baseline in this comparison. These are descriptive improvements, not significance claims.
- Random forest is slightly better overall on development MAE, but is a nonlinear benchmark, not the chosen linear-regression model.
- Final Ridge alpha from development tuning: 3.1623.
- Explain MAE in goal units: average absolute prediction error, not percentage accuracy.

### 4. Retrospective evaluation and limitations

Fit the selected model on all 72 group-stage rows; evaluate on 32 knockout rows:

| Metric | Value |
|---|---:|
| MAE | 1.167 goals |
| RMSE | 1.487 goals |
| R² | 0.110 |

- Keep these scores separate from development CV scores. Different matches and training sizes mean the lower knockout MAE is not evidence of further optimisation.
- Knockout outcomes were previously inspected. This is a retrospective check, not an untouched test.
- R² = 0.110 is not 11% classification accuracy.
- Features update before each match using earlier outcomes; this is not a forecast of all knockout games made at the start of the knockout stage.
- Discuss incomplete provenance, unusual rest gaps, ±7 clipping, repeated-team dependence, small folds and the unlearned stage effect.
- Conclude with a qualified recommendation and what evidence would be needed to strengthen it.

## Figure selection

Prioritise `fig3_cv_comparison.png` for model selection. If space allows, use `fig6_pred_vs_actual.png` to show prediction error, or `fig1_eda.png` to explain the strength correlations. The group does not need to place every repository chart in the four-page report. Caption the sample/evaluation accurately and refer to each selected figure in your prose.

## Before signing off

- Verify original sources, ranking publication date and missing historical matches.
- Confirm at most four shared predictors with the 2.2 author.
- Have a teammate review the code and interpretation; record their actual feedback in Teams.
- Write and proofread your own report section; do not paste this guide as the assessed report.
- Contribute an accurate AI acknowledgement and reflection to the group's official declaration template.
