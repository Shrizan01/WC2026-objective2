# Data definitions and unresolved source checks

These files were supplied with the project. Their original download URLs and retrieval dates were not recorded. `data/source_manifest.json` identifies the exact raw files by SHA-256; it does not certify their authenticity. Do not invent retrieval dates or describe the data as externally verified.

| File | Used for | Status |
|---|---|---|
| `results.csv` | Scores, dates, historical Elo, form, rest, WC appearances | Original source and completeness need confirmation |
| `fifa_rank_june2026.csv` | Pre-tournament ranking | Publication date and all 48 rankings need primary-source confirmation |
| `wc2026_lr21_v2.csv` | Match IDs, orientation, stages | Fixtures/stages need primary-source confirmation |

The project brief suggests FIFA, The Stats Don't Lie and FBref. Those are potential verification sources, not proven origins of these files. Record the exact page, publication/retrieval date and any corrections when verified.

## Dataset dictionary

One row is one match. `match_id`, `date`, `team_a`, `team_b` are metadata, never predictors. `goal_diff` is A's score minus B's score, using supplied scores (intended to include extra time and exclude shootout goals; confirm source convention).

Exactly eight model inputs:

| Predictor | Definition and caveat |
|---|---|
| rank_diff | Rank B minus rank A; positive means A ranks higher. Ranking date remains unverified. |
| elo_diff | A minus B custom Elo-style ratings before the match; starts teams at 1500, uses tournament K factors, goal margin and home adjustment from the builder. Not an official rating series. |
| winrate10_diff | A minus B proportion of wins in the previous ten recorded internationals; shootout advancement does not count as a score win. Rounded to two decimals. |
| wc_exp_diff | A minus B number of distinct prior WC years under the recorded team name; predecessor countries excluded. |
| tourn_pts_diff | A minus B cumulative 3/1/0 result score before the match, including earlier knockout games. A custom form measure, not official knockout standings points. |
| host_diff | A's co-host indicator minus B's; Canada, Mexico, United States. Not actual home advantage at a particular venue. |
| rest_diff | Days since A's previous recorded international minus B's, clipped to ±7. A gap of 76 days for Uruguay and 14 for Japan requires investigation; clipping does not fix missing source matches. |
| stage | Group=1, R32=2, R16=3, QF=4, SF=5, third place=6, final=7. Match 104 is recoded from supplied 6 to 7. Numeric encoding assumes equal increments. |

Historical features depend on source completeness. Missing games may affect Elo and win rates as well as rest. No values have been fabricated to repair these gaps. The source data is unchanged by the validation revision.

## Evaluation and limitations

- The 72 group-stage matches form the development sample. Four expanding chronological validation blocks compare six methods; all games on one date stay together.
- Ridge and Lasso tune alpha in three expanding inner date blocks. The complete scaler/model pipeline is fitted within each inner training split.
- Selection uses mean fold MAE among OLS, Ridge, Lasso and Huber. Mean prediction and random forest are benchmarks. Fold sizes differ: mean fold MAE is an equal-fold average, not a pooled match-weighted MAE.
- The selected linear model is fitted on the 72 development matches and evaluated on the remaining 32. Those outcomes were previously inspected, so this is retrospective evaluation, not an untouched test or independent confirmation of model selection.
- Historical features update before each evaluated match. Model parameters stay fixed across the knockout period. This represents sequential pre-match prediction, not forecasting every knockout game at the end of the group stage.
- Stage is constant during development, so its coefficient cannot be learned. It remains one of the eight inputs, but contributes no learned stage effect. The descriptive OLS summary omits constant predictors to avoid a singular intercept/stage design.
- Descriptive OLS uses HC3 standard errors; these do not address repeated-team dependence. Coefficients are conditional associations, not causal effects. Small folds make model rankings uncertain. No significance claim is made.
- Confirm the maximum-four-variable overlap with Regression 2.2 before submission; renaming or differencing a shared concept does not automatically establish a new variable.
