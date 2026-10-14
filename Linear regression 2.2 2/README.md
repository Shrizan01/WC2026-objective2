# HIT140 Assessment 3 – Linear Regression 2.2

Predicting **goals scored by a team** in each FIFA World Cup 2026 match (104 matches × 2 teams = 208 rows).
Author: Sumit Maharjan

## Setup (once)
From the repository root:
```bash
cd "Linear regression 2.2"
bash setup.sh
```
Creates `.venv`, installs `requirements.txt`, registers the Jupyter kernel **Python (hit140-lr22)**, builds the dataset and runs the verification checks.

Afterwards: `source .venv/bin/activate`, then `jupyter lab` and open `notebooks/01_lr22_analysis.ipynb`.

Using conda instead: `conda install pandas numpy scipy scikit-learn statsmodels matplotlib seaborn jupyterlab`, then run
`python src/build_lr22.py`, `python src/verify_lr22.py` and open the notebook.

## Structure
```
data/raw/          src_matches.csv, src_match_team_stats.csv, src_teams.csv,
                   wc2026_lr21_v2.csv (shared fixture list), results.csv (for verification)
data/processed/    wc2026_lr22_v2.csv        <- modelling dataset
src/               build_lr22.py             <- builds the dataset from raw data
                   verify_lr22.py            <- independent checks of every column
notebooks/         01_lr22_analysis.ipynb    <- full analysis (Sections 0-11)
figures/           11 exported figures for the report
```

## Variables (team-level, all known before kick-off)
| Variable | Meaning | Shared with 2.1? |
|---|---|---|
| `avg_gf_team` | Rolling avg goals scored by the team in its earlier matches at this World Cup | No |
| `avg_ga_opp` | Rolling avg goals conceded by the opponent in its earlier matches | No |
| `stage` | Group=1, R32=2, R16=3, QF=4, SF=5, Final=6 (3rd-place match also 6) | Yes (2.1 codes the Final as 7) |
| `rest_team` | Days since the team's previous match at this World Cup (0 = first match) | Related (2.1 uses rest difference) |
| `avg_shots_team` | Rolling avg shots taken by the team | No |
| `opp_avg_shots_against` | Rolling avg shots the opponent has allowed | No |
| `avg_possession_team` | Rolling avg possession % of the team | No |
| `team_host` | 1 if the team is Canada, Mexico or United States, else 0 | Related (2.1 uses `host_diff`) |

Target: `goals_scored` (whole number, 0–7, mean 1.48).

## Analysis (notebook sections)
| Section | Content |
|---|---|
| 1 | Data sources and wrangling steps |
| 2 | 80/20 train-test split (seed 42), StandardScaler fitted on train only |
| 3 | Descriptive statistics and Pearson correlations with significance |
| 4 | OLS regression (statsmodels) with full summary |
| 5 | Assumption tests: Shapiro-Wilk, Durbin-Watson, Breusch-Pagan, VIF |
| 6 | OLS, Ridge, Lasso, ElasticNet with GridSearchCV tuning |
| 7 | 5-fold cross-validation |
| 8 | Welch t-test: group stage vs knockout goals |
| 9 | Why R² is low / negative in some folds |
| 10 | Figures 1–11 |
| 11 | Plain-English findings summary |

## Data decisions (cite in report)
- One row per team per match; each match appears twice (once from each side).
- Rolling averages use only matches **before** the current one, so there is no data leakage.
- Each team's first match (48 rows, 23%) has no history: all rolling values and `rest_team` are 0. Structural, not imputed.
- Goals include extra time; penalty shootout goals are not counted.
- `match_id` and local match dates come from the shared fixture list (`wc2026_lr21_v2.csv`), so LR 2.1 and 2.2 join on `match_id`. `src_matches.csv` numbers matches differently and gives the UTC date, which is one day late for 30 evening kick-offs; all dates were checked against `results.csv`.
- Team names standardised to the names used in `results.csv` and LR 2.1 (e.g. Czechia → Czech Republic, Türkiye → Turkey, IR Iran → Iran).

## Limitations (mention in report)
- **Possession definition:** in 24 of 104 matches the two teams' possession adds to 81–93% rather than 100%, because fifa.com reports a separate "ball in contest" share. Values are used as published.
- **First-match zeros:** a value of 0 means "no history", not "zero form", and the 48 first-match rows share the same zero pattern across all rolling variables.
- **Count target:** goals are whole numbers, so OLS residuals are non-normal (Shapiro-Wilk p < 0.001).
- Small sample: 208 rows, 64 of them knockout rows.

## To do
- [x] Data verified: run `python src/verify_lr22.py` (18/18 checks pass)
- [x] Analysis notebook: `notebooks/01_lr22_analysis.ipynb` (figures in `figures/`)
- [ ] Write the LR 2.2 section of the group report
