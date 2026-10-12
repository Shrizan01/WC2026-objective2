# HIT140 Assessment 3 – Linear Regression 2.1

Predicting **goal difference** (Team A − Team B) for all 104 FIFA World Cup 2026 matches.

## Setup (once)
From the repository root:
```bash
cd "Linear regression 2.1"
bash setup.sh
```
Creates `.venv`, installs `requirements.txt`, registers the Jupyter kernel **Python (hit140-lr21)** and builds the dataset.

Afterwards: `source .venv/bin/activate`

## Structure
```
data/raw/          results.csv, fifa_rank_june2026.csv, wc2026_lr21_v2.csv (fixture list)
data/processed/    wc2026_lr21_refined.csv   <- modelling dataset
src/               build_lr21_refined.py     <- builds the dataset from raw data
                   verify_lr21.py            <- independent checks of every column
notebooks/         analysis notebooks
figures/           exported plots for the report
```

## Variables (all Team A − Team B, all known before kick-off)
| Variable | Meaning | Shared with 2.2? |
|---|---|---|
| `rank_diff` | FIFA rank of B − rank of A (positive = A ranked higher) | Grey area |
| `stage` | 1 Group, 2 R32, 3 R16, 4 QF, 5 SF, 6 3rd place, 7 Final | Yes |
| `rest_diff` | Rest days A − B, capped at ±7 | Yes |
| `elo_diff` | Elo rating A − B, from earlier matches only | No |
| `winrate10_diff` | Win rate in last 10 internationals | No |
| `wc_exp_diff` | Previous World Cups played | No |
| `host_diff` | +1 A is host, −1 B is host, 0 neither | Depends on 2.2 version |
| `tourn_pts_diff` | Points earned earlier in this tournament | No |

## Data decisions (cite in report)
- Goals include extra time; penalty shootouts are treated as draws.
- `rest_diff` capped at ±7 days: Uruguay showed 76 days' rest before its opener (pre-tournament friendlies missing from `results.csv`) and Japan 14. Affects 2 matches (11 and 16).
- Final (Spain v Argentina, match 104) coded as stage 7 and third-place match (France v England, 103) as stage 6. The original fixture file had both as 6; this now matches the official stage IDs in `src_matches.csv`.
- FIFA ranks: `fifa_rank_june2026.csv` matches the published pre-tournament June 2026 ranking for 46 of 48 teams (Haiti/Curaçao may be swapped at 82/83 — one place, negligible). The ranks in `src_teams.csv` are wrong for 13 teams and are not used.
- `wc_exp_diff` counts World Cups under the current team name, so Czech Republic gets 1 (Czechoslovakia's 8 appearances are excluded). Mention as a limitation.

## To do
- [x] Data verified: run `python src/verify_lr21.py` (all checks pass)
- [ ] Optional: confirm Haiti (82/83) and Curaçao (82/83) on fifa.com
- [x] Model comparison notebook: `notebooks/01_lr21_model_comparison.ipynb` (outputs in `figures/`)
