"""
Independent checks of data/processed/wc2026_lr21_refined.csv.
Checks internal consistency against supplied raw files, not external authenticity.
Elo is checked for plausibility only; this is not a complete independent audit.
Run:  python src/verify_lr21.py      (prints PASS/FAIL for each check)
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data/processed/wc2026_lr21_refined.csv", parse_dates=["date"])
res = pd.read_csv(ROOT / "data/raw/results.csv", parse_dates=["date"])
rank = pd.read_csv(ROOT / "data/raw/fifa_rank_june2026.csv").set_index("team")["fifa_rank_june2026"]
wc = res[(res.tournament == "FIFA World Cup") & (res.date >= "2026-06-01")]
checks = {}

# Structure
checks["104 unique matches, no missing values"] = (sorted(df.match_id) == list(range(1, 105))
                                                   and df.isna().sum().sum() == 0)
teams = pd.concat([df.team_a, df.team_b])
checks["48 teams, each plays 3-8 matches"] = teams.nunique() == 48 and teams.value_counts().between(3, 8).all()

# Target and date against results.csv
lookup = {frozenset([r.home_team, r.away_team]): r for r in wc.itertuples()}
gd_ok = date_ok = True
for r in df.itertuples():
    m = lookup[frozenset([r.team_a, r.team_b])]
    gd = m.home_score - m.away_score if m.home_team == r.team_a else m.away_score - m.home_score
    gd_ok &= gd == r.goal_diff
    date_ok &= m.date == r.date
checks["goal_diff matches results.csv"] = gd_ok
checks["date matches results.csv"] = date_ok

# Stage: 72 group, 16 R32, 8 R16, 4 QF, 2 SF, 1 third place, 1 final; final is the last match
checks["stage counts 72/16/8/4/2/1/1"] = df.stage.value_counts().sort_index().tolist() == [72, 16, 8, 4, 2, 1, 1]
checks["stage never decreases over time"] = df.sort_values(["date", "match_id"]).stage.is_monotonic_increasing
last = df.sort_values("date").iloc[-1]
checks["final (stage 7) is Spain v Argentina, last match"] = (last.stage == 7 and {last.team_a, last.team_b} == {"Spain", "Argentina"})

# Predictors
checks["rank_diff = rank B - rank A"] = (df.rank_diff == df.team_b.map(rank) - df.team_a.map(rank)).all()
hosts = {"Mexico", "United States", "Canada"}
checks["host_diff correct"] = (df.host_diff == df.team_a.isin(hosts).astype(int) - df.team_b.isin(hosts).astype(int)).all()

pts, ok = {}, True
for r in df.sort_values(["date", "match_id"]).itertuples():
    ok &= pts.get(r.team_a, 0) - pts.get(r.team_b, 0) == r.tourn_pts_diff     # uses earlier matches only
    a, b = (3, 0) if r.goal_diff > 0 else (0, 3) if r.goal_diff < 0 else (1, 1)
    pts[r.team_a] = pts.get(r.team_a, 0) + a
    pts[r.team_b] = pts.get(r.team_b, 0) + b
checks["tourn_pts_diff uses only earlier matches"] = ok

def winrate10(team, date):
    g = res[((res.home_team == team) | (res.away_team == team)) & (res.date < date)].tail(10)
    won = ((g.home_team == team) & (g.home_score > g.away_score)) | ((g.away_team == team) & (g.away_score > g.home_score))
    return won.mean()
checks["winrate10_diff uses only earlier matches"] = all(
    abs(winrate10(r.team_a, r.date) - winrate10(r.team_b, r.date) - r.winrate10_diff) < 0.011 for r in df.itertuples())

def rest(team, date):
    prev = res[((res.home_team == team) | (res.away_team == team)) & (res.date < date)].date.max()
    return (date - prev).days
checks["rest_diff correct (after +/-7 cap)"] = all(
    np.clip(rest(r.team_a, r.date) - rest(r.team_b, r.date), -7, 7) == r.rest_diff for r in df.itertuples())

checks["elo_diff plausible (|diff| < 700, correlates with rank_diff)"] = (df.elo_diff.abs().max() < 700
                                                                         and df.elo_diff.corr(df.rank_diff) > 0.8)

width = max(map(len, checks))
for name, passed in checks.items():
    print(f"{name:<{width}}  {'PASS' if passed else 'FAIL'}")
print(f"\n{sum(checks.values())}/{len(checks)} checks passed")

if not all(checks.values()):
    raise SystemExit(1)
