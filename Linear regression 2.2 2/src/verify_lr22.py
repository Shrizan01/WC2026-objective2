"""
Independent checks of data/processed/wc2026_lr22_v2.csv.
Every column is recomputed a second way (plain loops, different joins) and compared.
Goals and dates are checked against data/raw/results.csv, an independent source.
Run:  python src/verify_lr22.py      (prints PASS/FAIL for each check)
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data/processed/wc2026_lr22_v2.csv", parse_dates=["date"])
res = pd.read_csv(ROOT / "data/raw/results.csv", parse_dates=["date"])
src = pd.read_csv(ROOT / "data/raw/src_matches.csv")
st = pd.read_csv(ROOT / "data/raw/src_match_team_stats.csv")
names = pd.read_csv(ROOT / "data/raw/src_teams.csv").set_index("team_id")["team_name"].replace(
    {"USA": "United States", "Czechia": "Czech Republic", "Türkiye": "Turkey",
     "Côte d'Ivoire": "Ivory Coast", "IR Iran": "Iran", "Cabo Verde": "Cape Verde", "Congo DR": "DR Congo"})
wc = res[(res.tournament == "FIFA World Cup") & (res.date >= "2026-06-01")]
checks = {}

# Structure
checks["208 rows, 104 matches x 2 teams, no missing values"] = (
    len(df) == 208 and sorted(df.match_id.unique()) == list(range(1, 105))
    and (df.match_id.value_counts() == 2).all() and df.isna().sum().sum() == 0)
checks["48 teams, each plays 3-8 matches"] = df.team.nunique() == 48 and df.team.value_counts().between(3, 8).all()
pairs = df.groupby("match_id").apply(lambda g: g.team.iloc[0] == g.opponent.iloc[1] and g.team.iloc[1] == g.opponent.iloc[0])
checks["the two rows of each match mirror each other"] = pairs.all()

# Target and date against results.csv
lookup = {frozenset([r.home_team, r.away_team]): r for r in wc.itertuples()}
gs_ok = date_ok = True
for r in df.itertuples():
    m = lookup[frozenset([r.team, r.opponent])]
    gs_ok &= (m.home_score if m.home_team == r.team else m.away_score) == r.goals_scored
    date_ok &= m.date == r.date
checks["goals_scored matches results.csv"] = gs_ok
checks["date matches results.csv"] = date_ok

# Stage: 72 group, 16 R32, 8 R16, 4 QF, 2 SF, 2 final-weekend matches (3rd place + Final = 6)
checks["stage counts 72/16/8/4/2/2 matches"] = (
    df.drop_duplicates("match_id").stage.value_counts().sort_index().tolist() == [72, 16, 8, 4, 2, 2])
checks["stage never decreases over time"] = df.sort_values(["date", "match_id"]).stage.is_monotonic_increasing
last = df.sort_values("date").iloc[-1]
checks["final (stage 6) is Spain v Argentina, last match"] = (
    last.stage == 6 and {last.team, last.opponent} == {"Spain", "Argentina"})

# Host flag
checks["team_host = 1 only for Mexico, United States, Canada"] = (
    set(df.loc[df.team_host == 1, "team"]) == {"Mexico", "United States", "Canada"}
    and (df.team_host == df.team.isin({"Mexico", "United States", "Canada"}).astype(int)).all())

# Rolling form, recomputed with explicit loops over each team's earlier matches
# Raw per-team match stats keyed by (team, opponent) name pair -> different join path from the build script
raw = {}
for m in src.itertuples():
    h, a = names[m.home_team_id], names[m.away_team_id]
    sh = st[(st.match_id == m.match_id) & (st.team_id == m.home_team_id)].iloc[0]
    sa = st[(st.match_id == m.match_id) & (st.team_id == m.away_team_id)].iloc[0]
    raw[(h, a)] = dict(shots=sh.total_shots, shots_against=sa.total_shots, poss=sh.possession_pct)
    raw[(a, h)] = dict(shots=sa.total_shots, shots_against=sh.total_shots, poss=sa.possession_pct)

def history(team, date):
    return df[(df.team == team) & (df.date < date)]

def mean0(values):
    return float(np.mean(values)) if len(values) else 0.0

form_ok = {c: True for c in ["avg_gf_team", "avg_ga_opp", "avg_shots_team",
                             "opp_avg_shots_against", "avg_possession_team", "rest_team"]}
for r in df.itertuples():
    h_team, h_opp = history(r.team, r.date), history(r.opponent, r.date)
    # goals conceded by the opponent = goals scored by whoever the opponent played
    opp_conceded = [df[(df.match_id == x.match_id) & (df.team != r.opponent)].goals_scored.iloc[0]
                    for x in h_opp.itertuples()]
    expect = {
        "avg_gf_team": mean0(h_team.goals_scored),
        "avg_ga_opp": mean0(opp_conceded),
        "avg_shots_team": mean0([raw[(r.team, x.opponent)]["shots"] for x in h_team.itertuples()]),
        "opp_avg_shots_against": mean0([raw[(r.opponent, x.opponent)]["shots_against"] for x in h_opp.itertuples()]),
        "avg_possession_team": mean0([raw[(r.team, x.opponent)]["poss"] for x in h_team.itertuples()]),
        "rest_team": (r.date - h_team.date.max()).days if len(h_team) else 0,
    }
    for c, v in expect.items():
        form_ok[c] &= abs(getattr(r, c) - v) < 1e-3
for c, ok in form_ok.items():
    checks[f"{c} uses only earlier matches"] = ok

first = df.sort_values("date").groupby("team").head(1)
checks["all form variables are 0 in each team's first match (48 rows)"] = (
    len(first) == 48 and (first[list(form_ok)] == 0).all().all())

# Plausibility of the raw match statistics
poss_sum = st.groupby("match_id").possession_pct.sum()
checks["raw possession of both teams never exceeds 100%"] = poss_sum.le(100).all() and st.possession_pct.between(0, 100).all()
checks["rest_team between 0 and 10 days"] = df.rest_team.between(0, 10).all()

width = max(map(len, checks))
for name, passed in checks.items():
    print(f"{name:<{width}}  {'PASS' if passed else 'FAIL'}")
print(f"\n{sum(checks.values())}/{len(checks)} checks passed")

short = poss_sum[poss_sum < 100]
print(f"\nNOTE: in {len(short)} of 104 matches the two possession values sum to "
      f"{short.min()}-{short.max()}% instead of 100% (fifa.com reports a separate 'ball in contest' share). "
      "Values are used as published; see README limitations.")
