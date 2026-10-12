"""
Build the refined LR 2.1 dataset (one row per match, 104 matches).

Target:    goal_diff = Team A goals - Team B goals (incl. extra time; shootouts ignored)

Predictors (all "Team A minus Team B", all known BEFORE kick-off):
    rank_diff       FIFA rank of B - rank of A (positive = A ranked higher)
    stage           1=Group 2=R32 3=R16 4=QF 5=SF 6=3rd place 7=Final
    rest_diff       days since last international, A - B (capped at +/-7)
    elo_diff        Elo rating of A - B, computed only from earlier matches
    winrate10_diff  win rate in each team's last 10 internationals, A - B
    wc_exp_diff     number of previous World Cups played, A - B
    host_diff       +1 if A is a host nation, -1 if B is, else 0
    tourn_pts_diff  points earned earlier in THIS tournament, A - B

Inputs  (data/raw):       results.csv, fifa_rank_june2026.csv, wc2026_lr21_v2.csv
Output  (data/processed): wc2026_lr21_refined.csv
Run:    python src/build_lr21_refined.py
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- 1. Load data
results = (pd.read_csv(RAW / "results.csv", parse_dates=["date"])
             .sort_values("date", kind="stable").reset_index(drop=True))
fifa_rank = pd.read_csv(RAW / "fifa_rank_june2026.csv").set_index("team")["fifa_rank_june2026"]
fixtures = pd.read_csv(RAW / "wc2026_lr21_v2.csv", parse_dates=["date"])  # used only for match_id, team order, stage

is_wc2026 = (results.tournament == "FIFA World Cup") & (results.date >= "2026-06-01")
HOSTS = {"Mexico", "United States", "Canada"}

# ---------------------------------------------------------------- 2. Elo ratings
# World Football Elo method (eloratings.net). We walk through every international in
# date order and store each team's rating BEFORE the match -> no data leakage.
def k_factor(tournament: str) -> int:
    t = tournament.lower()
    if t == "fifa world cup":
        return 60
    if "qualification" in t:
        return 40
    if t in ("uefa euro", "copa américa", "african cup of nations", "afc asian cup", "gold cup", "confederations cup"):
        return 50
    if t == "friendly":
        return 20
    return 30

elo, elo_before = {}, {}
for i, m in results.iterrows():
    eh, ea = elo.get(m.home_team, 1500), elo.get(m.away_team, 1500)
    elo_before[i] = (eh, ea)                                   # pre-match ratings
    rating_gap = eh - ea + (0 if m.neutral else 100)           # +100 home advantage
    expected_home = 1 / (10 ** (-rating_gap / 400) + 1)
    margin = abs(m.home_score - m.away_score)
    g = 1 if margin <= 1 else 1.5 if margin == 2 else (11 + margin) / 8
    actual_home = 1 if m.home_score > m.away_score else 0.5 if m.home_score == m.away_score else 0
    change = k_factor(m.tournament) * g * (actual_home - expected_home)
    elo[m.home_team], elo[m.away_team] = eh + change, ea - change

# ---------------------------------------------------------------- 3. Form and rest
# One row per team per match, so we can use groupby + shift() (shift = "previous matches only").
team_games = pd.concat([
    pd.DataFrame({"i": results.index, "date": results.date, "team": results.home_team,
                  "gf": results.home_score, "ga": results.away_score}),
    pd.DataFrame({"i": results.index, "date": results.date, "team": results.away_team,
                  "gf": results.away_score, "ga": results.home_score}),
]).sort_values(["date", "i"])
team_games["win"] = (team_games.gf > team_games.ga).astype(float)
team_games["winrate10"] = (team_games.groupby("team")["win"]
                           .transform(lambda s: s.shift().rolling(10, min_periods=1).mean()))
team_games["rest"] = (team_games.date - team_games.groupby("team")["date"].shift()).dt.days
team_games = team_games.set_index(["i", "team"])

# ---------------------------------------------------------------- 4. World Cup experience
past_wc = results[(results.tournament == "FIFA World Cup") & (results.date < "2026-01-01")]
wc_exp = (pd.concat([past_wc[["date", "home_team"]].rename(columns={"home_team": "team"}),
                     past_wc[["date", "away_team"]].rename(columns={"away_team": "team"})])
            .assign(year=lambda d: d.date.dt.year)
            .groupby("team")["year"].nunique())

# ---------------------------------------------------------------- 5. Build one row per WC 2026 match
points, rows = {}, []
for i, m in results[is_wc2026].iterrows():
    pair = fixtures[((fixtures.team_a == m.home_team) & (fixtures.team_b == m.away_team)) |
                    ((fixtures.team_a == m.away_team) & (fixtures.team_b == m.home_team))].iloc[0]
    A, B = pair.team_a, pair.team_b
    flipped = A != m.home_team
    elo_h, elo_a = elo_before[i]
    elo_A, elo_B = (elo_a, elo_h) if flipped else (elo_h, elo_a)
    goals_A, goals_B = (m.away_score, m.home_score) if flipped else (m.home_score, m.away_score)

    rows.append({
        "match_id": pair.match_id, "date": m.date.date(), "team_a": A, "team_b": B,
        "goal_diff": goals_A - goals_B,
        "rank_diff": fifa_rank[B] - fifa_rank[A],
        "stage": pair.stage,
        "rest_diff": team_games.loc[(i, A), "rest"] - team_games.loc[(i, B), "rest"],
        "elo_diff": round(elo_A - elo_B, 1),
        "winrate10_diff": team_games.loc[(i, A), "winrate10"] - team_games.loc[(i, B), "winrate10"],
        "wc_exp_diff": wc_exp.get(A, 0) - wc_exp.get(B, 0),
        "host_diff": int(A in HOSTS) - int(B in HOSTS),
        "tourn_pts_diff": points.get(A, 0) - points.get(B, 0),   # points BEFORE this match
    })
    # update tournament points AFTER recording the row (shootouts count as draws)
    points[A] = points.get(A, 0) + (3 if goals_A > goals_B else 1 if goals_A == goals_B else 0)
    points[B] = points.get(B, 0) + (3 if goals_B > goals_A else 1 if goals_A == goals_B else 0)

df = pd.DataFrame(rows).sort_values("match_id").reset_index(drop=True)

# ---------------------------------------------------------------- 6. Cleaning decisions (document in report)
df.loc[df.match_id == 104, "stage"] = 7                 # final = 7; match 103 (3rd place) stays 6
# Uruguay shows 76 days' rest before its opener (pre-tournament friendlies missing from results.csv),
# Japan 14 days. Cap so two data gaps do not dominate the fit (affects matches 11 and 16).
df["rest_diff"] = df.rest_diff.clip(-7, 7)
df["winrate10_diff"] = df.winrate10_diff.round(2)

assert len(df) == 104 and df.isna().sum().sum() == 0
df.to_csv(OUT / "wc2026_lr21_refined.csv", index=False)
print(f"Saved {len(df)} rows -> {OUT / 'wc2026_lr21_refined.csv'}")
print(df.head())
