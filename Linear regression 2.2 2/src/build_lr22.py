"""
Build the LR 2.2 dataset (one row per team per match: 104 matches x 2 teams = 208 rows).

Target:    goals_scored = goals the team scored in the match (incl. extra time; shootouts ignored)

Predictors (all team-level, all known BEFORE kick-off):
    avg_gf_team            team's average goals scored in its earlier matches at this World Cup
    avg_ga_opp             opponent's average goals conceded in its earlier matches
    stage                  1=Group 2=R32 3=R16 4=QF 5=SF 6=Final (3rd-place match also 6)
    rest_team              days since the team's previous match at this World Cup (0 = first match)
    avg_shots_team         team's average total shots in its earlier matches
    opp_avg_shots_against  average shots the opponent has allowed in its earlier matches
    avg_possession_team    team's average possession % in its earlier matches
    team_host              1 if the team is a host nation (Mexico, United States, Canada), else 0

Inputs  (data/raw):       src_matches.csv, src_match_team_stats.csv, src_teams.csv,
                          wc2026_lr21_v2.csv (shared fixture list: match_id, local date, team order)
Output  (data/processed): wc2026_lr22_v2.csv
Run:    python src/build_lr22.py
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- 1. Load data
matches = pd.read_csv(RAW / "src_matches.csv", parse_dates=["date"])
stats = pd.read_csv(RAW / "src_match_team_stats.csv")
teams = pd.read_csv(RAW / "src_teams.csv").set_index("team_id")["team_name"]

# FIFA's official short names -> the common English names used by the rest of the group
# (same spelling as results.csv and Suraj's LR 2.1 dataset).
RENAME = {"USA": "United States", "Czechia": "Czech Republic", "Türkiye": "Turkey",
          "Côte d'Ivoire": "Ivory Coast", "IR Iran": "Iran", "Cabo Verde": "Cape Verde",
          "Congo DR": "DR Congo"}
teams = teams.replace(RENAME)
HOSTS = {"Mexico", "United States", "Canada"}

# Match ID and local match date come from the group's shared fixture list (same as LR 2.1),
# so both datasets can be joined on match_id. src_matches.csv numbers matches differently and
# gives the UTC date, which is one day late for 30 evening kick-offs in the Americas.
fixtures = pd.read_csv(RAW / "wc2026_lr21_v2.csv", parse_dates=["date"])
pair_key = lambda a, b: frozenset([a, b])
fixture_of = {pair_key(a, b): (mid, d) for a, b, mid, d in
              zip(fixtures.team_a, fixtures.team_b, fixtures.match_id, fixtures.date)}
home, away = matches.home_team_id.map(teams), matches.away_team_id.map(teams)
matches["src_match_id"] = matches.match_id                      # needed to join src_match_team_stats
matches["match_id"] = [fixture_of[pair_key(h, a)][0] for h, a in zip(home, away)]
matches["date"] = [fixture_of[pair_key(h, a)][1] for h, a in zip(home, away)]

# ---------------------------------------------------------------- 2. One row per team per match
# Each match appears twice: once from the home side's point of view, once from the away side's.
def side(team_col, opp_col, gf_col, ga_col):
    return pd.DataFrame({
        "match_id": matches.match_id, "src_match_id": matches.src_match_id,
        "date": matches.date, "stage": matches.stage_id,
        "team_id": matches[team_col], "opp_id": matches[opp_col],
        "goals_scored": matches[gf_col], "goals_conceded": matches[ga_col],
    })

games = pd.concat([side("home_team_id", "away_team_id", "home_score", "away_score"),
                   side("away_team_id", "home_team_id", "away_score", "home_score")])

# Match statistics for the team itself, and for the opponent (= shots the team allowed)
stats = stats.rename(columns={"match_id": "src_match_id"})
own = stats[["src_match_id", "team_id", "total_shots", "possession_pct"]]
opp = stats[["src_match_id", "team_id", "total_shots"]].rename(
    columns={"team_id": "opp_id", "total_shots": "shots_against"})
games = (games.merge(own, on=["src_match_id", "team_id"], how="left")
              .merge(opp, on=["src_match_id", "opp_id"], how="left")
              .sort_values(["date", "match_id"], kind="stable").reset_index(drop=True))

# ---------------------------------------------------------------- 3. Pre-match rolling form (no leakage)
# shift() drops the current match, expanding().mean() averages every earlier match at this
# World Cup. A team's first match has no history, so its form values are set to 0.
def prior_mean(col):
    return games.groupby("team_id")[col].transform(lambda s: s.shift().expanding().mean())

games["avg_gf"] = prior_mean("goals_scored")
games["avg_ga"] = prior_mean("goals_conceded")
games["avg_shots"] = prior_mean("total_shots")
games["avg_shots_against"] = prior_mean("shots_against")
games["avg_possession"] = prior_mean("possession_pct")
games["rest_team"] = games.groupby("team_id")["date"].diff().dt.days

# Opponent's form: look up the opponent's own row for the same match
opp_form = games[["match_id", "team_id", "avg_ga", "avg_shots_against"]].rename(
    columns={"team_id": "opp_id", "avg_ga": "avg_ga_opp", "avg_shots_against": "opp_avg_shots_against"})
games = games.merge(opp_form, on=["match_id", "opp_id"], how="left")

# ---------------------------------------------------------------- 4. Assemble the modelling table
df = pd.DataFrame({
    "match_id": games.match_id, "date": games.date.dt.date,
    "team": games.team_id.map(teams), "opponent": games.opp_id.map(teams),
    "goals_scored": games.goals_scored,
    "avg_gf_team": games.avg_gf,
    "avg_ga_opp": games.avg_ga_opp,
    "stage": games.stage,
    "rest_team": games.rest_team,
    "avg_shots_team": games.avg_shots,
    "opp_avg_shots_against": games.opp_avg_shots_against,
    "avg_possession_team": games.avg_possession,
    "team_host": games.team_id.map(teams).isin(HOSTS).astype(int),
})

# ---------------------------------------------------------------- 5. Cleaning decisions (document in report)
# First match of each team (48 rows = 23%): no earlier World Cup matches exist, so every
# rolling average and rest_team are set to 0. This is structural, not imputed from other rows.
FORM = ["avg_gf_team", "avg_ga_opp", "rest_team", "avg_shots_team",
        "opp_avg_shots_against", "avg_possession_team"]
df[FORM] = df[FORM].fillna(0)
df["rest_team"] = df.rest_team.astype(int)
df[FORM] = df[FORM].round(4)
# Stage: src_matches.csv uses 6 = 3rd place and 7 = Final. Both final-weekend matches are
# coded 6 ("Final" round) so the scale is Group=1 ... Final=6. Affects only match 104 (2 rows).
df["stage"] = df.stage.clip(upper=6)

# Sort by match; within a match, Team A of the fixture list comes first (same order as LR 2.1)
team_a = fixtures.set_index("match_id")["team_a"]
df["_b_first"] = (df.team != df.match_id.map(team_a)).astype(int)
df = df.sort_values(["match_id", "_b_first"]).drop(columns="_b_first").reset_index(drop=True)
df.insert(0, "row_id", range(1, len(df) + 1))
assert len(df) == 208 and df.isna().sum().sum() == 0
df.to_csv(OUT / "wc2026_lr22_v2.csv", index=False)
print(f"Saved {len(df)} rows -> {OUT / 'wc2026_lr22_v2.csv'}")
print(df.head())
