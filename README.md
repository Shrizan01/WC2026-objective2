# ⚽ FIFA World Cup 2026 — Objective 2: Predicting Goals

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Unit](https://img.shields.io/badge/HIT140-Data%20Science-green)
![Group](https://img.shields.io/badge/Group-27-orange)

**HIT140 Data Science · Assessment 3: Group Project Report · Charles Darwin University**

Two linear regression analyses that predict scoring at the FIFA World Cup 2026 using only information known **before kick-off**.

---

## 👥 Group 27

| Name | Student ID | Contribution |
|---|---|---|
| Sumit Maharjan | <STUDENT ID> | Linear Regression 2.2: goals scored by a team |
| Suraj <SURNAME> | <STUDENT ID> | Linear Regression 2.1: match goal difference |
| <FULL NAME> | <STUDENT ID> | <CONTRIBUTION> |
| <FULL NAME> | <STUDENT ID> | <CONTRIBUTION> |

---

## 📊 Analyses

| | [Linear Regression 2.1](Linear%20regression%202.1/README.md) | [Linear Regression 2.2](Linear%20regression%202.2/README.md) |
|---|---|---|
| **Question** | How large is the goal difference between two teams? | How many goals will a team score? |
| **Unit of analysis** | One row per match (104 rows) | One row per team per match (208 rows) |
| **Target** | `goal_diff` (Team A − Team B) | `goals_scored` |
| **Predictors** | 8 pre-match variables (team strength, form, experience, rest, host, stage) | 8 pre-match variables (attacking form, opponent defence, possession, rest, host, stage) |
| **Models** | Mean baseline, OLS, Ridge, Lasso, Huber, Random Forest (benchmark) | OLS, Ridge, Lasso, ElasticNet |
| **Validation** | Chronological validation within the group stage; retrospective knockout test | 80/20 split, 5-fold cross-validation, group stage → knockout check |

---

## 🔗 Shared variables

At most **4** of the 8 predictors may be shared between the two analyses. Counted conservatively, **3** overlap:

| Concept | LR 2.1 | LR 2.2 |
|---|---|---|
| Tournament stage | `stage` (1–7) | `stage` (1–7, identical coding) |
| Rest days | `rest_diff` (Team A − Team B, capped ±7) | `rest_team` (days since the team's previous World Cup match) |
| Host nation | `host_diff` (+1 / −1 / 0) | `team_host` (1 / 0) |

---

## 📁 Repository structure

Both analyses follow the same reproducible layout:

```
WC2026-objective2/
├── Linear regression 2.1/
│   ├── data/raw/          original input files
│   ├── data/processed/    modelling dataset
│   ├── src/               build + verification scripts
│   ├── notebooks/         analysis notebook
│   ├── figures/           exported figures and results tables
│   └── README.md
├── Linear regression 2.2/
│   └── (same layout)
└── README.md
```

---

## ▶️ How to run

Each analysis is self-contained. From the repository root:

```bash
cd "Linear regression 2.1"   # or "Linear regression 2.2"
bash setup.sh                # creates .venv, installs packages, builds and checks the data
source .venv/bin/activate
jupyter lab                  # open the notebook in notebooks/
```

Each folder's README has full details, data decisions and limitations.

---

## 🗂️ Data sources

- **Match results and history:** `results.csv` (international results)
- **FIFA rankings:** pre-tournament ranking, June 2026
- **Match statistics:** shots and possession per team per match (fifa.com, sofascore.com)
- **Fixture list:** shared match IDs, local dates and stages

All data wrangling, analysis and visualisation is done in **Python** (pandas, NumPy, SciPy, statsmodels, scikit-learn, matplotlib, seaborn).
