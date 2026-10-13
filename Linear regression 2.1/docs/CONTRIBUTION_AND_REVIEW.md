# Regression 2.1 contribution and review record

## Repository work completed

These are verified repository changes, not proof that a student independently authored or understands every change. Coding and documentation were AI-assisted. Student review, learning and decisions should be recorded honestly below.

| Commit | Recorded work |
|---|---|
| `0993eb8` | Initial supplied Regression 2.1 baseline |
| `20cb4c8` | Organised the analysis into its own folder |
| `c509785` | Chronological validation, executable analysis, refreshed outputs, data limitations and checks |

Validation at `c509785`: 13 internal data checks and two validation tests passed; script and notebook executed successfully. Checks establish internal consistency, not external data authenticity.

## Student contribution — complete after doing the work

- Name and student ID:
- Code sections I ran and can explain:
- Data sources I personally checked, with links and dates:
- Decisions I made and why:
- Errors I found or corrected:
- Feedback I gave another member:
- Feedback I received and changes made:
- Actual Teams post/reply links:

## Peer review — pending, not yet performed

Ask a teammate to check:
1. Eight predictors and their availability before each match.
2. Predictor overlap against their Regression 2.2 list.
3. Date splits and scaling within tuning.
4. Whether the figures and written claims match the result files.
5. Whether the remaining data limitations are clearly disclosed.

Record reviewer, date, comments, response and Teams link only after the review occurs.

## Questions to be able to answer

- Why is goal difference signed, and how is Team A chosen?
- What separates an explanatory variable from metadata and the target?
- Why does random splitting create a problem with match-history features?
- Why fit the scaler separately in each training fold?
- What does Ridge alpha control, and why compare it with OLS?
- Why is random forest a benchmark rather than the final task model?
- What do MAE, RMSE and R² measure?
- Why can't stage's effect be learned from group-stage-only training?
- Why aren't the knockout results an untouched test?
- What can our internal checks not establish about source quality?

## AI declaration preparation

Use the provided official form; this document does not replace it. Identify actual tools used, which coding/planning/proofreading tasks they assisted with, what you personally checked, what errors or limitations you identified, and what you learned. Include conversation-history links where applicable. Do not invent independent authorship, verification, understanding or reflection. All members must sign the required form.
