# Teams update draft

Adapt before posting. This describes repository status, not a claim of independent authorship. It has not been sent.

Hi everyone, the Regression 2.1 folder is updated on GitHub. The updated code uses chronological validation and fits scaling inside the tuning folds. The script and notebook run successfully, and the existing data checks plus the new validation tests pass.

Ridge has the lowest development MAE among the linear models (1.503 goals, compared with 1.844 for OLS). Random forest scores slightly better as a benchmark. The knockout results are labelled retrospective because those matches were already examined earlier.

There are still a few things to resolve: the original data sources, the unusual rest-day gaps, and the predictor overlap with 2.2. Could the person doing 2.2 share their final eight variables? Could someone also review the model comparison and the limitations before we finalise the report?

Folder: https://github.com/Shrizan01/WC2026-objective2/tree/main/Linear%20regression%202.1

## Follow-up after a real review

Reply to the teammate's actual feedback, describe what you changed or why you kept the approach, and link the relevant commit. Do not create retrospective meeting minutes or claim reviews that did not occur.
