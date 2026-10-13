"""Run reproducible, date-blocked LR 2.1 comparisons. Outputs are retrospective."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, HuberRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ['rank_diff', 'elo_diff', 'winrate10_diff', 'wc_exp_diff',
            'tourn_pts_diff', 'host_diff', 'rest_diff', 'stage']

def date_splits(dates, n_splits=4):
    """Expanding windows; all matches on a date stay together."""
    dates = np.asarray(dates)
    blocks = np.array_split(np.unique(dates), n_splits + 1)
    for i in range(1, len(blocks)):
        train = np.flatnonzero(np.isin(dates, np.concatenate(blocks[:i])))
        test = np.flatnonzero(np.isin(dates, blocks[i]))
        assert dates[train].max() < dates[test].min()
        yield train, test

def estimator(name, dates):
    models = {'Mean baseline': DummyRegressor(), 'OLS': LinearRegression(),
              'Ridge': Ridge(), 'Lasso': Lasso(max_iter=50000),
              'Huber': HuberRegressor(max_iter=3000),
              'Random forest': RandomForestRegressor(n_estimators=200,
                   min_samples_leaf=5, random_state=42, n_jobs=1)}
    pipe = make_pipeline(StandardScaler(), models[name])
    if name in ('Ridge', 'Lasso'):
        # Scaling is refitted inside each chronological inner training fold.
        return GridSearchCV(pipe, {name.lower() + '__alpha': np.logspace(-3, 3, 13)},
                            cv=list(date_splits(dates, 3)),
                            scoring='neg_mean_absolute_error', n_jobs=1,
                            error_score='raise')
    return pipe

def metrics(y, pred):
    return dict(MAE=mean_absolute_error(y, pred),
                RMSE=np.sqrt(mean_squared_error(y, pred)), R2=r2_score(y, pred))

def run():
    df = pd.read_csv(ROOT/'data/processed/wc2026_lr21_refined.csv', parse_dates=['date'])
    df = df.sort_values(['date', 'match_id']).reset_index(drop=True)
    assert len(df) == 104 and len(FEATURES) == 8 and not df[FEATURES].isna().any().any()
    # Previously inspected knockout results cannot become an untouched test set.
    dev = df[df.stage == 1].reset_index(drop=True)
    later = df[df.stage != 1].reset_index(drop=True)
    names = ['Mean baseline', 'OLS', 'Ridge', 'Lasso', 'Huber', 'Random forest']
    rows, predictions, audit = [], [], []
    for fold, (tr, va) in enumerate(date_splits(dev.date, 4), 1):
        train, valid = dev.iloc[tr], dev.iloc[va]
        audit.append(dict(fold=fold, train_n=len(tr), validation_n=len(va),
                          train_end=str(train.date.max().date()),
                          validation_start=str(valid.date.min().date()),
                          validation_end=str(valid.date.max().date())))
        for name in names:
            model = estimator(name, train.date)
            model.fit(train[FEATURES], train.goal_diff)
            pred = model.predict(valid[FEATURES])
            rows.append(dict(model=name, fold=fold, **metrics(valid.goal_diff, pred),
                             parameters=str(getattr(model, 'best_params_', {}))))
            for mid, y, p in zip(valid.match_id, valid.goal_diff, pred):
                predictions.append(dict(model=name, fold=fold, match_id=mid, actual=y, predicted=p))
    scores = pd.DataFrame(rows)
    summary = scores.groupby('model')[['MAE', 'RMSE', 'R2']].mean().sort_values('MAE')
    # Task requires a linear model: nonlinear forest is a benchmark only.
    linear_names = ['OLS', 'Ridge', 'Lasso', 'Huber']
    chosen = summary.loc[linear_names].MAE.idxmin()
    model = estimator(chosen, dev.date)
    model.fit(dev[FEATURES], dev.goal_diff)
    pred = model.predict(later[FEATURES])
    later_metrics = metrics(later.goal_diff, pred)
    out = ROOT/'figures'; out.mkdir(exist_ok=True)
    scores.to_csv(out/'cv_fold_metrics.csv', index=False)
    summary.to_csv(out/'table_model_comparison.csv')
    pd.DataFrame(audit).to_csv(out/'validation_splits.csv', index=False)
    pd.DataFrame(predictions).to_csv(out/'cv_predictions.csv', index=False)
    later.assign(predicted_goal_diff=pred)[['match_id','date','goal_diff','predicted_goal_diff']].to_csv(out/'retrospective_predictions.csv',index=False)
    # Development-only explanatory diagnostics. HC3 does not correct repeated-team dependence.
    ols = sm.OLS(dev.goal_diff, sm.add_constant(dev[[c for c in FEATURES if dev[c].nunique() > 1]], has_constant='add')).fit(cov_type='HC3')
    (out/'ols_development_summary.txt').write_text(ols.summary().as_text())
    plt.rcParams.update({'figure.dpi': 130, 'savefig.dpi': 300, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(1,2,figsize=(11,4))
    ax[0].hist(dev.goal_diff,bins=np.arange(-7.5,8.5),color='#287c8e')
    ax[0].set(xlabel='Goal difference (A − B)',ylabel='Matches',title='Development sample: goal differences')
    im=ax[1].imshow(dev[FEATURES+['goal_diff']].corr(),vmin=-1,vmax=1,cmap='RdBu_r')
    ax[1].set_xticks(range(9),FEATURES+['goal_diff'],rotation=90,fontsize=7)
    ax[1].set_yticks(range(9),FEATURES+['goal_diff'],fontsize=7)
    ax[1].set_title('Group-stage correlation\nBlank: constant stage (undefined)', fontsize=10);fig.colorbar(im,ax=ax[1]);fig.tight_layout();fig.savefig(out/'fig1_eda.png');plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(10,4))
    ax[0].scatter(ols.fittedvalues,ols.resid,alpha=.7);ax[0].axhline(0,color='black',lw=1)
    ax[0].set(xlabel='Fitted goal difference',ylabel='Residual',title='Development OLS residuals')
    sm.qqplot(ols.resid,line='s',ax=ax[1]);ax[1].set_title('Development residual Q–Q plot')
    fig.tight_layout();fig.savefig(out/'fig2_ols_diagnostics.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4))
    summary.MAE.plot.barh(ax=ax,color='#287c8e');ax.invert_yaxis()
    ax.set(xlabel='Mean chronological fold MAE (goals; lower is better)',ylabel='',title='Earlier matches predict later group-stage matches')
    fig.tight_layout();fig.savefig(out/'fig3_cv_comparison.png');plt.close(fig)
    # Standardized selected-model coefficients, fitted only to development data.
    fitted = model.best_estimator_ if hasattr(model,'best_estimator_') else model
    coef=pd.Series(fitted[-1].coef_,index=FEATURES)
    fig,ax=plt.subplots(figsize=(8,4));coef.sort_values().plot.barh(ax=ax,color='#287c8e')
    ax.axvline(0,color='black',lw=1);ax.set(xlabel='Goals per one development-sample SD',title=f'{chosen}: conditional coefficients, not causal effects')
    fig.tight_layout();fig.savefig(out/'fig4_coefficients.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5));ax.scatter(later.goal_diff,pred,alpha=.7)
    lim=[min(later.goal_diff.min(),pred.min())-1,max(later.goal_diff.max(),pred.max())+1]
    ax.plot(lim,lim,'k--');ax.set(xlabel='Actual goal difference',ylabel='Predicted goal difference',title=f'{chosen}: retrospective knockout evaluation',xlim=lim,ylim=lim)
    fig.tight_layout();fig.savefig(out/'fig6_pred_vs_actual.png');plt.close(fig)
    result=dict(selected_linear_model=chosen, selection_rule='Lowest mean chronological development-fold MAE among linear candidates',development_rows=len(dev),retrospective_rows=len(later),retrospective_metrics=later_metrics,parameters=getattr(model,'best_params_',{}),limitations=['Knockout outcomes were inspected in the previous analysis: not an untouched test.', 'Features update with results available before each match; this is not a forecast of the entire knockout stage made at its start.', 'Stage is constant in group-stage training; its effect cannot be estimated.', 'Source provenance and completeness remain unverified; rest differences remain capped at ±7.', 'Small chronological folds and repeated teams limit precision; no significance claim.'])
    (out/'results_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(summary.round(3).to_string());print(json.dumps(result,indent=2))
    return summary, result

if __name__ == '__main__':
    run()
