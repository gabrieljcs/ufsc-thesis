"""Post-review fractional-response analysis; frozen outcomes are never modified."""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import patsy
from scipy.special import expit
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.cov_struct import Independence
from statsmodels.genmod.families import Binomial
from statsmodels.stats.outliers_influence import variance_inflation_factor


def summarize(data: pd.DataFrame, *, seed: int = 20260929, draws: int = 2000) -> dict:
    """Prompt-cluster bootstrap, retaining all selected rows of sampled prompts."""
    score = data.strongreject_score.to_numpy(float)
    if not len(score) or not np.isfinite(score).all() or ((score < 0) | (score > 1)).any():
        raise ValueError('Expected nonempty finite scores in [0,1]')
    grouped = data.assign(positive=(score > 0).astype(int)).groupby('prompt_id').agg(
        score_sum=('strongreject_score','sum'), n=('strongreject_score','size'), positive=('positive','sum'))
    a = grouped.to_numpy(float)
    rng = np.random.default_rng(seed)
    totals = a[rng.integers(0, len(a), size=(draws,len(a)))].sum(axis=1)
    means = totals[:,0]/totals[:,1]
    positives = score[score > 0]
    return dict(n=len(score), n_prompts=len(a), asr=float((score>0).mean()),
                mean_score=float(score.mean()), mean_ci_low=float(np.quantile(means,.025)),
                mean_ci_high=float(np.quantile(means,.975)), positive_n=len(positives),
                positive_mean=float(positives.mean()) if len(positives) else None)


def design(data: pd.DataFrame, predictors: tuple[str,...]):
    """Standardize within the complete estimation cohort, weighted by its rows."""
    data = data.copy()
    scaling = []
    for p in predictors:
        mean, sd = float(data[p].mean()), float(data[p].std(ddof=0))
        if not np.isfinite(sd) or sd <= 0: raise ValueError(f'Cannot standardize {p}')
        data[p+'_z'] = (data[p]-mean)/sd
        scaling.append(dict(predictor=p,mean=mean,sd=sd))
    formula = ' + '.join([p+'_z' for p in predictors]+['C(model)','C(attack_language)'])
    x = patsy.dmatrix(formula, data, return_type='dataframe')
    if np.linalg.matrix_rank(x) != x.shape[1]: raise ValueError('Rank-deficient design')
    return data, x, scaling


def fit_outcomes(data: pd.DataFrame, predictors: tuple[str,...], name: str):
    """Quasi-binomial mean equations; fractional scores are NOT Bernoulli trials."""
    data, x, scaling = design(data, predictors)
    diagnostics, effects = [], []
    trigger = False
    for outcome in ('binary','score','positive_score'):
        keep = data.strongreject_score > 0 if outcome=='positive_score' else np.ones(len(data),dtype=bool)
        xx = x.loc[keep]; subset = data.loc[keep]
        y = (subset.strongreject_score > 0).astype(float) if outcome=='binary' else subset.strongreject_score.astype(float)
        if np.linalg.matrix_rank(xx) != xx.shape[1]: raise ValueError(f'Rank-deficient {name}/{outcome}')
        vifs = {p: float(variance_inflation_factor(xx.to_numpy(),xx.columns.get_loc(p+'_z'))) for p in predictors}
        trigger = trigger or max(vifs.values()) > 5
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = GEE(y,xx,groups=subset.prompt_id,family=Binomial(),cov_struct=Independence()).fit(maxiter=500,ctol=1e-8,cov_type='robust')
        cov = np.asarray(result.cov_params())
        eig = float(np.linalg.eigvalsh(cov).min())
        valid = bool(result.converged and np.isfinite(result.params).all() and np.isfinite(cov).all() and eig>0)
        diagnostics.append(dict(fit=name,outcome=outcome,n=len(y),n_prompts=subset.prompt_id.nunique(),
            n_languages=subset.attack_language.nunique(),n_models=subset.model.nunique(),converged=bool(result.converged),
            covariance_min_eigenvalue=eig,valid=valid,condition_number=float(np.linalg.cond(xx)),
            iterations=len(result.fit_history.get('params',[])),warnings=' | '.join(str(w.message) for w in caught),
            substantive_vif=vifs,scaling=scaling,coding='Treatment indicators, intercept; cohort-row z scores (ddof=0); positive subset retains full-cohort scaling',
            covariance='prompt-clustered sandwich; working independence'))
        if not valid: raise RuntimeError(f'Invalid fit {name}/{outcome}: {diagnostics[-1]}')
        mu = expit(np.asarray(xx) @ np.asarray(result.params)); w = mu*(1-mu)
        for p in predictors:
            j = xx.columns.get_loc(p+'_z')
            b, se = float(result.params.iloc[j]), float(result.bse.iloc[j])
            ame = b*w.mean()
            grad = b*np.mean((w*(1-2*mu))[:,None]*np.asarray(xx),axis=0)
            grad[j] += w.mean()
            ame_se = float(np.sqrt(grad@cov@grad))
            effects.append(dict(fit=name,outcome=outcome,predictor=p,n=len(y),estimate=b,std_error=se,
                ci_low=b-1.96*se,ci_high=b+1.96*se,p_value=float(result.pvalues.iloc[j]),
                average_derivative=float(ame),derivative_ci_low=float(ame-1.96*ame_se),derivative_ci_high=float(ame+1.96*ame_se),
                interpretation='logit of conditional mean; not an event odds ratio for graded outcomes'))
    return effects, diagnostics, trigger


CLUSTERS = ('prompt_id', 'cell', 'twoway')


def cluster_groups(data: pd.DataFrame, cluster: str) -> pd.Series:
    """Grouping variable for the robust sandwich covariance.

    'prompt_id' groups the translations and models of one source prompt.
    'cell' groups a model--attack-language cell. Distance and SPEC vary between
    cells, so prompt clusters cut across the variation being tested and cannot
    absorb within-cell residual dependence.
    """
    if cluster == 'prompt_id':
        return data.prompt_id.astype(str)
    if cluster == 'cell':
        return data.model.astype(str) + '|' + data.attack_language.astype(str)
    raise ValueError(f'Unknown cluster {cluster!r}; expected prompt_id or cell')


def combine_twoway_covariance(v_a, v_b, v_intersection):
    """Cameron--Gelbach--Miller two-way covariance: V_a + V_b - V_intersection.

    Each row is a unique (prompt, model, language) combination, so the
    intersection of the two clusterings is the row level and its covariance is
    the heteroskedasticity-robust one. The sum is not guaranteed positive
    definite, so the caller is told whether it came out usable.
    """
    v = np.asarray(v_a, float) + np.asarray(v_b, float) - np.asarray(v_intersection, float)
    symmetric = (v + v.T) / 2
    return v, bool(np.isfinite(v).all() and np.linalg.eigvalsh(symmetric).min() > 0)


def fit_cluster_variants(data: pd.DataFrame, predictors: tuple[str, ...], name: str) -> list[dict]:
    """Refit the graded mean equation under prompt, cell and two-way clustering.

    Working independence means the grouping affects only the sandwich, so the
    coefficients are identical by construction and the comparison isolates
    inference. Reported for the substantive predictors only.
    """
    data, x, scaling = design(data, predictors)
    y = data.strongreject_score.astype(float)
    fits = {}
    for cluster in ('prompt_id', 'cell'):
        groups = cluster_groups(data, cluster)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = GEE(y, x, groups=groups, family=Binomial(),
                         cov_struct=Independence()).fit(maxiter=500, ctol=1e-8, cov_type='robust')
        if not (result.converged and np.isfinite(result.params).all()):
            raise RuntimeError(f'Invalid {name} fit under {cluster} clustering')
        fits[cluster] = (result, np.asarray(result.cov_params()), groups.nunique())
    # Row-level (intersection) covariance for the two-way combination.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        row_result = GEE(y, x, groups=np.arange(len(y)), family=Binomial(),
                         cov_struct=Independence()).fit(maxiter=500, ctol=1e-8, cov_type='robust')
    twoway_cov, twoway_ok = combine_twoway_covariance(
        fits['prompt_id'][1], fits['cell'][1], np.asarray(row_result.cov_params()))

    reference = fits['prompt_id'][0]
    out: list[dict] = []
    for cluster in CLUSTERS:
        if cluster == 'twoway':
            cov, n_clusters, usable = twoway_cov, None, twoway_ok
        else:
            result, cov, n_clusters = fits[cluster]
            usable = True
        for p in predictors:
            j = x.columns.get_loc(p + '_z')
            b = float(reference.params.iloc[j])
            se = float(np.sqrt(cov[j, j])) if usable and cov[j, j] > 0 else float('nan')
            out.append(dict(fit=name, cluster=cluster, predictor=p, n=int(len(y)),
                            n_clusters=n_clusters, estimate=b, std_error=se,
                            ci_low=b - 1.96 * se, ci_high=b + 1.96 * se,
                            covers_zero=bool(not np.isfinite(se) or (b - 1.96 * se) * (b + 1.96 * se) <= 0),
                            positive_definite=usable, scaling=scaling,
                            covariance=f'{cluster} robust sandwich; working independence'))
    return out
