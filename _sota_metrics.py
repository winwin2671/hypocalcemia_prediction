"""State-of-the-art evaluation metrics for clinical prediction models.
Pure helpers; will be inlined into the notebook.

- DeLong (Sun & Xu 2014) variance and CI for the AUC, plus a paired test
  for comparing two AUCs on the same subjects (DeLong et al. 1988).
- Calibration-in-the-large, calibration slope, and the Integrated
  Calibration Index (ICI; Austin et al. 2020) via LOWESS.
"""
import numpy as np
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess


# ---------- DeLong variance / CI ----------
def _delong_components(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    m, n = len(pos), len(neg)
    # structural components
    v10 = np.array([np.sum(pos[i] > neg) + 0.5 * np.sum(pos[i] == neg) for i in range(m)]) / n
    v01 = np.array([np.sum(neg[i] < pos) + 0.5 * np.sum(neg[i] == pos) for i in range(n)]) / m
    return v10, v01, m, n


def delong_auc_ci(y_true, y_score, alpha=0.05):
    """AUC with DeLong 95% CI (fast O(m*n) version; Sun & Xu 2014)."""
    v10, v01, m, n = _delong_components(y_true, y_score)
    auc = v10.mean()
    var = v10.var(ddof=1) / m + v01.var(ddof=1) / n
    z = norm.ppf(1 - alpha / 2)
    return float(auc), float(auc - z * np.sqrt(var)), float(auc + z * np.sqrt(var))


def delong_paired_test(y_true, s1, s2):
    """Two-sided p-value for H0: AUC1 == AUC2 on the same subjects."""
    v10_1, v01_1, m, n = _delong_components(y_true, s1)
    v10_2, v01_2, _, _ = _delong_components(y_true, s2)
    # covariance of the two AUCs
    cov_10 = np.cov(v10_1, v10_2, ddof=1) / m
    cov_01 = np.cov(v01_1, v01_2, ddof=1) / n
    var_diff = cov_10[0, 0] + cov_01[0, 0] - 2 * (cov_10[0, 1] + cov_01[0, 1])
    lrc = (delong_auc_ci(y_true, s1)[0] - delong_auc_ci(y_true, s2)[0]) / np.sqrt(max(var_diff, 1e-30))
    return float(2 * norm.sf(abs(lrc)))


# ---------- Calibration: intercept, slope, ICI ----------
def calibration_metrics(y_true, p):
    """Recalibration-in-the-large (intercept), slope, and ICI (Austin 2020).
    Ideal calibration: intercept=0, slope=1, ICI≈0."""
    y_true = np.asarray(y_true)
    p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p))
    try:
        res = sm.Logit(y_true, sm.add_constant(logit)).fit(disp=0, maxiter=200)
        cal_int, cal_slope = float(res.params[0]), float(res.params[1])
    except Exception:
        cal_int, cal_slope = np.nan, np.nan
    sm_at_p = lowess(y_true, p, xvals=p, frac=2 / 3, return_sorted=False)
    ici = float(np.mean(np.abs(sm_at_p - p)))
    return cal_int, cal_slope, ici
