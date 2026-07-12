"""Development script for the hypocalcemia prediction analysis (7-step plan).
Mirrors what will go into model_analysis.ipynb. Run top-to-bottom."""
import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.optimize import minimize
from scipy.stats import mannwhitneyu, shapiro
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from sklearn.calibration import calibration_curve

np.random.seed(42)
os.makedirs("figures", exist_ok=True)
RAW = "Thyroid_Sx_research_(ReadyToClean).csv"
PREDICTORS = ["Pre-op PTH", "Pre-op Ca", "Vit D(preop)"]
THRESH = 2.1  # mmol/L


# ============================================================
# Firth's bias-reduced logistic regression (no external deps)
# Refs: Firth 1993 (Biometrika); Heinze & Schemper 2002 (Stat Med)
# ============================================================
class FirthLogisticRegression:
    """Penalizes the log-likelihood with 0.5*log|I(beta)| (Jeffreys prior),
    removing first-order finite-sample bias and handling separation."""

    def __init__(self, max_iter=500, tol=1e-10):
        self.max_iter = max_iter
        self.tol = tol

    @staticmethod
    def _design(X):
        X = np.asarray(X, dtype=float)
        return np.column_stack([np.ones(len(X)), X])

    def _neg_penalized_loglik(self, beta, Xc, y):
        eta = np.clip(Xc @ beta, -30, 30)
        p = expit(eta)
        ll = np.sum(y * eta - np.logaddexp(0.0, eta))  # stable log-likelihood
        W = np.clip(p * (1 - p), 1e-12, None)
        XtWX = (Xc * W[:, None]).T @ Xc + np.eye(Xc.shape[1]) * 1e-10
        sign, logdet = np.linalg.slogdet(XtWX)
        if sign <= 0:
            logdet = np.log(max(np.linalg.det(XtWX), 1e-300))
        return -(ll + 0.5 * logdet)

    def fit(self, X, y):
        Xc = self._design(X)
        y = np.asarray(y, dtype=float)
        beta0 = np.zeros(Xc.shape[1])
        res = minimize(self._neg_penalized_loglik, beta0, args=(Xc, y),
                       method="BFGS", options={"maxiter": self.max_iter, "gtol": self.tol})
        self.beta_ = res.x
        self.converged_ = bool(res.success)
        self.intercept_ = self.beta_[0]
        self.coef_ = self.beta_[1:]
        return self

    def predict_proba(self, X):
        Xc = self._design(X)
        p1 = expit(np.clip(Xc @ self.beta_, -30, 30))
        return np.column_stack([1 - p1, p1])


# ============================================================
# Model wrappers: each handles its own standardization so the
# bootstrap re-standardizes per resample (proper optimism).
# ============================================================
class StandardLR:
    def fit(self, X, y):
        self.mu = X.mean(0); self.sd = X.std(0)
        Xs = (X - self.mu) / self.sd
        Xc = sm.add_constant(Xs)
        try:
            self.res = sm.Logit(y, Xc).fit(disp=0, maxiter=200)
        except Exception:
            self.res = sm.Logit(y, Xc).fit(disp=0, method="bfgs", maxiter=1000)
        self.coef_std = np.asarray(self.res.params[1:])
        self.intercept = float(self.res.params[0])
        return self

    def predict_proba(self, X):
        Xs = (X - self.mu) / self.sd
        Xc = sm.add_constant(Xs)
        p = np.asarray(self.res.predict(Xc))
        return np.column_stack([1 - p, p])


class FirthLR:
    def fit(self, X, y):
        self.mu = X.mean(0); self.sd = X.std(0)
        Xs = (X - self.mu) / self.sd
        self.m = FirthLogisticRegression().fit(Xs, y)
        self.coef_std = self.m.coef_.copy()
        self.intercept = float(self.m.intercept_)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba((X - self.mu) / self.sd)


class LassoLR:
    def fit(self, X, y):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        self.pipe = Pipeline([
            ("sc", StandardScaler()),
            ("lr", LogisticRegressionCV(penalty="l1", solver="liblinear", cv=cv,
                                        scoring="roc_auc", max_iter=5000,
                                        random_state=42, Cs=20)),
        ]).fit(X, y)
        lr = self.pipe.named_steps["lr"]
        self.C_ = float(lr.C_[0])
        self.coef_std = lr.coef_[0].copy()
        self.intercept = float(lr.intercept_[0])
        return self

    def predict_proba(self, X):
        return self.pipe.predict_proba(X)


MODELS = {"Standard LR": StandardLR, "Firth LR": FirthLR, "LASSO LR": LassoLR}


# ============================================================
# Step 1: Data preparation
# ============================================================
print("=" * 70); print("STEP 1: DATA PREPARATION"); print("=" * 70)
df = pd.read_csv(RAW)
df.columns = [c.strip() for c in df.columns]
df = df[df["case No."].astype(str).str.strip().str.fullmatch(r"\d+")].copy()
df["case No."] = df["case No."].astype(int)
df = df.drop(columns=["Name", "Unnamed: 25"], errors="ignore").reset_index(drop=True)

for col in PREDICTORS + ["Postop Ca", "Age", "BMI"]:
    df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors="coerce")

df["Hypocalcemia"] = (df["Postop Ca"] < THRESH).astype(int)
df["Sex"] = df["Sex"].astype(str).str.strip().str.title()

n = len(df)
n_ev = int(df["Hypocalcemia"].sum())
print(f"N = {n}")
print(f"Outcome: Postop Ca < {THRESH} mmol/L  ->  events = {n_ev} ({n_ev/n*100:.1f}%), non-events = {n-n_ev}")
print(f"Missing in predictors: {[int(df[c].isna().sum()) for c in PREDICTORS]}")
print(f"EPV (3 predictors) = {n_ev/3:.1f}")
print("\nPredictor distributions (mean +/- SD, median [IQR]):")
for c in PREDICTORS:
    s = df[c]
    print(f"  {c:14s}: {s.mean():.2f} +/- {s.std():.2f}   median {s.median():.2f} [{s.quantile(.25):.2f}-{s.quantile(.75):.2f}]   Shapiro p={shapiro(s).pvalue:.3f}")


# ============================================================
# Step 2 & 3: Descriptive table + group comparison
# ============================================================
print("\n" + "=" * 70); print("STEPS 2 & 3: DESCRIPTIVE + GROUP COMPARISON"); print("=" * 70)
rows = []
g0 = df[df.Hypocalcemia == 0]; g1 = df[df.Hypocalcemia == 1]
for label, col, cont in [("Age (yr)", "Age", True), ("BMI (kg/m2)", "BMI", True),
                         ("Pre-op PTH (pg/mL)", "Pre-op PTH", True),
                         ("Pre-op Ca (mmol/L)", "Pre-op Ca", True),
                         ("Vit D (ng/mL)", "Vit D(preop)", True)]:
    a = g0[col]; b = g1[col]
    p = shapiro(df[col]).pvalue
    if p < 0.05:
        fmt = lambda s: f"{s.median():.2f} [{s.quantile(.25):.2f}-{s.quantile(.75):.2f}]"
        stat = mannwhitneyu(a, b, alternative="two-sided").pvalue
        test = "Mann-Whitney"
    else:
        fmt = lambda s: f"{s.mean():.2f} +/- {s.std():.2f}"
        stat = mannwhitneyu(a, b, alternative="two-sided").pvalue  # spec: Mann-Whitney
        test = "Mann-Whitney*"
    rows.append([label, fmt(a), fmt(b), f"{stat:.3f}", test])
# Sex (categorical) - Fisher exact
from scipy.stats import fisher_exact
tab = pd.crosstab(df["Sex"], df["Hypocalcemia"])
fe = fisher_exact(tab) if tab.shape == (2, 2) else (np.nan, np.nan)
def cat(s):
    f = (s == "Female").sum()
    return f"F {f} ({f/len(s)*100:.0f}%) / M {len(s)-f} ({(len(s)-f)/len(s)*100:.0f}%)"
rows.append(["Sex", cat(g0.Sex), cat(g1.Sex), f"{fe[1]:.3f}", "Fisher"])

table1 = pd.DataFrame(rows, columns=["Variable", f"No hypocalcemia (n={n-n_ev})", f"Hypocalcemia (n={n_ev})", "p", "test"])
print(table1.to_string(index=False))
print("\n(* normally distributed; Mann-Whitney used per analysis plan)")


# ============================================================
# Step 4: Build the three models (apparent fit)
# ============================================================
print("\n" + "=" * 70); print("STEP 4: MODEL BUILDING (apparent fit, standardized predictors)"); print("=" * 70)
y = df["Hypocalcemia"].values
X = df[PREDICTORS].values.astype(float)

fitted = {}
for name, Cls in MODELS.items():
    m = Cls().fit(X, y)
    fitted[name] = m
    print(f"\n{name}  (coef on standardized scale, i.e. per 1-SD)")
    print(f"  intercept={m.intercept:.3f}")
    for p_, c in zip(PREDICTORS, m.coef_std):
        print(f"  {p_:14s}: beta_std={c:+.3f}   OR(per-SD)={np.exp(c):.3f}")
    if hasattr(m, "C_"):
        print(f"  LASSO chosen C (inverse lambda) = {m.C_:.4g}; "
              f"retained {int(np.sum(m.coef_std != 0))}/{len(PREDICTORS)} predictors")
    proba = m.predict_proba(X)[:, 1]
    print(f"  apparent AUC = {roc_auc_score(y, proba):.3f}; Brier = {brier_score_loss(y, proba):.4f}")


# ============================================================
# Step 5: Bootstrap internal validation (1000x) -> optimism-corrected AUC
# ============================================================
print("\n" + "=" * 70); print("STEP 5: BOOTSTRAP INTERNAL VALIDATION (B=1000)"); print("=" * 70)
B = 1000
rng = np.random.default_rng(42)

def bootstrap_optimism(name, X, y, B=1000):
    Cls = MODELS[name]
    idx_all = np.arange(len(y))
    apparent_full = roc_auc_score(y, Cls().fit(X, y).predict_proba(X)[:, 1])
    test_aucs, optimisms, coefs_std = [], [], []
    skipped = 0
    for _ in range(B):
        bidx = rng.choice(idx_all, size=len(y), replace=True)
        Xb, yb = X[bidx], y[bidx]
        if np.unique(yb).size < 2:
            skipped += 1; continue
        try:
            m = Cls().fit(Xb, yb)
            auc_app = roc_auc_score(yb, m.predict_proba(Xb)[:, 1])
            auc_test = roc_auc_score(y, m.predict_proba(X)[:, 1])
        except Exception:
            skipped += 1; continue
        test_aucs.append(auc_test)
        optimisms.append(auc_app - auc_test)
        coefs_std.append(m.coef_std.copy())
    optimisms = np.array(optimisms); test_aucs = np.array(test_aucs)
    corrected = apparent_full - optimisms.mean()
    ci = (np.percentile(test_aucs, 2.5), np.percentile(test_aucs, 97.5))
    return {
        "apparent": apparent_full,
        "optimism": optimisms.mean(),
        "corrected": corrected,
        "ci_low": ci[0], "ci_high": ci[1],
        "test_aucs": test_aucs,
        "coefs_std": np.array(coefs_std),
        "skipped": skipped,
    }

boot = {name: bootstrap_optimism(name, X, y, B=B) for name in MODELS}
for name, r in boot.items():
    print(f"{name:12s}: apparent AUC={r['apparent']:.3f}  mean optimism={r['optimism']:+.3f}  "
          f"optimism-corrected AUC={r['corrected']:.3f}  (95% CI {r['ci_low']:.3f}-{r['ci_high']:.3f})  [skipped {r['skipped']}]")


# ============================================================
# Step 6: Performance evaluation (sens/spec, ROC, calibration)
# ============================================================
print("\n" + "=" * 70); print("STEP 6: PERFORMANCE EVALUATION"); print("=" * 70)
perf_rows = []
for name, m in fitted.items():
    proba = m.predict_proba(X)[:, 1]
    fpr, tpr, thr = roc_curve(y, proba)
    j = np.argmax(tpr - fpr)
    cut = thr[j]; sens = tpr[j]; spec = 1 - fpr[j]
    perf_rows.append([name, roc_auc_score(y, proba), boot[name]["corrected"],
                      boot[name]["ci_low"], boot[name]["ci_high"], brier_score_loss(y, proba),
                      cut, sens, spec])
    print(f"{name:12s}: apparent AUC={roc_auc_score(y,proba):.3f} | corrected={boot[name]['corrected']:.3f} "
          f"(95% CI {boot[name]['ci_low']:.3f}-{boot[name]['ci_high']:.3f}) | Brier={brier_score_loss(y,proba):.4f} "
          f"| sens={sens:.2f} spec={spec:.2f} @p>={cut:.2f}")
perf = pd.DataFrame(perf_rows, columns=["Model", "apparent_AUC", "optimism_corrected_AUC",
                                        "CI_low", "CI_high", "Brier", "threshold", "sens", "spec"])

# ROC curves
plt.figure(figsize=(5.2, 4.6))
for name, m in fitted.items():
    proba = m.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y,proba):.2f})")
plt.plot([0, 1], [0, 1], "k--", lw=0.7)
plt.xlabel("1 - Specificity"); plt.ylabel("Sensitivity"); plt.title("ROC curves (apparent)")
plt.legend(); plt.tight_layout(); plt.savefig("figures/roc_curves.png", dpi=150); plt.close()

# Calibration + Brier
plt.figure(figsize=(5.2, 4.6))
for name, m in fitted.items():
    proba = m.predict_proba(X)[:, 1]
    frac_pos, mean_pred = calibration_curve(y, proba, n_bins=5, strategy="quantile")
    plt.plot(mean_pred, frac_pos, marker="o", label=f"{name} (Brier={brier_score_loss(y,proba):.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=0.7)
plt.xlabel("Predicted probability"); plt.ylabel("Observed proportion"); plt.title("Calibration (apparent)")
plt.legend(); plt.tight_layout(); plt.savefig("figures/calibration.png", dpi=150); plt.close()


# ============================================================
# Step 7 + extras: comparison, selection, DCA, final model formula
# ============================================================
print("\n" + "=" * 70); print("STEP 7: MODEL COMPARISON + SELECTION"); print("=" * 70)
print(perf.round(3).to_string(index=False))

# DCA
def net_benefit(y_true, proba, thrs):
    N = len(y_true)
    out = []
    for t in thrs:
        pred = (proba >= t).astype(int)
        tp = np.sum((pred == 1) & (y_true == 1))
        fp = np.sum((pred == 1) & (y_true == 0))
        out.append(tp / N - fp / N * (t / (1 - t)))
    return np.array(out)

thrs = np.linspace(0.01, 0.60, 60)
plt.figure(figsize=(5.6, 4.6))
for name, m in fitted.items():
    proba = m.predict_proba(X)[:, 1]
    plt.plot(thrs, net_benefit(y, proba, thrs), label=name)
plt.plot(thrs, (thrs*0) + np.array([(y.mean()) - (1-y.mean())*(t/(1-t)) for t in thrs]), "k:", label="Treat all")
plt.axhline(0, color="gray", lw=0.6, label="Treat none")
plt.xlabel("Threshold probability"); plt.ylabel("Net benefit"); plt.title("Decision curve analysis")
plt.legend(fontsize=8); plt.ylim(bottom=-0.05); plt.tight_layout()
plt.savefig("figures/dca.png", dpi=150); plt.close()

# Coefficient stability (bootstrap CIs for ORs per-SD)
print("\nBootstrap OR (per-1-SD) with 95% percentile CI:")
for name, r in boot.items():
    cs = r["coefs_std"]
    med = np.median(cs, axis=0)
    lo = np.percentile(cs, 2.5, axis=0); hi = np.percentile(cs, 97.5, axis=0)
    parts = [f"{p}: {np.exp(med[i]):.2f} ({np.exp(lo[i]):.2f}-{np.exp(hi[i]):.2f})"
             for i, p in enumerate(PREDICTORS)]
    print(f"  {name:12s}: " + "  ".join(parts))

print("\nFigures written: figures/roc_curves.png, figures/calibration.png, figures/dca.png")
print("DONE.")
