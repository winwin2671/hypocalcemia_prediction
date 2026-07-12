"""Quick validation of SOTA metric helpers + dcurves API before notebook build."""
import numpy as np
import pandas as pd
from _sota_metrics import delong_auc_ci, delong_paired_test, calibration_metrics

rng = np.random.default_rng(0)
n = 78
y = rng.integers(0, 2, size=n)
s1 = rng.random(n)
s2 = s1 + rng.normal(0, 0.3, n)

print("DeLong AUC CI (s1):", delong_auc_ci(y, s1))
print("DeLong AUC CI (s2):", delong_auc_ci(y, s2))
print("DeLong paired test p:", delong_paired_test(y, s1, s2))
print("Calibration (intercept, slope, ICI) for s1:", calibration_metrics(y, s1))

# ---- dcurves API check ----
import dcurves
print("\ndcurves version:", getattr(dcurves, "__version__", "unknown"))
print("dcurves exports:", [x for x in dir(dcurves) if not x.startswith("_")])
try:
    from dcurves import dca
    df = pd.DataFrame({"y": y, "m1": s1, "m2": s2})
    out = dca(data=df, outcome="y", model_names=["m1", "m2"], thresholds=np.arange(0.05, 0.5, 0.05))
    print("dca() worked. type:", type(out))
    print(out.head() if hasattr(out, "head") else str(out)[:300])
except Exception as e:
    print("dca() signature/error:", repr(e))
    import inspect
    try:
        print("dca signature:", inspect.signature(dca))
    except Exception as e2:
        print("no sig:", e2)
