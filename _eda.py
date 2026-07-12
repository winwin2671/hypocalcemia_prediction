"""Exploratory profile of the raw dataset to ground methodology decisions."""
import pandas as pd
import numpy as np

RAW = "Thyroid_Sx_research_(ReadyToClean).csv"

df = pd.read_csv(RAW)
# Strip whitespace from column names
df.columns = [c.strip() for c in df.columns]

# Drop trailing summary rows (Mean / SD) — keep only rows whose 'case No.' is an int
df = df[df["case No."].astype(str).str.strip().str.fullmatch(r"\d+")].copy()
df["case No."] = df["case No."].astype(int)
df = df.reset_index(drop=True)

print("=" * 70)
print(f"N (patient rows after dropping Mean/SD) = {len(df)}")
print("=" * 70)

print("\nCOLUMNS:", df.columns.tolist())

# ---- Patient ID sanity (anonymization artifacts?) ----
pid = df["Patient ID"].astype(str)
weird = pid[~pid.str.fullmatch(r"\d+")]
print(f"\nPatient IDs that are NOT pure integers: {len(weird)}  -> examples: {weird.head(8).tolist()}")
print(f"Duplicate Patient IDs: {df['Patient ID'].duplicated().sum()}")

# ---- Missingness ----
print("\n--- MISSINGNESS (raw, as stored) ---")
miss = df.isna().sum()
print(miss[miss > 0].to_string() if miss.sum() else "No NaN cells (but check for '0'/blank sentinels)")

# ---- Helper to coerce numeric ----
def num(series):
    return pd.to_numeric(series.astype(str).str.strip(), errors="coerce")

# ---- Outcome candidates ----
postca = num(df["Postop Ca"])
postpth = num(df["Postop PTH"])
print("\n--- POSTOP CALCIUM (mmol/L) ---")
print(postca.describe().round(3).to_string())
print("\nEvent counts at candidate hypocalcemia thresholds:")
for t in [1.90, 2.00, 2.05, 2.10, 2.12, 2.15, 2.20]:
    n_ev = int((postca < t).sum())
    print(f"  Postop Ca < {t:.2f}:  n_event = {n_ev}  ({n_ev/len(df)*100:.1f}%)   EPV(3 preds) = {n_ev/3:.1f}   EPV(5 preds) = {n_ev/5:.1f}")

print("\n--- POSTOP PTH (pg/mL) ---")
print(postpth.describe().round(2).to_string())
for t in [10, 15, 20]:
    n_ev = int((postpth < t).sum())
    print(f"  Postop PTH < {t}:  n = {n_ev} ({n_ev/len(df)*100:.1f}%)")

# ---- Key preoperative predictors ----
preds = {
    "Pre-op PTH (pg/mL)": num(df["Pre-op PTH"]),
    "Pre-op Ca (mmol/L)": num(df["Pre-op Ca"]),
    "Vit D(preop) (ng/mL)": num(df["Vit D(preop)"]),
    "Age": num(df["Age"]),
    "BMI": num(df["BMI"]),
    "Preop Mg": num(df["Preop Mg"]),
    "Pre P": num(df["Pre P"]),
    "Surgical duration (min)": num(df["Surgical duration"]),
    "Intraop bleeding": num(df["Intraoperative bleeding"]),
}
print("\n--- CONTINUOUS PREDICTOR DISTRIBUTIONS ---")
for name, s in preds.items():
    qs = s.quantile([.25, .5, .75]).round(2).to_dict()
    print(f"  {name:28s} n={s.notna().sum():3d}  mean={s.mean():.2f}  sd={s.std():.2f}  min={s.min():.2f}  max={s.max():.2f}  IQR=[{qs[.25]}, {qs[.75]}]")

# ---- Categorical predictors ----
print("\n--- CATEGORICAL BREAKDOWNS ---")
for col in ["Sex", "Diagnosis", "Extent of Sx", "LND", "CLND", "Antibody", "T stage", "PTA"]:
    if col in df.columns:
        vc = df[col].astype(str).str.strip().value_counts(dropna=False)
        print(f"\n  {col}  ({vc.size} distinct values):")
        print("    " + vc.to_string().replace("\n", "\n    "))

# ---- Cross-tab: outcome (Ca<2.0) vs Extent / CLND to eyeball signal ----
df["_hypo_ca20"] = postca < 2.0
df["_hypo_ca21"] = postca < 2.10
print("\n--- EVENT RATE BY EXTENT OF SURGERY ---")
print(pd.crosstab(df["Extent of Sx"].astype(str).str.strip(), df["_hypo_ca21"], normalize="index").round(3).to_string())
print("\n--- EVENT RATE BY CLND ---")
print(pd.crosstab(df["CLND"].astype(str).str.strip(), df["_hypo_ca21"], normalize="index").round(3).to_string())
