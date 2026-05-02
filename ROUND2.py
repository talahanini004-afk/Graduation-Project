import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import jarque_bera

warnings.filterwarnings("ignore")

# =========================================================
# 1) SETTINGS
# =========================================================
FILE_PATH = "GRADUATION.xlsx"
SHEET_NAME = "Final Data"
OUTPUT_DIR = "FINAL_PROJECT_OUTPUTS_IMPROVED"

TARGET = "Unemployment_Rate"

# Final selected regression specification
BASE_FEATURES = [
    "Establishments_per_1000_15plus",
    "Population_Density",
    "Schools_per_1000_population"
]

# Optional second specification with quadratic time trend
USE_YEAR_SQUARED = True

# Clustering features
CLUSTER_FEATURES = [
    "Unemployment_Rate",
    "Establishments_per_1000_15plus",
    "Population_Density",
    "Students_per_Class",
    "Schools_per_1000_population",
    "Student_Ratio_5_19"
]

# =========================================================
# 2) HELPERS
# =========================================================
def make_output_dir(path):
    os.makedirs(path, exist_ok=True)

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def summarize_predictions(y_true, y_pred, label):
    return {
        "Model": label,
        "R2": r2_score(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred)
    }

def compute_vif(df, features):
    X = df[features].copy()
    X = sm.add_constant(X)
    out = pd.DataFrame()
    out["Variable"] = X.columns
    out["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return out

def plot_actual_vs_pred(y_true, y_pred, title, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_true, y_pred)
    mn = min(np.min(y_true), np.min(y_pred))
    mx = max(np.max(y_true), np.max(y_pred))
    ax.plot([mn, mx], [mn, mx])
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def plot_residuals(y_true, y_pred, title, path):
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_pred, residuals)
    ax.axhline(0)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residuals")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def plot_coefficients(coef_series, title, path):
    coef_series = coef_series.sort_values(key=np.abs, ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(coef_series.index, coef_series.values)
    ax.set_title(title)
    ax.set_ylabel("Coefficient")
    ax.set_xticks(range(len(coef_series.index)))
    ax.set_xticklabels(coef_series.index, rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def overfitting_flag(in_r2, out_r2, in_rmse, out_rmse):
    if (in_r2 - out_r2 > 0.15) or (out_rmse > in_rmse * 1.25):
        return "Potential Overfitting"
    return "No Strong Overfitting"

def residual_diagnostics(model, model_name):
    resid = model.resid
    exog = model.model.exog

    jb_stat, jb_pvalue, skew, kurtosis = jarque_bera(resid)
    bp_test = het_breuschpagan(resid, exog)
    bp_lm_stat, bp_lm_pvalue, bp_f_stat, bp_f_pvalue = bp_test

    return pd.DataFrame([{
        "Model": model_name,
        "Jarque_Bera_Stat": jb_stat,
        "Jarque_Bera_pvalue": jb_pvalue,
        "Skewness": skew,
        "Kurtosis": kurtosis,
        "BreuschPagan_LM_Stat": bp_lm_stat,
        "BreuschPagan_LM_pvalue": bp_lm_pvalue,
        "BreuschPagan_F_Stat": bp_f_stat,
        "BreuschPagan_F_pvalue": bp_f_pvalue
    }])

def extract_coeff_table(model, variables, model_name):
    rows = []
    for var in variables:
        if var in model.params.index:
            rows.append({
                "Model": model_name,
                "Variable": var,
                "Coefficient": model.params[var],
                "Std_Error": model.bse[var],
                "t_value": model.tvalues[var],
                "p_value": model.pvalues[var],
                "CI_95_Lower": model.conf_int().loc[var, 0],
                "CI_95_Upper": model.conf_int().loc[var, 1]
            })
    return pd.DataFrame(rows)

# =========================================================
# 3) LOAD & CLEAN DATA
# =========================================================
make_output_dir(OUTPUT_DIR)

df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)

rename_map = {
    "Governorates": "Governorate",
    "YEAR": "Year",
    "Establishments per 1,000 (15+)": "Establishments_per_1000_15plus",
    "Employment Rate": "Employment_Rate",
    "Unemployment Rate": "Unemployment_Rate",
    "Num of Schools ": "Num_of_Schools",
    "Num of Schools": "Num_of_Schools",
    "Num of Students": "Num_of_Students",
    "Number of Classes": "Number_of_Classes",
    "Students per School": "Students_per_School",
    "Classes per School": "Classes_per_School",
    "Students per Class": "Students_per_Class",
    "Population 5-19": "Population_5_19",
    "Student Ratio (5–19": "Student_Ratio_5_19",
    "Student Ratio (5–19)": "Student_Ratio_5_19",
    "Schools per 1,000 population": "Schools_per_1000_population",
    "Area ": "Area",
    "Area": "Area",
    "Total Population": "Total_Population",
    "Population Density": "Population_Density",
    "Population +15": "Population_15plus",
    "Labor Force": "Labor_Force",
    "Unemployed": "Unemployed",
    "Employed": "Employed"
}

df = df.rename(columns=rename_map).copy()
df["Governorate"] = df["Governorate"].astype(str).str.strip()
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

all_needed = {"Governorate", "Year", TARGET}
all_needed.update(BASE_FEATURES)
all_needed.update(CLUSTER_FEATURES)

missing = [c for c in all_needed if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

for col in list(all_needed - {"Governorate"}):
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["Governorate", "Year", TARGET] + BASE_FEATURES).reset_index(drop=True)
df["Year"] = df["Year"].astype(int)

if USE_YEAR_SQUARED:
    df["Year_Squared"] = df["Year"] ** 2

df.to_excel(os.path.join(OUTPUT_DIR, "cleaned_panel_data.xlsx"), index=False)

# =========================================================
# 4) DESCRIPTIVE STATS + VIF
# =========================================================
desc_cols = ["Year", TARGET] + BASE_FEATURES
if USE_YEAR_SQUARED:
    desc_cols.append("Year_Squared")

desc = df[desc_cols].describe()
desc.to_excel(os.path.join(OUTPUT_DIR, "descriptive_statistics.xlsx"))

vif_df = compute_vif(df, BASE_FEATURES)
vif_df.to_excel(os.path.join(OUTPUT_DIR, "vif_results.xlsx"), index=False)

# =========================================================
# 5) FORMULAS
# =========================================================
pooled_formula = f"{TARGET} ~ " + " + ".join(BASE_FEATURES)

fe_formula_in_sample = (
    f"{TARGET} ~ " +
    " + ".join(BASE_FEATURES) +
    " + C(Governorate) + C(Year)"
)

if USE_YEAR_SQUARED:
    fe_formula_loyo = (
        f"{TARGET} ~ " +
        " + ".join(BASE_FEATURES) +
        " + Year + Year_Squared + C(Governorate)"
    )
    loyo_vars_for_export = BASE_FEATURES + ["Year", "Year_Squared"]
else:
    fe_formula_loyo = (
        f"{TARGET} ~ " +
        " + ".join(BASE_FEATURES) +
        " + Year + C(Governorate)"
    )
    loyo_vars_for_export = BASE_FEATURES + ["Year"]

# =========================================================
# 6) POOLED OLS
# =========================================================
pooled_model = smf.ols(pooled_formula, data=df).fit(cov_type="HC3")

with open(os.path.join(OUTPUT_DIR, "pooled_ols_summary.txt"), "w", encoding="utf-8") as f:
    f.write(pooled_model.summary().as_text())

df["pooled_pred"] = pooled_model.predict(df)
pooled_metrics = summarize_predictions(df[TARGET], df["pooled_pred"], "Pooled OLS")
pooled_metrics["Adjusted_R2"] = pooled_model.rsquared_adj
pooled_metrics["F_Statistic"] = pooled_model.fvalue
pooled_metrics["F_pvalue"] = pooled_model.f_pvalue

plot_actual_vs_pred(
    df[TARGET],
    df["pooled_pred"],
    "Actual vs Predicted - Pooled OLS",
    os.path.join(OUTPUT_DIR, "pooled_actual_vs_predicted.png")
)

plot_residuals(
    df[TARGET],
    df["pooled_pred"],
    "Residuals - Pooled OLS",
    os.path.join(OUTPUT_DIR, "pooled_residuals.png")
)

pooled_diag = residual_diagnostics(pooled_model, "Pooled OLS")
pooled_diag.to_excel(os.path.join(OUTPUT_DIR, "pooled_residual_diagnostics.xlsx"), index=False)

pooled_coef = extract_coeff_table(pooled_model, BASE_FEATURES, "Pooled OLS")
pooled_coef.to_excel(os.path.join(OUTPUT_DIR, "pooled_coefficients.xlsx"), index=False)

# =========================================================
# 7) FIXED EFFECTS OLS
# =========================================================
fe_model = smf.ols(fe_formula_in_sample, data=df).fit(
    cov_type="cluster",
    cov_kwds={"groups": df["Governorate"]}
)

with open(os.path.join(OUTPUT_DIR, "fixed_effects_summary.txt"), "w", encoding="utf-8") as f:
    f.write(fe_model.summary().as_text())

df["fe_pred"] = fe_model.predict(df)
fe_metrics = summarize_predictions(df[TARGET], df["fe_pred"], "Fixed Effects OLS")
fe_metrics["Adjusted_R2"] = fe_model.rsquared_adj
fe_metrics["F_Statistic"] = fe_model.fvalue
fe_metrics["F_pvalue"] = fe_model.f_pvalue

plot_actual_vs_pred(
    df[TARGET],
    df["fe_pred"],
    "Actual vs Predicted - Fixed Effects",
    os.path.join(OUTPUT_DIR, "fe_actual_vs_predicted.png")
)

plot_residuals(
    df[TARGET],
    df["fe_pred"],
    "Residuals - Fixed Effects",
    os.path.join(OUTPUT_DIR, "fe_residuals.png")
)

base_coef = fe_model.params[[c for c in fe_model.params.index if c in BASE_FEATURES]]
base_coef.to_excel(os.path.join(OUTPUT_DIR, "fixed_effects_base_coefficients.xlsx"))

plot_coefficients(
    base_coef,
    "Fixed Effects - Base Variable Coefficients",
    os.path.join(OUTPUT_DIR, "fixed_effects_base_coefficients.png")
)

fe_diag = residual_diagnostics(fe_model, "Fixed Effects OLS")
fe_diag.to_excel(os.path.join(OUTPUT_DIR, "fixed_effects_residual_diagnostics.xlsx"), index=False)

fe_coef = extract_coeff_table(fe_model, BASE_FEATURES, "Fixed Effects OLS")
fe_coef.to_excel(os.path.join(OUTPUT_DIR, "fixed_effects_coefficients_detailed.xlsx"), index=False)

# =========================================================
# 8) LEAVE-ONE-YEAR-OUT
# =========================================================
year_results = []
all_preds = []

years = sorted(df["Year"].unique())

for test_year in years:
    train_df = df[df["Year"] != test_year].copy()
    test_df = df[df["Year"] == test_year].copy()

    model = smf.ols(fe_formula_loyo, data=train_df).fit(
        cov_type="cluster",
        cov_kwds={"groups": train_df["Governorate"]}
    )

    preds = model.predict(test_df)

    res = summarize_predictions(
        test_df[TARGET],
        preds,
        f"TestYear_{test_year}"
    )
    res["Test_Year"] = test_year
    res["Adjusted_R2_Train"] = model.rsquared_adj
    year_results.append(res)

    tmp = test_df[["Governorate", "Year", TARGET]].copy()
    tmp["Predicted"] = preds
    tmp["Residual"] = tmp[TARGET] - tmp["Predicted"]
    all_preds.append(tmp)

year_results_df = pd.DataFrame(year_results)
predictions_df = pd.concat(all_preds, ignore_index=True)

year_results_df.to_excel(os.path.join(OUTPUT_DIR, "leave_one_year_out_results.xlsx"), index=False)
predictions_df.to_excel(os.path.join(OUTPUT_DIR, "leave_one_year_out_predictions.xlsx"), index=False)

overall_loyo = summarize_predictions(
    predictions_df[TARGET],
    predictions_df["Predicted"],
    "Leave-One-Year-Out Overall"
)
overall_loyo["Adjusted_R2"] = np.nan
overall_loyo["F_Statistic"] = np.nan
overall_loyo["F_pvalue"] = np.nan

plot_actual_vs_pred(
    predictions_df[TARGET],
    predictions_df["Predicted"],
    "Actual vs Predicted - Leave-One-Year-Out",
    os.path.join(OUTPUT_DIR, "loyo_actual_vs_predicted.png")
)

plot_residuals(
    predictions_df[TARGET],
    predictions_df["Predicted"],
    "Residuals - Leave-One-Year-Out",
    os.path.join(OUTPUT_DIR, "loyo_residuals.png")
)

# =========================================================
# 9) OVERFITTING CHECK
# =========================================================
overfit_flag = overfitting_flag(
    fe_metrics["R2"],
    overall_loyo["R2"],
    fe_metrics["RMSE"],
    overall_loyo["RMSE"]
)

# =========================================================
# 10) FINAL MODEL COMPARISON
# =========================================================
comparison_df = pd.DataFrame([
    pooled_metrics,
    fe_metrics,
    overall_loyo
])
comparison_df["Overfitting_Check"] = [np.nan, np.nan, overfit_flag]
comparison_df.to_excel(os.path.join(OUTPUT_DIR, "model_comparison.xlsx"), index=False)

# =========================================================
# 11) CLUSTERING
# =========================================================
cluster_df = df.groupby("Governorate")[CLUSTER_FEATURES].mean().reset_index()
cluster_df.to_excel(os.path.join(OUTPUT_DIR, "cluster_governorate_averages.xlsx"), index=False)

scaler = StandardScaler()
X_cluster = scaler.fit_transform(cluster_df[CLUSTER_FEATURES])

# Elbow
inertias = []
k_values = range(2, 7)

for k in k_values:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_cluster)
    inertias.append(km.inertia_)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(list(k_values), inertias, marker="o")
ax.set_xlabel("Number of Clusters (k)")
ax.set_ylabel("Inertia")
ax.set_title("Elbow Method")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "clustering_elbow_method.png"), dpi=300)
plt.close()

# Silhouette
silhouette_results = []
for k in k_values:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_cluster)
    sil = silhouette_score(X_cluster, labels)
    silhouette_results.append({"k": k, "Silhouette Score": sil})

silhouette_df = pd.DataFrame(silhouette_results)
silhouette_df.to_excel(os.path.join(OUTPUT_DIR, "clustering_silhouette_scores.xlsx"), index=False)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(silhouette_df["k"], silhouette_df["Silhouette Score"], marker="o")
ax.set_xlabel("Number of Clusters (k)")
ax.set_ylabel("Silhouette Score")
ax.set_title("Silhouette Scores by k")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "clustering_silhouette_scores.png"), dpi=300)
plt.close()

best_k = int(silhouette_df.sort_values("Silhouette Score", ascending=False).iloc[0]["k"])

# Final KMeans
final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_df["Cluster"] = final_kmeans.fit_predict(X_cluster)

cluster_profile = cluster_df.groupby("Cluster")[CLUSTER_FEATURES].mean()
cluster_profile.to_excel(os.path.join(OUTPUT_DIR, "cluster_profiles.xlsx"))
cluster_df.to_excel(os.path.join(OUTPUT_DIR, "governorates_with_clusters.xlsx"), index=False)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_cluster)

cluster_df["PCA1"] = X_pca[:, 0]
cluster_df["PCA2"] = X_pca[:, 1]

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(cluster_df["PCA1"], cluster_df["PCA2"], c=cluster_df["Cluster"])
for _, row in cluster_df.iterrows():
    ax.text(row["PCA1"], row["PCA2"], row["Governorate"], fontsize=9)
ax.set_title(f"K-Means Clustering of Governorates (k={best_k})")
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "governorates_clusters_pca.png"), dpi=300)
plt.close()

# Hierarchical
linked = linkage(X_cluster, method="ward")

fig, ax = plt.subplots(figsize=(10, 6))
dendrogram(
    linked,
    labels=cluster_df["Governorate"].tolist(),
    leaf_rotation=45,
    leaf_font_size=10,
    ax=ax
)
ax.set_title("Hierarchical Clustering Dendrogram")
ax.set_ylabel("Distance")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "hierarchical_dendrogram.png"), dpi=300)
plt.close()

# =========================================================
# 12) FINAL SUMMARY FILE
# =========================================================
with open(os.path.join(OUTPUT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
    f.write("FINAL PROJECT SUMMARY\n")
    f.write("=" * 60 + "\n\n")

    f.write("Selected Regression Features:\n")
    for feat in BASE_FEATURES:
        f.write(f"- {feat}\n")

    f.write("\nModel Comparison:\n")
    f.write(comparison_df.to_string(index=False))

    f.write("\n\nLOYO by Year:\n")
    f.write(year_results_df.to_string(index=False))

    f.write(f"\n\nOverfitting Check: {overfit_flag}\n")

    f.write("\n\nInterpretation Notes:\n")
    f.write("- Pooled OLS is the baseline model.\n")
    f.write("- Fixed Effects OLS is the main explanatory model.\n")
    f.write("- LOYO is used to evaluate temporal generalization.\n")
    f.write("- VIF checks multicollinearity.\n")
    f.write("- Adjusted R2 penalizes unnecessary model complexity.\n")
    f.write("- Jarque-Bera tests residual normality.\n")
    f.write("- Breusch-Pagan tests heteroskedasticity.\n")

    f.write("\n\nClustering Summary:\n")
    f.write(f"Best number of clusters: {best_k}\n")
    f.write("\nSilhouette Scores:\n")
    f.write(silhouette_df.to_string(index=False))

    f.write("\n\nCluster Profiles:\n")
    f.write(cluster_profile.to_string())

    f.write("\n\nGovernorates by Cluster:\n")
    f.write(cluster_df[["Governorate", "Cluster"]].sort_values("Cluster").to_string(index=False))

print("\n=== MODEL COMPARISON ===")
print(comparison_df)

print("\n=== LOYO RESULTS ===")
print(year_results_df)

print(f"\nOverfitting Check: {overfit_flag}")

print("\n=== CLUSTERING SUMMARY ===")
print(f"Best k: {best_k}")
print(cluster_df[["Governorate", "Cluster"]].sort_values("Cluster"))

print(f"\nAll outputs saved in: {OUTPUT_DIR}")