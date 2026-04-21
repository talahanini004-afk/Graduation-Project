import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")

# =========================================================
# 1) SETTINGS
# =========================================================
FILE_PATH = "GRADUATION.xlsx"
SHEET_NAME = "Final Data"
OUTPUT_DIR = "panel_outputs"

TARGET = "Unemployment_Rate"

BASE_FEATURES = [
    "Establishments_per_1000_15plus",
    "Population_Density",
    "Students_per_Class",
    "Schools_per_1000_population"
]

# =========================================================
# 2) HELPERS
# =========================================================
def make_output_dir(path):
    os.makedirs(path, exist_ok=True)

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

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

def summarize_predictions(y_true, y_pred, label):
    return {
        "Model": label,
        "R2": r2_score(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred)
    }

# =========================================================
# 3) LOAD AND CLEAN DATA
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
    "Labor Force": "Labor_Force"
}

df = df.rename(columns=rename_map).copy()

needed_cols = ["Governorate", "Year", TARGET] + BASE_FEATURES
missing = [c for c in needed_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

for col in [TARGET] + BASE_FEATURES:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df = df.dropna(subset=["Governorate", "Year", TARGET] + BASE_FEATURES).reset_index(drop=True)
df["Year"] = df["Year"].astype(int)
df["Governorate"] = df["Governorate"].astype(str).str.strip()

df.to_excel(os.path.join(OUTPUT_DIR, "cleaned_panel_data.xlsx"), index=False)

# =========================================================
# 4) DESCRIPTIVE STATS + VIF
# =========================================================
desc = df[["Year", TARGET] + BASE_FEATURES].describe()
desc.to_excel(os.path.join(OUTPUT_DIR, "descriptive_statistics.xlsx"))

vif_df = compute_vif(df, BASE_FEATURES)
vif_df.to_excel(os.path.join(OUTPUT_DIR, "vif_results.xlsx"), index=False)

# =========================================================
# 5) FORMULAS
# =========================================================
# داخل العينة
fe_formula_in_sample = (
    f"{TARGET} ~ " +
    " + ".join(BASE_FEATURES) +
    " + C(Governorate) + C(Year)"
)

# في LOYO
fe_formula_loyo = (
    f"{TARGET} ~ " +
    " + ".join(BASE_FEATURES) +
    " + C(Governorate)"
)

# baseline
pooled_formula = (
    f"{TARGET} ~ " +
    " + ".join(BASE_FEATURES)
)

# =========================================================
# 6) POOLED OLS
# =========================================================
pooled_model = smf.ols(pooled_formula, data=df).fit(cov_type="HC3")

with open(os.path.join(OUTPUT_DIR, "pooled_ols_summary.txt"), "w", encoding="utf-8") as f:
    f.write(pooled_model.summary().as_text())

df["pooled_pred"] = pooled_model.predict(df)
pooled_metrics = summarize_predictions(df[TARGET], df["pooled_pred"], "Pooled OLS")

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

# =========================================================
# 7) FIXED EFFECTS (IN-SAMPLE)
# =========================================================
fe_model = smf.ols(fe_formula_in_sample, data=df).fit(
    cov_type="cluster",
    cov_kwds={"groups": df["Governorate"]}
)

with open(os.path.join(OUTPUT_DIR, "fixed_effects_summary.txt"), "w", encoding="utf-8") as f:
    f.write(fe_model.summary().as_text())

df["fe_pred"] = fe_model.predict(df)
fe_metrics = summarize_predictions(df[TARGET], df["fe_pred"], "Fixed Effects OLS")

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

# =========================================================
# 8) LEAVE-ONE-YEAR-OUT VALIDATION
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
    year_results.append(res)

    temp = test_df[["Governorate", "Year", TARGET]].copy()
    temp["Predicted"] = preds
    temp["Residual"] = temp[TARGET] - temp["Predicted"]
    all_preds.append(temp)

year_results_df = pd.DataFrame(year_results)
year_results_df.to_excel(os.path.join(OUTPUT_DIR, "leave_one_year_out_results.xlsx"), index=False)

predictions_df = pd.concat(all_preds, ignore_index=True)
predictions_df.to_excel(os.path.join(OUTPUT_DIR, "leave_one_year_out_predictions.xlsx"), index=False)

overall_loyo = summarize_predictions(
    predictions_df[TARGET],
    predictions_df["Predicted"],
    "Leave-One-Year-Out Overall"
)

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
fe_in_sample_r2 = fe_metrics["R2"]
fe_out_sample_r2 = overall_loyo["R2"]
fe_in_sample_rmse = fe_metrics["RMSE"]
fe_out_sample_rmse = overall_loyo["RMSE"]

if (fe_in_sample_r2 - fe_out_sample_r2 > 0.15) or (fe_out_sample_rmse > fe_in_sample_rmse * 1.25):
    overfit_flag = "Potential Overfitting"
else:
    overfit_flag = "No Strong Overfitting"

# =========================================================
# 11) CLUSTERING OF GOVERNORATES
# =========================================================
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram

# -------------------------------
# اختاري متغيرات معبرة وغير مكررة كثيرًا
# -------------------------------
CLUSTER_FEATURES = [
    "Unemployment_Rate",
    "Establishments_per_1000_15plus",
    "Population_Density",
    "Students_per_Class",
    "Schools_per_1000_population",
    "Student_Ratio_5_19"
]

missing_cluster_cols = [c for c in ["Governorate"] + CLUSTER_FEATURES if c not in df.columns]
if missing_cluster_cols:
    print(f"Skipping clustering بسبب أعمدة ناقصة: {missing_cluster_cols}")
else:
    # ---------------------------------------
    # 1) نجمع حسب المحافظة باستخدام المتوسط
    # ---------------------------------------
    cluster_df = df.groupby("Governorate")[CLUSTER_FEATURES].mean().reset_index()

    cluster_df.to_excel(
        os.path.join(OUTPUT_DIR, "cluster_governorate_averages.xlsx"),
        index=False
    )

    # ---------------------------------------
    # 2) Standardization
    # ---------------------------------------
    scaler = StandardScaler()
    X_cluster = scaler.fit_transform(cluster_df[CLUSTER_FEATURES])

    # ---------------------------------------
    # 3) Elbow Method
    # ---------------------------------------
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

    # ---------------------------------------
    # 4) Silhouette Scores
    # ---------------------------------------
    silhouette_results = []
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_cluster)
        sil = silhouette_score(X_cluster, labels)
        silhouette_results.append({"k": k, "Silhouette Score": sil})

    silhouette_df = pd.DataFrame(silhouette_results)
    silhouette_df.to_excel(
        os.path.join(OUTPUT_DIR, "clustering_silhouette_scores.xlsx"),
        index=False
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(silhouette_df["k"], silhouette_df["Silhouette Score"], marker="o")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Scores by k")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "clustering_silhouette_scores.png"), dpi=300)
    plt.close()

    # ---------------------------------------
    # 5) اختاري أفضل k
    # ---------------------------------------
    best_k = silhouette_df.sort_values("Silhouette Score", ascending=False).iloc[0]["k"]
    best_k = int(best_k)

    # ---------------------------------------
    # 6) Final KMeans
    # ---------------------------------------
    final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_df["Cluster"] = final_kmeans.fit_predict(X_cluster)

    # ---------------------------------------
    # 7) Cluster Profile
    # ---------------------------------------
    cluster_profile = cluster_df.groupby("Cluster")[CLUSTER_FEATURES].mean()
    cluster_profile.to_excel(
        os.path.join(OUTPUT_DIR, "cluster_profiles.xlsx")
    )

    cluster_df.to_excel(
        os.path.join(OUTPUT_DIR, "governorates_with_clusters.xlsx"),
        index=False
    )

    # ---------------------------------------
    # 8) PCA for 2D visualization
    # ---------------------------------------
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_cluster)

    cluster_df["PCA1"] = X_pca[:, 0]
    cluster_df["PCA2"] = X_pca[:, 1]

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(cluster_df["PCA1"], cluster_df["PCA2"], c=cluster_df["Cluster"])
    for _, row in cluster_df.iterrows():
        ax.text(row["PCA1"], row["PCA2"], row["Governorate"], fontsize=9)
    ax.set_title(f"K-Means Clustering of Governorates (k={best_k})")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "governorates_clusters_pca.png"), dpi=300)
    plt.close()

    # ---------------------------------------
    # 9) Hierarchical Clustering Dendrogram
    # ---------------------------------------
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

    # ---------------------------------------
    # 10) Text summary
    # ---------------------------------------
    with open(os.path.join(OUTPUT_DIR, "clustering_summary.txt"), "w", encoding="utf-8") as f:
        f.write("Governorate Clustering Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Features used: {CLUSTER_FEATURES}\n")
        f.write(f"Best number of clusters (by silhouette score): {best_k}\n\n")
        f.write("Silhouette scores:\n")
        f.write(silhouette_df.to_string(index=False))
        f.write("\n\n")
        f.write("Cluster profiles:\n")
        f.write(cluster_profile.to_string())
        f.write("\n\n")
        f.write("Governorates by cluster:\n")
        f.write(cluster_df[["Governorate", "Cluster"]].sort_values("Cluster").to_string(index=False))

    print("\n=== CLUSTERING DONE ===")
    print(f"Best k: {best_k}")
    print("\nGovernorates with clusters:")
    print(cluster_df[["Governorate", "Cluster"]].sort_values("Cluster"))

# =========================================================
# 10) SAVE FINAL COMPARISON
# =========================================================
comparison_df = pd.DataFrame([
    pooled_metrics,
    fe_metrics,
    overall_loyo
])

comparison_df.to_excel(os.path.join(OUTPUT_DIR, "model_comparison.xlsx"), index=False)

summary_lines = []
summary_lines.append("Panel / Fixed Effects Modeling Summary")
summary_lines.append("=" * 50)
summary_lines.append("")
summary_lines.append("Base Features Used:")
for feat in BASE_FEATURES:
    summary_lines.append(f"- {feat}")
summary_lines.append("")
summary_lines.append("Pooled OLS Performance:")
summary_lines.append(str(pooled_metrics))
summary_lines.append("")
summary_lines.append("Fixed Effects In-Sample Performance:")
summary_lines.append(str(fe_metrics))
summary_lines.append("")
summary_lines.append("Leave-One-Year-Out Performance:")
summary_lines.append(str(overall_loyo))
summary_lines.append("")
summary_lines.append(f"Overfitting Check: {overfit_flag}")
summary_lines.append("")
summary_lines.append("Interpretation:")
summary_lines.append("- In-sample FE uses Governorate and Year fixed effects.")
summary_lines.append("- LOYO uses Governorate fixed effects only to avoid unseen year-category errors.")
summary_lines.append("- If LOYO R2 remains strongly negative, predictors are still weak for explaining unemployment.")

with open(os.path.join(OUTPUT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))

print("\n=== MODEL COMPARISON ===")
print(comparison_df)
print("\n=== LEAVE-ONE-YEAR-OUT RESULTS ===")
print(year_results_df)
print(f"\nOverfitting Check: {overfit_flag}")
print(f"\nAll outputs saved in: {OUTPUT_DIR}")