
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "figure.figsize": (10, 6)})

print("=" * 65)
print("  SCT_ML_T1 - House Price Prediction using Linear Regression")
print("=" * 65)



DATA_PATH = "train.csv"   

try:
    df_raw = pd.read_csv(DATA_PATH)
    print("[OK] Dataset loaded -> {} rows x {} columns".format(
        df_raw.shape[0], df_raw.shape[1]))
except FileNotFoundError:
    print(
        "\n[!] 'train.csv' not found in the current directory.\n"
        "    A synthetic dataset will be generated for demonstration.\n"
        "    Replace DATA_PATH with the correct path to the real dataset."
    )

    np.random.seed(42)
    n = 1460
    sqft       = np.random.randint(500, 4500, size=n)
    bedrooms   = np.random.randint(1, 6, size=n)
    full_baths = np.random.randint(1, 4, size=n)
    half_baths = np.random.randint(0, 2, size=n)
    noise      = np.random.normal(0, 15000, size=n)
    price = (
        80 * sqft
        + 10000 * bedrooms
        + 15000 * full_baths
        + 5000  * half_baths
        + 50000
        + noise
    ).clip(min=50000)
    df_raw = pd.DataFrame({
        "GrLivArea"    : sqft,
        "BedroomAbvGr" : bedrooms,
        "FullBath"     : full_baths,
        "HalfBath"     : half_baths,
        "SalePrice"    : price.astype(int),
    })
    print("[OK] Synthetic dataset created -> {} rows x {} columns".format(
        df_raw.shape[0], df_raw.shape[1]))

FEATURES     = ["GrLivArea", "BedroomAbvGr", "FullBath", "HalfBath"]
TARGET       = "SalePrice"


df = df_raw[FEATURES + [TARGET]].dropna().copy()


df["TotalBaths"] = df["FullBath"] + 0.5 * df["HalfBath"]
FEATURES_ENG = ["GrLivArea", "BedroomAbvGr", "TotalBaths"]

print("[Features used]   : {}".format(FEATURES_ENG))
print("[Target variable] : {}".format(TARGET))
print("[Clean rows]      : {}".format(len(df)))


print("\n--- Descriptive Statistics ---")
print(df[FEATURES_ENG + [TARGET]].describe().round(2).to_string())


fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df[TARGET] / 1000, bins=40, color="#4C72B0", edgecolor="white", linewidth=0.6)
axes[0].set_title("Distribution of Sale Price", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Sale Price ($ thousands)")
axes[0].set_ylabel("Frequency")

axes[1].hist(np.log1p(df[TARGET]), bins=40, color="#DD8452", edgecolor="white", linewidth=0.6)
axes[1].set_title("Distribution of log(Sale Price)", fontsize=14, fontweight="bold")
axes[1].set_xlabel("log(Sale Price)")
axes[1].set_ylabel("Frequency")

plt.suptitle("Target Variable Distribution", fontsize=16, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("01_target_distribution.png", bbox_inches="tight")
plt.show()
print("[OK] Saved -> 01_target_distribution.png")


fig, ax = plt.subplots(figsize=(7, 5))
corr = df[FEATURES_ENG + [TARGET]].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
    linewidths=0.5, square=True, ax=ax
)
ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("02_correlation_heatmap.png", bbox_inches="tight")
plt.show()
print("[OK] Saved -> 02_correlation_heatmap.png")


fig, axes = plt.subplots(1, 3, figsize=(16, 5))
feature_labels = {
    "GrLivArea"    : "Above-Grade Living Area (sq ft)",
    "BedroomAbvGr" : "Number of Bedrooms",
    "TotalBaths"   : "Total Bathrooms",
}
for ax, feat in zip(axes, FEATURES_ENG):
    ax.scatter(df[feat], df[TARGET] / 1000, alpha=0.35, s=18, color="#4C72B0")
    ax.set_xlabel(feature_labels[feat], fontsize=11)
    ax.set_ylabel("Sale Price ($ thousands)", fontsize=11)
    ax.set_title("{}\nvs Sale Price".format(feature_labels[feat]),
                 fontsize=12, fontweight="bold")

plt.suptitle("Feature vs Sale Price Scatter Plots", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("03_feature_scatter_plots.png", bbox_inches="tight")
plt.show()
print("[OK] Saved -> 03_feature_scatter_plots.png")


X = df[FEATURES_ENG].values
y = df[TARGET].values


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)


scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print("\n[Train set] : {} samples".format(X_train.shape[0]))
print("[Test  set] : {} samples".format(X_test.shape[0]))


model = LinearRegression()
model.fit(X_train, y_train)

print("\n--- Model Coefficients ---")
print("   Intercept : ${:,.2f}".format(model.intercept_))
for feat, coef in zip(FEATURES_ENG, model.coef_):
    print("   {:<20s}: ${:,.2f}".format(feat, coef))

y_pred_train = model.predict(X_train)
y_pred_test  = model.predict(X_test)

def evaluate(y_true, y_pred, split_name):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print("\n  [{}]".format(split_name))
    print("    MAE  (Mean Absolute Error) : ${:>12,.2f}".format(mae))
    print("    RMSE (Root Mean Sq Error)  : ${:>12,.2f}".format(rmse))
    print("    R2   (Coefficient of Det.) :  {:>11.4f}".format(r2))
    return mae, rmse, r2

print("\n--- Evaluation Metrics ---")
train_metrics = evaluate(y_train, y_pred_train, "Training Set")
test_metrics  = evaluate(y_test,  y_pred_test,  "Test Set    ")


fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test / 1000, y_pred_test / 1000, alpha=0.5, s=30,
           color="#4C72B0", label="Predicted vs Actual")
lims = [
    min(y_test.min(), y_pred_test.min()) / 1000 - 10,
    max(y_test.max(), y_pred_test.max()) / 1000 + 10,
]
ax.plot(lims, lims, "r--", linewidth=1.8, label="Perfect Prediction")
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("Actual Price ($ thousands)", fontsize=12)
ax.set_ylabel("Predicted Price ($ thousands)", fontsize=12)
ax.set_title("Actual vs Predicted Sale Price", fontsize=14, fontweight="bold")
ax.legend()
ax.text(
    0.05, 0.92, "R2 = {:.4f}".format(test_metrics[2]),
    transform=ax.transAxes, fontsize=12,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8)
)
plt.tight_layout()
plt.savefig("04_actual_vs_predicted.png", bbox_inches="tight")
plt.show()
print("[OK] Saved -> 04_actual_vs_predicted.png")


residuals = y_test - y_pred_test

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(y_pred_test / 1000, residuals / 1000, alpha=0.5, s=25, color="#DD8452")
axes[0].axhline(0, color="red", linewidth=1.5, linestyle="--")
axes[0].set_xlabel("Predicted Price ($ thousands)", fontsize=12)
axes[0].set_ylabel("Residuals ($ thousands)", fontsize=12)
axes[0].set_title("Residuals vs Predicted Price", fontsize=13, fontweight="bold")

axes[1].hist(residuals / 1000, bins=35, color="#4C72B0", edgecolor="white", linewidth=0.6)
axes[1].set_xlabel("Residuals ($ thousands)", fontsize=12)
axes[1].set_ylabel("Frequency", fontsize=12)
axes[1].set_title("Residual Distribution", fontsize=13, fontweight="bold")

plt.suptitle("Residual Analysis", fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("05_residual_analysis.png", bbox_inches="tight")
plt.show()
print("[OK] Saved -> 05_residual_analysis.png")


fig, ax = plt.subplots(figsize=(7, 4))
coef_abs = np.abs(model.coef_)
colors   = ["#4C72B0", "#DD8452", "#55A868"]
bars = ax.barh(FEATURES_ENG, coef_abs, color=colors, edgecolor="white")
ax.bar_label(bars, fmt="%.0f", padding=4, fontsize=10)
ax.set_xlabel("Absolute Coefficient (scaled units)", fontsize=12)
ax.set_title("Feature Importance (Absolute Coefficients)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("06_feature_importance.png", bbox_inches="tight")
plt.show()
print("[OK] Saved -> 06_feature_importance.png")


print("\n--- Sample Predictions ---")
examples = pd.DataFrame({
    "GrLivArea"    : [1500, 2500, 800,  3200],
    "BedroomAbvGr" : [3,    4,    2,    5   ],
    "TotalBaths"   : [2.0,  3.0,  1.0,  3.5 ],
})
examples_scaled              = scaler.transform(examples[FEATURES_ENG].values)
examples["PredictedPrice ($)"] = model.predict(examples_scaled).astype(int)
print(examples.to_string(index=False))

    
print("\n" + "=" * 65)
print("  SUMMARY")
print("=" * 65)
print("  Algorithm  : Multiple Linear Regression")
print("  Features   : {}".format(", ".join(FEATURES_ENG)))
print("  Train R2   : {:.4f}".format(train_metrics[2]))
print("  Test  R2   : {:.4f}".format(test_metrics[2]))
print("  Test  MAE  : ${:,.2f}".format(test_metrics[0]))
print("  Test  RMSE : ${:,.2f}".format(test_metrics[1]))
print("=" * 65)
print("[OK] Task 01 complete. All plots saved as PNG files.")
