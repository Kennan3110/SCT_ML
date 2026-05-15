
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

df = pd.read_csv("house_prices.csv")
print("Dataset shape:", df.shape)
print(df.head())

features = ["GrLivArea", "BedroomAbvGr", "FullBath", "HalfBath"]
X = df[features]
y = df["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"\nMAE:  {mean_absolute_error(y_test, y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R2:   {r2_score(y_test, y_pred):.4f}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("House Price Prediction - Actual vs Predicted")
plt.savefig("prediction_plot.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nFeature Coefficients:")
for f, c in zip(features, model.coef_):
    print(f"  {f}: {c:.2f}")
print(f"  Intercept: {model.intercept_:.2f}")
