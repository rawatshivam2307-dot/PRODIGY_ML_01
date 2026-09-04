import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("train.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

df["TotalBathrooms"] = df["FullBath"] + (0.5 * df["HalfBath"])

X = df[[
    "GrLivArea",
    "BedroomAbvGr",
    "TotalBathrooms"
]]

# House price
y = df["SalePrice"]

data = pd.concat([X, y], axis=1)

data = data.dropna()

X = data[[
    "GrLivArea",
    "BedroomAbvGr",
    "TotalBathrooms"
]]

y = data["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel trained successfully!")

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)


print("\n========== MODEL PERFORMANCE ==========")

print("MAE :", round(mae, 2))
print("MSE :", round(mse, 2))
print("RMSE:", round(rmse, 2))
print("R²  :", round(r2, 4))

print("\n========== MODEL COEFFICIENTS ==========")
print("Intercept:", round(model.intercept_, 2))

for feature, coefficient in zip(X.columns, model.coef_):
    print(feature, ":", round(coefficient, 2))

new_house = pd.DataFrame({
    "GrLivArea": [2000],
    "BedroomAbvGr": [3],
    "TotalBathrooms": [2]
})

predicted_price = model.predict(new_house)

print("\n========== NEW HOUSE PREDICTION ==========")

print(
    "Predicted House Price:",
    round(predicted_price[0], 2)
)

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")

plt.title("Actual vs Predicted House Prices")

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.show()

residuals = y_test - y_pred

plt.figure(figsize=(8, 6))

plt.scatter(y_pred, residuals)

plt.axhline(0)

plt.xlabel("Predicted House Prices")

plt.ylabel("Residuals")

plt.title("Residual Plot")

plt.show()