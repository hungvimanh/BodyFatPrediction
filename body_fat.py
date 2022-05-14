import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error
import xgboost as XGB

df = pd.read_csv('Data.csv')


data = data.drop(columns=['BodyFat','Density'], axis=1)
features = list(data.columns)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#linear regression
scores = []
regr = LinearRegression()
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
score = regr.score(X_test, y_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"Score --> {score}")
print(f"RMSE --> {rmse}")

score = {"name": "linear_regression", "score": score, "rmse": rmse}
scores.append(score)

#lasso
lasso_regr = Lasso(alpha=0.5)
lasso_regr.fit(X_train, y_train)
cv_score = cross_val_score(lasso_regr, X_train, y_train, cv=10)
print(f"CV Score --> {np.mean(cv_score)}")
y_pred = lasso_regr.predict(X_test)
print(f"Score --> {lasso_regr.score(X_test, y_test)}")
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"RMSE --> {rmse}")

score = {"name": "lasso", "score": np.mean(cv_score), "rmse": rmse}
scores.append(score)

#ridge
ridge_regr = Ridge(alpha=0.5)
ridge_regr.fit(X_train, y_train)
cv_score = cross_val_score(ridge_regr, X_train, y_train, cv=10)
print(f"CV Score --> {np.mean(cv_score)}")
y_pred = ridge_regr.predict(X_test)
print(f"Score --> {ridge_regr.score(X_test, y_test)}")
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"RMSE --> {rmse}")

score = {"name": "ridge", "score": np.mean(cv_score), "rmse": rmse}
scores.append(score)

#XGboost
xgb_regr = XGB.XGBRegressor(learning_rate = 0.01, n_estimators=1000)
xgb_regr.fit(X_train, y_train)
cv_score = cross_val_score(xgb_regr, X_train, y_train, cv=10)
print(f"CV Score --> {np.mean(cv_score)}")
y_pred = regr.predict(X_test)
print(f"Score --> {xgb_regr.score(X_test, y_test)}")
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"RMSE --> {rmse}")

score = {"name": "XGBoost", "score": np.mean(cv_score), "rmse": rmse}
scores.append(score)

print(scores)


