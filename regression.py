# імпортуємо необхідні бібліотеки для регресії
import numpy as np
import pandas as pd

#функції метрик якості
def RMSE(y_test, y_pred):
    return (sum((y_test - y_pred)**2)/len(y_test))**(0.5)
def MAE(y_test, y_pred):
    return np.mean(np.abs(y_test - y_pred))
def mape(y_test, y_pred): 
    return np.mean(np.abs((y_test+1 - y_pred) / (y_test+1))) * 100

# імпортуємо датасет
dataset = pd.read_excel(r"C:\Users\podlo\OneDrive\Документы\__learn_aspirantura_kneu\_4_semestr\2023-11-20_bigdata\dataset_regression.xlsx")

# переглядаємо датасет
print(dataset.shape)
print(dataset.head())
print(dataset.info())

# формуємо вибірки даних
data_y = dataset['gdp_per_capita'].values
data_x = dataset.select_dtypes(exclude=['object']).drop(columns=['gdp_per_capita']).values

# розбиваємо вибірки даних на тестову та трейнову
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size = 0.2, random_state=42)

# регресійний аналіз
# -----------------------------------------------------------------------------
# KNeighborsRegressor
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors = 10, n_jobs=-1) 
knn.fit(train_x, train_y)
np.sort(knn.predict(test_x))

print("RMSE: ",RMSE(test_y, knn.predict(test_x)))
print("MAE: ",MAE(test_y, knn.predict(test_x)))
print("MAPE: ",mape(test_y, knn.predict(test_x)))

from sklearn.metrics import r2_score
pred_y = knn.predict(test_x)
print("R2: ",r2_score(test_y, pred_y))

# підбір параметрів
for i in range(1,11):
    knn = KNeighborsRegressor(n_neighbors = i, metric='euclidean', n_jobs=-1)
    knn.fit(train_x, train_y)
    pred_y =  knn.predict(test_x)
    print(i, "- RMSE:",  RMSE(test_y, pred_y), "MAPE:", mape(test_y, pred_y), "R^2:", r2_score(test_y, pred_y), "MAE:", MAE(test_y, pred_y))

# аналогічно з логарифмуванням
knn.fit(train_x, np.log(train_y))
np.exp(np.sort(knn.predict(test_x)))

print("RMSE: ",RMSE(test_y, knn.predict(test_x)))
print("MAE: ",MAE(test_y, knn.predict(test_x)))
print("MAPE: ",mape(test_y, knn.predict(test_x)))

pred_y = knn.predict(test_x)
print("R2: ",r2_score(test_y, pred_y))

# аналогічно з шкалюванням
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_x)
x_train_scaled = scaler.transform(train_x)
x_test_scaled = scaler.transform(test_x)

knn.fit(x_train_scaled, np.log1p(train_y))
pred_y = knn.predict(x_test_scaled)

print("RMSE: ",RMSE(test_y, np.exp(pred_y)))
print("MAE: ",MAE(test_y, np.exp(pred_y)))
print("MAPE: ",mape(test_y, np.exp(pred_y)))
print("R2: ",r2_score(test_y, np.exp(pred_y)))

# -----------------------------------------------------------------------------
# LinearRegression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression(n_jobs=-1)
lin_reg.fit(train_x[:,:], train_y)
lin_reg.intercept_, lin_reg.coef_
pred_y = lin_reg.predict(test_x)

print("RMSE: ",RMSE(test_y, pred_y))
print("MAE: ",MAE(test_y, pred_y))
print("MAPE: ",mape(test_y, pred_y))
print("R2: ",r2_score(test_y, pred_y))

# без константи
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression(n_jobs=-1, fit_intercept=False)
lin_reg.fit(train_x, train_y)
lin_reg.intercept_, lin_reg.coef_
pred_y = lin_reg.predict(test_x)

print("RMSE: ",RMSE(test_y, pred_y))
print("MAE: ",MAE(test_y, pred_y))
print("MAPE: ",mape(test_y, pred_y))
print("R2: ",r2_score(test_y, pred_y))

# тільки додатні значення оцінок параметрів
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression(n_jobs=-1, fit_intercept=False, positive=True)
lin_reg.fit(train_x, train_y)
lin_reg.intercept_, lin_reg.coef_
pred_y = lin_reg.predict(test_x)

print("RMSE: ",RMSE(test_y, pred_y))
print("MAE: ",MAE(test_y, pred_y))
print("MAPE: ",mape(test_y, pred_y))
print("R2: ",r2_score(test_y, pred_y))

# -----------------------------------------------------------------------------
# регуляризовані методи регресії
from sklearn.linear_model import Ridge, Lasso, ElasticNet
reg = Ridge(alpha=0.5)

# Ridge
from sklearn.preprocessing import PolynomialFeatures
for i in [0, 0.000001, 0.1, 0.5, 1, 2, 3,4, 5, 6, 7, 11, 20, 100, 1000]:
    poly_features = PolynomialFeatures(degree=2, include_bias=False) 
    x_poly = poly_features.fit_transform(train_x)
    x_poly_test = poly_features.fit_transform(test_x)
    lin_reg = Ridge(alpha=i)
    lin_reg.fit(x_poly, train_y)
    pred_y = np.round(lin_reg.predict(x_poly_test))
    pred_y[pred_y <= 0] = 0
    print(str(i)+": "+str(round(MAE(test_y, pred_y),4))+" R^2: "+ str(round(r2_score(test_y, pred_y),4)))
    
# Lasso
for i in [0, 0.000001, 0.1, 0.5, 1, 2, 3,4, 5, 6, 7, 11, 20, 100, 1000]:
    poly_features = PolynomialFeatures(degree=2, include_bias=False) 
    x_poly = poly_features.fit_transform(train_x)
    x_poly_test = poly_features.fit_transform(test_x)
    lin_reg = Lasso(alpha=i)
    lin_reg.fit(x_poly, train_y)
    pred_y = np.round(lin_reg.predict(x_poly_test))
    pred_y[pred_y <= 0] = 0
    print(str(i)+": "+str(round(MAE(test_y, pred_y),4))+" R^2: "+ str(round(r2_score(test_y, pred_y),4)))

# ElasticNet
for i in [0, 0.000001, 0.1, 0.5, 1, 2, 3,4, 5, 6, 7, 11, 20, 100, 1000]:
    poly_features = PolynomialFeatures(degree=2, include_bias=False) 
    x_poly = poly_features.fit_transform(train_x)
    x_poly_test = poly_features.fit_transform(test_x)
    lin_reg = ElasticNet(l1_ratio=0.6, alpha=i)# l1_ratio=0 - Ridge, l1_ratio=1 - Lasso
    lin_reg.fit(x_poly, train_y)
    pred_y = np.round(lin_reg.predict(x_poly_test))
    pred_y[pred_y <= 0] = 0
    print(str(i)+": "+str(round(MAE(test_y, pred_y),4))+" R^2: "+ str(round(r2_score(test_y, pred_y),4)))

# -----------------------------------------------------------------------------
# Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor(
    criterion='friedman_mse',
    max_depth=18, 
    min_samples_leaf=29, 
    max_features=0.9, 
    random_state=42)
dt_reg.fit(train_x, train_y)
pred_y = dt_reg.predict(test_x)

print("RMSE: ",RMSE(test_y, pred_y))
print("MAE: ",MAE(test_y, pred_y))
print("MAPE: ",mape(test_y, pred_y))
print("R2: ",r2_score(test_y, pred_y))
