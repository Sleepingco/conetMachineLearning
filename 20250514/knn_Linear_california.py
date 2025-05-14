# %%
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# %%
# 캘리포니아 집값 데이터셋 로드
california_housing = fetch_california_housing(as_frame=True)

X = california_housing.data
y = california_housing.target

X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)

# %%
knn = KNeighborsRegressor(n_neighbors=5,weights='uniform',algorithm='auto',n_jobs=1)
knn.fit(X_train,y_train)
# 분류 결과 예측
y_pred = knn.predict(X_test)

# 분류 결과 평가
mse = mean_squared_error(y_test,y_pred)
print(mse)


# %%
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsRegressor())
])
# 탐색할 파라미터 그리드 정의
param_grid = {
    'n_neighbors' : [3,5,7], # 이웃의 수
    'weights': ['uniform', 'distance'],
    'algorithm':['ball_tree','kd_tree','brute'] # 가중치 함수
}

grid_search = GridSearchCV(knn,param_grid,cv=5)
grid_search.fit(X_train,y_train)

# %%
from sklearn.metrics import mean_squared_error, r2_score

# 최적 모델과 파라미터 출력
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# 테스트 데이터 예측
y_pred = best_model.predict(X_test)

# MSE 계산
mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)

# R² 계산
r2 = r2_score(y_test, y_pred)
print("R²:", r2)

# %%
# %%
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 데이터 로드
california_housing = fetch_california_housing(as_frame=True)
X = california_housing.data
y = california_housing.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)

# 파이프라인 구성 (스케일링 + KNN)
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsRegressor())
])

# 파라미터 그리드
param_grid = {
    'knn__n_neighbors': [3, 5, 7, 9, 11],
    'knn__weights': ['uniform', 'distance'],
    'knn__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}

# GridSearchCV 수행
grid_search = GridSearchCV(pipe, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# 최적 모델 추출 및 평가
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
print("✅ Best Parameters:", best_params)

y_pred = best_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("📌 Test MSE:", mse)
print("📌 Test R² Score:", r2)



