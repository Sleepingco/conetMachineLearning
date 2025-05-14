# %%
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

# %%
california_housing = fetch_california_housing(as_frame=True)

X = california_housing.data
y = california_housing.target

X_sample,_,y_sample,_=train_test_split(X,y,train_size=0.1,random_state=42)
print(california_housing. DESCR)

# %%
# 데이터 스케일링
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sample)

# 커널 함수 리스트
kernels = ['linear', 'poly', 'rbf', 'sigmoid']

# 커널 함수별로 학습, 예측, 평가 수행
for kernel in kernels:
    # SVR 모델 초기화
    svr = SVR(kernel=kernel)
    
    # 모델 학습
    svr.fit(X_scaled, y_sample)
    
    # 예측
    y_pred = svr.predict(X_scaled)
    
    # 평가
    mse = mean_squared_error(y_sample, y_pred)
    print(f"Kernel: {kernel}, Mean Squared Error: {mse}")


# %%
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import numpy as np

# SVR 모델 초기화
svr = SVR()

# 파라미터 그리드 설정
param_grid = {
    'kernel': ['linear', 'rbf', 'sigmoid'],
    'C': np.logspace(-2, 3, 6),         # [0.01, 0.1, 1, 10, 100, 1000]
    'gamma': np.logspace(-4, 1, 6),     # [0.0001, 0.001, 0.01, 0.1, 1, 10]
}

# GridSearchCV를 이용한 파라미터 튜닝
grid_search = GridSearchCV(svr, param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X_scaled, y_sample)

# 최적 파라미터 및 성능 출력
best_params = grid_search.best_params_
best_score = -grid_search.best_score_  # 음수 MSE이므로 부호 반전
print(f"Best Parameters: {best_params}")
print(f"Best CV MSE: {best_score}")


# %%
# 최적의 파라미터와 최적의 모델 출력
print("Best Parameters:", grid_search.best_params_)
print("Best Model:", grid_search.best_estimator_)

# 최적의 모델로 예측
y_pred = grid_search.best_estimator_.predict(X_scaled)

# 평가
mse = mean_squared_error(y_sample, y_pred)
print("Mean Squared Error:", mse)
from sklearn.metrics import r2_score

r2 = r2_score(y_sample, y_pred)
print("R² Score:", r2)




