# %%
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
cancer = load_breast_cancer()
X = cancer. data
y = cancer.target

# %%
#데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

#커널 종류 리스트

kernel_list = ['linear', 'poly', 'rbf']

#각 커널 종류에 대해 모델 학습 및 평가
for kernel in kernel_list:
    # SVM 모델 초기화
    svm = SVC(kernel=kernel)

    # 모델 학습
    svm.fit(X_train, y_train)

    #예측
    y_pred = svm.predict (X_test)
    # 분류 보고서 출력
    print (f"Kernel: {kernel}")
    print (classification_report(y_test, y_pred))
    print("="*50)

# %%
import pandas as pd

df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df['target'] = pd.Series(cancer.target)

df.head()

# %%
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 데이터 표준화
scaler = MinMaxScaler()
X_scaled= scaler.fit_transform(X)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y,test_size=0.2, random_state=42)

#SVM 모델 초기화
svm = SVC(kernel='rbf')

# 모델 학습
svm.fit(X_train, y_train)

#예측
y_pred = svm.predict (X_test)

# 정확도 평가
print (classification_report(y_test, y_pred))

# %%
import streamlit as st
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Title
st.title("California Housing: 회귀 모델 성능 비교")

# 데이터 로드
@st.cache_data
def load_data():
    data = fetch_california_housing(as_frame=True)
    return data.data, data.target

X, y = load_data()

# 스케일러 선택
scaler_option = st.selectbox("스케일러 선택", ["StandardScaler", "MinMaxScaler"])
scaler = StandardScaler() if scaler_option == "StandardScaler" else MinMaxScaler()

# 데이터 분할 및 스케일링
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 전체 모델 비교 실행 여부
compare_all = st.checkbox("모든 모델 비교 실행")

# 결과 저장 리스트
comparison_results = []

if compare_all:
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.001, max_iter=10000),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5),
        "KNN": KNeighborsRegressor(n_neighbors=9, weights='distance'),
        "SVR": SVR(C=10, gamma=0.01, kernel='rbf')
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (name, model) in enumerate(models.items()):
        model.fit(X_train_scaled, y_train)
        
        best_model = model.best_estimator_ if isinstance(model, GridSearchCV) else model
        y_pred = best_model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        comparison_results.append({"Model": name, "MSE": round(mse, 4), "R2": round(r2, 4), "Best Params": "-"})

        axes[idx].scatter(y_test, y_pred, alpha=0.5)
        axes[idx].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
        axes[idx].set_title(f"{name}")
        axes[idx].set_xlabel("Actual")
        axes[idx].set_ylabel("Predicted")

    st.subheader("모든 모델 비교 결과")
    comparison_df = pd.DataFrame(comparison_results)
    st.dataframe(comparison_df.set_index("Model"))
    
    st.subheader("모든 모델 예측 시각화 (2x3)")
    plt.tight_layout()
    st.pyplot(fig)

else:
    # 모델 선택
    model_choice = st.selectbox("모델 선택", ["Linear Models", "KNN Regressor", "SVR"])

    if model_choice == "Linear Models":
        linear_type = st.selectbox("선형 회귀 모델 선택", ["LinearRegression", "Ridge", "Lasso", "ElasticNet"])

        if linear_type == "LinearRegression":
            model = LinearRegression()
        elif linear_type == "Ridge":
            alpha = st.slider("alpha (Ridge)", 0.01, 10.0, 1.0)
            model = Ridge(alpha=alpha)
        elif linear_type == "Lasso":
            alpha = st.slider("alpha (Lasso)", 0.0001, 10.0, 1.0)
            model = Lasso(alpha=alpha)
        elif linear_type == "ElasticNet":
            alpha = st.slider("alpha (ElasticNet)", 0.01, 10.0, 1.0)
            l1_ratio = st.slider("l1_ratio", 0.0, 1.0, 0.5)
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        st.subheader(f"{linear_type} 회귀 결과")

    elif model_choice == "KNN Regressor":
        n_neighbors = st.slider("n_neighbors", 1, 20, 5)
        weights = st.selectbox("weights", ["uniform", "distance"])
        model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        st.subheader("KNN 회귀 결과")
        st.text(f"Parameters: n_neighbors={n_neighbors}, weights={weights}")

    elif model_choice == "SVR":
        C = st.selectbox("C", [0.1, 1, 10, 100], index=2)
        gamma = st.selectbox("gamma", [0.0001, 0.001, 0.01, 0.1, 1], index=2)
        kernel = st.selectbox("kernel", ["linear", "poly", "rbf", "sigmoid"], index=2)
        model = SVR(C=C, gamma=gamma, kernel=kernel)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        st.subheader("SVM 회귀 결과")
        st.text(f"Parameters: C={C}, gamma={gamma}, kernel={kernel}")

    # 평가 결과 출력
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write(f"Mean Squared Error (MSE): {mse:.4f}")
    st.write(f"R² Score: {r2:.4f}")

    # 예측 결과 시각화
    st.subheader("예측 결과 시각화")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)

    # 성능 요약표 출력
    st.subheader("모델 성능 요약")
    result_df = pd.DataFrame({
        "Metric": ["Mean Squared Error", "R² Score"],
        "Value": [round(mse, 4), round(r2, 4)]
    })
    st.table(result_df.set_index("Metric"))



