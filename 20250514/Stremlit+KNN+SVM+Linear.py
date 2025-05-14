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

st.markdown(
    """
    <h1 style='text-align: center;'>
        California Housing:<br>회귀 모델 성능 비교
    </h1>
    """,
    unsafe_allow_html=True
)


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
        comparison_results.append({"Model": name, "MSE": round(mse, 4), "R2": round(r2, 4)})

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
st.markdown("""
### 모델별 주요 특징 및 장단점 요약

- **Linear Regression**: 가장 기본적인 선형 회귀 모델로, 해석이 간단하고 빠르지만 복잡한 패턴을 학습하는 데는 한계가 있습니다.  
- **Ridge Regression**: L2 정규화를 통해 다중공선성을 완화하고 과적합을 방지합니다. 모든 특성을 활용하되 계수를 축소합니다.  
- **Lasso Regression**: L1 정규화를 통해 특성 선택 기능을 수행합니다. 일부 계수를 0으로 만들어 모델을 간결하게 하지만, 중요한 특성까지 제거될 수 있습니다.  
- **ElasticNet**: L1과 L2를 혼합한 정규화 모델로 Ridge와 Lasso의 장점을 적절히 조합합니다. 특성 선택과 안정적인 일반화를 동시에 기대할 수 있습니다.  
- **KNN Regressor**: 거리 기반 예측 모델로, 비선형적인 관계도 포착 가능합니다. 하지만 데이터가 많아질수록 예측이 느려지고 이상치에 민감할 수 있습니다.  
- **SVR (Support Vector Regression)**: 마진 내 오류를 허용하면서 복잡한 비선형 회귀도 처리 가능. 하이퍼파라미터에 민감하고 학습 시간이 상대적으로 오래 걸립니다.  
""")