import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
import streamlit as st

# 데이터셋 로드
housing = fetch_california_housing(as_frame=True)
data = housing.frame
X = data.drop('MedHouseVal', axis=1)
y = data['MedHouseVal']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Streamlit 앱 설정
st.title("California Housing Price Prediction")
st.write("This app trains regression models on the California Housing dataset and displays R² score and RMSE.")

model_name = st.selectbox("Select Regression Model", ["Lasso", "Ridge", "Elastic Net", "Polynomial Regression"])

# 스케일러 선택 함수
def get_scaler(scaler_name):
    if scaler_name == "StandardScaler":
        return StandardScaler()
    elif scaler_name == "MinMaxScaler":
        return MinMaxScaler()
    elif scaler_name == "RobustScaler":
        return RobustScaler()

# 공통 스케일러 선택
scaler_name = st.selectbox("Select a Scaler", ["StandardScaler", "MinMaxScaler", "RobustScaler"])
scaler = get_scaler(scaler_name)

# Polynomial Features 처리
if model_name == "Polynomial Regression":
    degree = st.slider("Select Polynomial Degree", min_value=1, max_value=5, value=2)
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    X_train_scaled = scaler.fit_transform(X_train_poly)
    X_test_scaled = scaler.transform(X_test_poly)
else:
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

# 모델 선택 및 파라미터
if model_name == "Lasso":
    alpha_lasso = st.slider("Select Lasso Alpha", 0.001, 10.0, 1.0, step=0.001)
    model = Lasso(alpha=alpha_lasso)
elif model_name == "Ridge":
    alpha_ridge = st.slider("Select Ridge Alpha", 0.001, 10.0, 1.0, step=0.001)
    model = Ridge(alpha=alpha_ridge)
elif model_name == "Elastic Net":
    alpha_elastic = st.slider("Select Elastic Net Alpha", 0.001, 10.0, 1.0, step=0.001)
    l1_ratio = st.slider("Select Elastic Net L1 Ratio", 0.1, 0.9, 0.5, step=0.1)
    model = ElasticNet(alpha=alpha_elastic, l1_ratio=l1_ratio)
elif model_name == "Polynomial Regression":
    model = LinearRegression()

# 모델 학습
try:
    # 모델 학습 (스케일 적용)
    model.fit(X_train_scaled, y_train)

    # 예측
    y_pred = model.predict(X_test_scaled)


    # 성능 평가
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # 결과 출력
    st.write(f"### {model_name} Results")
    if model_name == "Lasso":
        st.write(f"Alpha: {alpha_lasso}")
    elif model_name == "Ridge":
        st.write(f"Alpha: {alpha_ridge}")
    elif model_name == "Elastic Net":
        st.write(f"Alpha: {alpha_elastic}, L1 Ratio: {l1_ratio}")
    elif model_name == "Polynomial Regression":
        st.write(f"Degree: {degree}")

    st.write(f"R² Score: {r2:.4f}")
    st.write(f"RMSE: {rmse:.4f}")

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Actual vs Predicted Values")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.scatter(y_test, y_pred, color='blue', alpha=0.5)
        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax1.set_xlabel('Actual Values')
        ax1.set_ylabel('Predicted Values')
        ax1.set_title(f'{model_name} - Actual vs Predicted')
        st.pyplot(fig1)

    with col2:
        st.write("### 잔차(residual) 분포 그래프")
        st.write("예측값과 실제값의 차이(잔차)가 어떻게 분포하는지 시각화하여 모델의 편향 여부와 오차 분포를 평가할 수 있습니다.")
        residuals = y_test - y_pred
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.hist(residuals, bins=30, edgecolor='black')
        ax2.set_title("Residuals Distribution")
        ax2.set_xlabel("Residual")
        ax2.set_ylabel("Frequency")
        st.pyplot(fig2)

    col3, col4 = st.columns(2)

    with col3:
        st.write("### Feature 중요도 시각화")
        st.write("모델이 어떤 특성(feature)에 더 많은 가중치를 부여했는지 확인함으로써 모델 해석력 향상.")
        if model_name in ["Lasso", "Ridge", "Elastic Net", "Polynomial Regression"]:
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            coef = model.coef_
            feature_names = poly.get_feature_names_out(X.columns) if model_name == "Polynomial Regression" else X.columns
            ax3.barh(feature_names, coef)
            ax3.set_title("Feature Importance")
            st.pyplot(fig3)

    with col4:
        st.write("### 예측값 vs 잔차 산점도 (Residual Plot)")
        st.write("예측값과 잔차 사이의 관계를 시각화해서, 잔차가 무작위로 분포하지 않는다면 모델이 시스템적으로 잘못된 예측을 하고 있음을 확인할 수 있습니다")
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        ax4.scatter(y_pred, residuals, alpha=0.5)
        ax4.hlines(0, y_pred.min(), y_pred.max(), color='red')
        ax4.set_xlabel("Predicted Values")
        ax4.set_ylabel("Residuals")
        ax4.set_title("Residuals vs Predicted Values")
        st.pyplot(fig4)


except Exception as e:
    st.error(f"An error occurred: {str(e)}")


# 모델 성능 비교 테이블
if st.button("Compare All Models"):
    

    def evaluate_model(model, X_train, X_test, y_train, y_test):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        return r2, rmse

    models = {
        "Lasso": Lasso(alpha=0.1),
        "Ridge": Ridge(alpha=1.0),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5),
        "PolynomialRegression": LinearRegression()
    }

    results = []
    for name, model in models.items():
        if name == "PolynomialRegression":
            degree =1
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            X_train_poly = poly.fit_transform(X_train)
            X_test_poly = poly.transform(X_test)
            X_train_scaled = scaler.fit_transform(X_train_poly)
            X_test_scaled = scaler.transform(X_test_poly)
        else:
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

        r2, rmse = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test)
        results.append({"Model": name, "R² Score": round(r2, 4), "RMSE": round(rmse, 4)})

    st.write("### 📊 모델별 성능 비교")
    st.dataframe(pd.DataFrame(results))

# 데이터셋 정보
st.write("### Dataset Info")
st.write(f"Features: {X.columns.tolist()}")
st.write(f"Number of Samples: {len(data)}")
st.write(f"Target: Median House Value (in $100,000s)")

# 재실행 버튼
if st.button("Retrain Model"):
    st.rerun()
