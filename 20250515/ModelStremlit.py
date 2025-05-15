import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import fetch_california_housing, load_diabetes, load_iris, load_wine, load_breast_cancer
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import (BaggingRegressor, BaggingClassifier, RandomForestRegressor, RandomForestClassifier,
                              ExtraTreesRegressor, ExtraTreesClassifier, AdaBoostRegressor, AdaBoostClassifier,
                              GradientBoostingRegressor, GradientBoostingClassifier, VotingRegressor, VotingClassifier)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
import numpy as np

st.title("📊 머신러닝 모델 성능 비교")

# 문제 유형 선택
problem_type = st.selectbox("문제 유형을 선택하세요", ["회귀 (Regression)", "분류 (Classification)"])

# 데이터셋 및 모델 설정
X, y, model_options, scoring, dataset_label = None, None, [], None, ""
if problem_type == "회귀 (Regression)":
    dataset_option = st.selectbox("데이터셋을 선택하세요", ["California Housing", "Diabetes"])
    dataset_label = dataset_option
    if dataset_option == "California Housing":
        data = fetch_california_housing()
        X_full, y_full = data.data, data.target
        X, _, y, _ = train_test_split(X_full, y_full, train_size=0.3, random_state=42)
    else:
        data = load_diabetes()
        X, y = data.data, data.target
    model_options = [
        'KNN', 'KNN Bagging', 'SVM', 'SVM Bagging',
        'Decision Tree', 'Decision Tree Bagging', 'Random Forest',
        'Extra Trees', 'AdaBoost', 'Gradient Boosting',
        'Hard Voting', 'Soft Voting'
    ]
    scoring = 'r2'

elif problem_type == "분류 (Classification)":
    dataset_option = st.selectbox("데이터셋을 선택하세요", ["Iris", "Wine", "Breast Cancer"])
    dataset_label = dataset_option
    if dataset_option == "Iris":
        data = load_iris()
    elif dataset_option == "Wine":
        data = load_wine()
    else:
        data = load_breast_cancer()
    X, y = data.data, data.target
    model_options = [
        'KNN', 'KNN Bagging', 'SVM', 'SVM Bagging',
        'Decision Tree', 'Decision Tree Bagging', 'Random Forest',
        'Extra Trees', 'AdaBoost', 'Gradient Boosting',
        'Hard Voting', 'Soft Voting'
    ]
    scoring = 'accuracy'

# 모델 선택 및 실행
if X is not None and y is not None:
    selected_models = st.multiselect("비교할 모델을 선택하세요", model_options)

    if selected_models:
        if 'model_cache' not in st.session_state:
            st.session_state.model_cache = {}
        if 'score_cache' not in st.session_state:
            st.session_state.score_cache = {}
        if 'point_colors' not in st.session_state:
            st.session_state.point_colors = {}
        if 'dataset_colors' not in st.session_state:
            st.session_state.dataset_colors = {}

        def get_models(selected, task):
            model_dict = {}
            for name in selected:
                if name in st.session_state.model_cache:
                    model_dict[name] = st.session_state.model_cache[name]
                    continue

                model = None
                if task == "회귀 (Regression)":
                    if name == 'KNN':
                        model = KNeighborsRegressor()
                    elif name == 'KNN Bagging':
                        model = BaggingRegressor(KNeighborsRegressor(), n_estimators=10, random_state=42)
                    elif name == 'SVM':
                        model = SVR()
                    elif name == 'SVM Bagging':
                        model = BaggingRegressor(SVR(), n_estimators=10, random_state=42)
                    elif name == 'Decision Tree':
                        model = DecisionTreeRegressor(random_state=42)
                    elif name == 'Decision Tree Bagging':
                        model = BaggingRegressor(DecisionTreeRegressor(), n_estimators=10, random_state=42)
                    elif name == 'Random Forest':
                        model = RandomForestRegressor(random_state=42)
                    elif name == 'Extra Trees':
                        model = ExtraTreesRegressor(random_state=42)
                    elif name == 'AdaBoost':
                        model = AdaBoostRegressor(random_state=42)
                    elif name == 'Gradient Boosting':
                        model = GradientBoostingRegressor(random_state=42)
                    elif name in ['Hard Voting', 'Soft Voting']:
                        model = VotingRegressor([
                            ('knn', KNeighborsRegressor()),
                            ('dt', DecisionTreeRegressor()),
                            ('rf', RandomForestRegressor())
                        ])
                elif task == "분류 (Classification)":
                    if name == 'KNN':
                        model = KNeighborsClassifier()
                    elif name == 'KNN Bagging':
                        model = BaggingClassifier(KNeighborsClassifier(), n_estimators=10, random_state=42)
                    elif name == 'SVM':
                        model = SVC(probability=True)
                    elif name == 'SVM Bagging':
                        model = BaggingClassifier(SVC(probability=True), n_estimators=10, random_state=42)
                    elif name == 'Decision Tree':
                        model = DecisionTreeClassifier(random_state=42)
                    elif name == 'Decision Tree Bagging':
                        model = BaggingClassifier(DecisionTreeClassifier(), n_estimators=10, random_state=42)
                    elif name == 'Random Forest':
                        model = RandomForestClassifier(random_state=42)
                    elif name == 'Extra Trees':
                        model = ExtraTreesClassifier(random_state=42)
                    elif name == 'AdaBoost':
                        model = AdaBoostClassifier(random_state=42)
                    elif name == 'Gradient Boosting':
                        model = GradientBoostingClassifier(random_state=42)
                    elif name in ['Hard Voting', 'Soft Voting']:
                        model = VotingClassifier([
                            ('knn', KNeighborsClassifier()),
                            ('dt', DecisionTreeClassifier()),
                            ('rf', RandomForestClassifier())
                        ], voting='hard' if name == 'Hard Voting' else 'soft')

                st.session_state.model_cache[name] = model
                model_dict[name] = model
            return model_dict

        models = get_models(selected_models, problem_type)

        results = {}
        for name in selected_models:
            score_key = f"{name}_{dataset_label}"
            if score_key in st.session_state.score_cache:
                results[name] = st.session_state.score_cache[score_key]
            else:
                pipeline = make_pipeline(StandardScaler(), models[name])
                scores = cross_val_score(pipeline, X, y, scoring=scoring, cv=5)
                results[name] = scores
                st.session_state.score_cache[score_key] = scores

        df_result = pd.DataFrame({f"{k} ({dataset_label})": v for k, v in results.items()})
        mean_scores = df_result.mean().sort_values(ascending=False)

        st.subheader(f"모델별 {'정확도' if problem_type.startswith('분류') else 'R² Score'} (평균)")
        fig, ax = plt.subplots(figsize=(10, 5))

        all_x = []
        all_y = []
        all_colors = []

        custom_palette = {
            'KNN': 'blue',
            'KNN Bagging': 'lightblue',
            'SVM': 'deeppink',
            'SVM Bagging': 'pink',
            'Decision Tree': 'green',
            'Decision Tree Bagging': 'lightgreen',
            'Random Forest': 'orange',
            'Extra Trees': 'gold',
            'AdaBoost': 'red',
            'Gradient Boosting': 'purple',
            'Hard Voting': 'gray',
            'Soft Voting': 'black'
        }

        for col in df_result.columns:
            base_name = col.split(' (')[0]
            scores = df_result[col]
            color = custom_palette.get(base_name, 'gray')
            all_x.extend([col] * len(scores))
            all_y.extend(scores)
            all_colors.extend([color] * len(scores))

        sns.stripplot(x=all_x, y=all_y, size=10, ax=ax, palette=all_colors)
        ax.set_ylabel("정확도" if problem_type.startswith("분류") else "R² Score")
        ax.set_ylim(0, 1.05)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        st.pyplot(fig)

        st.markdown("### 🔍 세부 점수표 (교차검증 Fold별)")
        st.dataframe(df_result.round(4))