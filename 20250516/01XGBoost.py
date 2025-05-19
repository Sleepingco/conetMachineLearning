# %%
from sklearn.datasets import load_iris,load_breast_cancer,fetch_california_housing,load_diabetes
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score, precision_score,recall_score

import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
from xgboost import plot_importance, plot_tree
import numpy as np

import graphviz
import matplotlib.pyplot as plt

# %%
cancer_data = load_breast_cancer()
X_train,X_test, y_train,y_test = train_test_split(
    cancer_data.data,cancer_data.target,test_size=0.3,random_state=123
)
cancer_data

# %%
dmtrain = xgb.DMatrix(data=X_train,label=y_train)
dmtest = xgb.DMatrix(data=X_test,label=y_test)

# %% [markdown]
# Dmatrix : Numpy 입력 파라미터로 만들어지는
# XGBoost의 전용 데이터 세트.
# Parameter : data(feature data set), label(label data/target data)

# %%
params = {
    'max_depth': 3,
    'eta':0.15,
    'objective': 'binary:logistic',
    'eval_metric':'error'
}
num_rounds = 500

# %% [markdown]
# Parameters 입력을 딕셔너리 형태로 지정하여 train 메서드에 입력
# eval_metric 설명(이진분류 기준)
# logloss Logarithmic Loss — 확률 기반 예측 평가, 낮을수록 좋음
# error 분류 오류율 (예: 0.1이면 10% 틀림)
# auc Area Under ROC Curve — 1에 가까울수록 좋음
# aucpr Area Under Precision-Recall Curve

# %%
evals = [(dmtrain,'train'),(dmtest,'test')]
xgb_model = xgb.train(
    params=params,
    dtrain = dmtrain,
    num_boost_round = num_rounds,
    early_stopping_rounds=20,
    evals=evals
)

# %% [markdown]
# early_stopping_rounds=20 : validation set(metric)의 점수가 "현재까지 가장 좋은 값(best
# score)"보다 작아야지만 개선으로 간주된다

# %%
predict_probability = xgb_model.predict(dmtest)
print(np.round(predict_probability[:10],3))

# %%
predict_integer = [1 if p > 0.5 else 0 for p in predict_probability]
print(predict_integer[:10])

# %% [markdown]
# [:10] : 전체 예측 결과 중 앞에서 10개만 출력
# dmtest를 사용한 모델의 예측값은 확률(실수)로 주어지므로
# 정수형의 변환이 필요할 수 있다.

# %%
print('Accuracy : {}'.format(accuracy_score(y_test,predict_integer)))
print('precision : {}'.format(precision_score(y_test,predict_integer)))
print('Accuracy : {}'.format(recall_score(y_test,predict_integer)))

# %% [markdown]
# 실제값(y_test)과 예측값(predict_integer)를 비교한 후
# 모델의 정확도/정밀도/재현율을 출력해 본다.

# %%
import pandas as pd
features = cancer_data.data
label = cancer_data.target

cancer_df = pd.DataFrame(data=features,columns=cancer_data.feature_names)
cancer_df['target'] = label
cancer_df.head()

# %% [markdown]
# DataFrame으로 변환 후 Breast cancer data의 features를 확인 !!

# %%
cancer_df.info()

# %% [markdown]
# Breast cancer data의 features를 확인 !!

# %%
cancer_df.describe()

# %%
fig, ax = plt.subplots(figsize = (10,12))
plot_importance(xgb_model,ax=ax)

# %% [markdown]
# feature(f21)이 예측결과에
# 가장 많은 영향(?)을 미쳐서
# 상대적으로
# 중요한 feature임을 보여준다.
# (relative importance)

# %%
xgb.to_graphviz(xgb_model)

# %% [markdown]
# 입력 feature이 순서적 관점에서 선택되므로
# 그 출력 결과값(prediction)이
# Plot_importance와 꼭 매칭되지는 않는다.
# (the order of features in modeling)

# %%
xgb.to_graphviz(xgb_model,num_trees=1,rankdir='LR',size='15,15')

# %% [markdown]
# 입력 feature이 순서적 관점에서
# 선택되므로
# 그 출력 결과값(prediction)이
# Plot_importance와 꼭 매칭되지는 않는다.
# (the order of features in modeling)
# 
# scikit learn pkl로 저장해서 내가 원하는 데이터를 테스트 가능

# %%
iris = load_iris()

X_train, X_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.25,random_state=42)


# %%
xgbc_model = XGBClassifier(n_estimators = 200, learning_rate = 0.01, max_depth=2,random_state=42)
xgbc_model.fit(X_train,y_train)
preds=xgbc_model.predict(X_test)
preds_probability = xgbc_model.predict_proba(X_test)[:,:]
print(preds_probability)


# %% [markdown]
# predict_proba() : 분류 모델이 각 클래스에 대해 예측한 확률값(Probability)을 반환하는 함수

# %%
accuracy = xgbc_model.score(X_test,y_test)
print("Accuracy : %.2f"%accuracy)

# %%
from sklearn.metrics import classification_report
report = classification_report(y_test,preds)
print(report)

# %%
features = iris.data
label = iris.target

iris_df =pd.DataFrame(data=features,columns=iris.feature_names)
iris_df['target']=label
iris_df.head()

# %% [markdown]
# DataFrame으로 변환 후 주요 특징값을 확인하기 위해 정보 출력 !

# %%
iris_df.info()

# %%
fig,ax=plt.subplots(figsize=(10,10))
plot_importance(xgbc_model, ax=ax)

# %% [markdown]
# 중요도 순으로 정리된
# 붓꽃 데이터의 특징
# (f1 > F2 > f0 > f3)

# %%
xgb.to_graphviz(xgb_model)

# %%
xgb.to_graphviz(xgbc_model,num_trees=0,rankdir='LR',size='10,10')

# %% [markdown]
# 의사결정나무를 그래픽으로 시각화
# (만들어진 트리 중 첫번째 트리 : num_trees =0)

# %%
xgb.to_graphviz(xgbc_model,num_trees=5,rankdir='LR',size='10,10')

# %%
xgb.to_graphviz(xgbc_model,num_trees=15,rankdir='LR',size='10,10')

# %%
xgb.to_graphviz(xgbc_model,num_trees=50,rankdir='LR',size='10,10')

# %%
california = fetch_california_housing()
X_train,X_test,y_train,y_test = train_test_split(california.data,california.target,test_size=0.25,random_state=123)

# %% [markdown]
# 캘리포니아 집값 데이터의 업로드 및 자료 분할

# %%
xgbr_model = XGBRegressor(n_estimators =20, learning_rate=0.2,objective='reg:squarederror',max_depth=4,random_state=123)

# %%
xgbr_model.fit(X_train,y_train)
preds = xgbr_model.predict(X_test)

# %% [markdown]
# 모델의 설정 및 예측 : regression의 MSE 오차를 사용

# %%
accuracy = xgbr_model.score(X_test,y_test)
print("Accuracy: %.2f"%accuracy)

# %%
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test,preds)
print("MSE: %.2f"%mae)

# %% [markdown]
# 
# 분류의 문제는
# XGBClassifier를 사용하여
# Precision/recall을 비교하나
# 회귀의 문제는
# 오류의 정도를 구하므로
# MAE, RMSE 등을 사용해야

# %%
features = california.data
label = california.target
california_df = pd.DataFrame(data = features, columns= california.feature_names)
california_df['target'] = label
california_df.head()

# %% [markdown]
# DataFrame으로 변환 후 특징값을 확인 !!

# %%
california_df.info()

# %% [markdown]
# 특징값을 확인 !!(f0 ~ f7)

# %%
fig,ax = plt.subplots(figsize =(10,10))
plot_importance(xgbr_model,ax=ax)

# %%
xgb.to_graphviz(xgbr_model)

# %%
xgb.to_graphviz(xgbr_model,num_trees=0,rankdir='LR',size='20,20')

# %%
xgb.to_graphviz(xgbr_model,num_trees=10,rankdir='LR',size='20,20')


