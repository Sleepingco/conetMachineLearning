# %% [markdown]
# 2) ML 주요 이론
# (LGBM : Light GBM XGBoost)
# (LGBMClassifier/LGBMRegressor)
# - 객체와 변수의 수를 감소시키며 효율적인 split point를 찾아간다
# - 중요하지 않은 객체는 제거(Large/small gradient)
# - Exclusive feature bundling 변수 감소

# %%
from lightgbm import LGBMClassifier, LGBMRegressor
from lightgbm import plot_importance,plot_metric,plot_tree
import lightgbm as lgb
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
iris = load_iris()
X_train,X_test, y_train,y_test = train_test_split(
    iris.data,iris.target,test_size=0.25,random_state=123
)
iris

# %%
lgbmc_model = LGBMClassifier(n_estimators= 3500)
evals = [(X_test,y_test)]
lgbmc_model.fit(X_train,y_train,eval_metric = 'logloss',eval_set=evals)
preds = lgbmc_model.predict(X_test)

# %%
cross_validation = cross_validate(estimator= lgbmc_model,
                                  X=iris.data,y=iris.target,
                                  cv=5)
print('AVG fit time: {}'.format(cross_validation['fit_time'].mean()))
print('AVG score time: {}'.format(cross_validation['score_time'].mean()))
print('AVG test time: {}'.format(cross_validation['test_score'].mean()))

# %% [markdown]
# In the context of using LightGBM,
# "fit time" refers to the time it takes to train the model on the training data,
# "score time" refers to the time it takes to evaluate the model on the test data,
# "test score" refers to the model's performance on the test data.

# %%
features = iris.data
label = iris.target
import pandas as pd
iris_df = pd.DataFrame(data=features,columns=iris.feature_names)
iris_df['target']=label
iris_df.head()

# %%
lgb.plot_importance(lgbmc_model,figsize=(10,10))

# %%
lgb.plot_tree(lgbmc_model,figsize=(20,10))

# %%
from lightgbm import LGBMClassifier, LGBMRegressor
from lightgbm import plot_importance,plot_metric,plot_tree

# %%
california = fetch_california_housing()
X_train,X_test,y_train,y_test = train_test_split(california.data,california.target,test_size=0.25,random_state=123)

# %%
lgbmr_model = LGBMRegressor(n_estimators=1000)
evals  = [(X_test,y_test)]
lgbmr_model.fit(X_train,y_train,
                eval_metric='mse',
                eval_set=evals)
preds = lgbmr_model.predict(X_test)

# %%
cross_validation = cross_validate(estimator=lgbmr_model,
                                  X=california.data,
                                  y=california.target,
                                  cv=5)

print('AVG fit time: {}'.format(cross_validation['fit_time'].mean()))
print('AVG score time: {}'.format(cross_validation['score_time'].mean()))
print('AVG test time: {}'.format(cross_validation['test_score'].mean()))

# %%
from sklearn.metrics import mean_squared_error

# 테스트 세트 예측
y_pred = lgbmr_model.predict(X_test)
# MSE 계산 및 출력
mse = mean_squared_error(y_test,y_pred)
print("MSE",mse)


# %%
plot_importance(lgbmr_model,figsize=(10,10))

# %%
plot_tree(lgbmr_model,figsize=(20,18))


