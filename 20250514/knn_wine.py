# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# 데이터 로드
wine = load_wine()
print(wine)

# %%
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = pd.Series(wine.target)

# 데이터셋 정보 출력
print(df.info())
df.head()

# %%
# 데이터셋 요약 통계량 출력
print(df.describe())

# %%
# 각 피처의 분포 확인
df.hist(bins=30,figsize=(15,10))
plt.show()

# %%
# 피처 간 상관 관계 시각화
sns.pairplot(df, hue='target')
plt.show()

# %%
X = wine.data
y = wine.target
X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# %%
# KNN 모델 학습 및 예측
knn = KNeighborsClassifier(n_neighbors=5, weights='uniform',algorithm='auto', leaf_size=30,p=2,metric='minkowski')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)


# %%
# 분류 평가
report = classification_report(y_test, y_pred)
print(report)

# %%
from sklearn.model_selection import GridSearchCV
import numpy as np

odds = np.arange(1, 100, 2)


param_grid = {
    'n_neighbors' : odds, # 이웃의 수
    'weights':['unform','distance'],
    'algorithm':['ball_tree','kd_tree','brute'] # 가중치 함수
}

# 그리드 서치 수행
grid_search = GridSearchCV(knn,param_grid,cv=5)
grid_search.fit(X_train,y_train)

best_params = grid_search.best_params_
print('Best Paramethers:',best_params)



# %%
# 최적 파라미터로 훈련된 모델 사용하여 예측
y_pred = grid_search.predict(X_test)

# 분류 결과 평가
report = classification_report(y_test,y_pred)
print("classification_report")
print(report)


