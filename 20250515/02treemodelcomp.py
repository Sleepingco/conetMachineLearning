# %%
from sklearn.datasets import load_iris,load_wine,load_breast_cancer,load_diabetes
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate

# %%
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# %%
from sklearn.ensemble import BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

# %%
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
from sklearn.model_selection import cross_validate, StratifiedKFold


# %%
iris = load_iris()
wine = load_wine()
cancer = load_breast_cancer()
cv_class = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# %%
KB_model = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier()
)

bagging_model = BaggingClassifier(KB_model, n_estimators=10, random_state=42)

# %% [markdown]
# ## KNN Bagging with IRIS

# %%
cross_val = cross_validate(
    estimator = KB_model,
    X=iris.data, y = iris.target,
    cv=cv
)

print('Average Fot Time : Time {}'.format(cross_val['fit_time'].mean()))
print('Average Score Time : {}'.format(cross_val['score_time'].mean()))
print('Average Test Score : {}'.format(cross_val['test_score'].mean()))

# %%
cross_val = cross_validate(
    estimator = bagging_model,
    X=iris.data, y=iris.target,
    # cv=cv
    cv=5
)
print('Average Fot Time : Time {}'.format(cross_val['fit_time'].mean()))
print('Average Score Time : {}'.format(cross_val['score_time'].mean()))
print('Average Test Score : {}'.format(cross_val['test_score'].mean()))

# %% [markdown]
# 붓꽃 데이터에 KNN 모델과 KNN을 bagging으로 사용한 모델 성능 비교(0.97>0.96)

# %%
cross_val = cross_validate(
    estimator= KB_model,
    X=wine.data, y = wine.target,
    cv =cv
)
print('Average Fot Time : Time {}'.format(cross_val['fit_time'].mean()))
print('Average Score Time : {}'.format(cross_val['score_time'].mean()))
print('Average Test Score : {}'.format(cross_val['test_score'].mean()))

# %%
cross_val = cross_validate(
    estimator= bagging_model,
    X=wine.data, y = wine.target,
    cv =cv
)
print('Average Fot Time : Time {}'.format(cross_val['fit_time'].mean()))
print('Average Score Time : {}'.format(cross_val['score_time'].mean()))
print('Average Test Score : {}'.format(cross_val['test_score'].mean()))

# %% [markdown]
# 와인 데이터에 KNN 모델과 KNN bagging으로 사용한 모델 성능 비교 (0.96>0.95)

# %%
cross_val = cross_validate(
    estimator= KB_model,
    X = cancer.data,y=cancer.target,
    cv=cv
)
print('Average Fit Time : {}'.format(cross_val['fit_time'].mean()))
print('Average Score Time : {}'.format(cross_val['score_time'].mean()))
print('Average Test Score : {}'.format(cross_val['test_score'].mean()))

# %%
cross_val = cross_validate(
    estimator=bagging_model,
    X=cancer.data, y = cancer.target,
    cv=cv
)
print('Average Fit Time : {}'.format(cross_val['fit_time'].mean()))
print('Average Score Time : {}'.format(cross_val['score_time'].mean()))
print('Average Test Score : {}'.format(cross_val['test_score'].mean()))

# %% [markdown]
# Breast cancer 데이터에 KNN 모델과 KNN bagging으로 사용한 모델 성능 비교(0.97>0.96)
# 결론 bagging 모델이 보통 성적이 좋음

# %% [markdown]
# ## SVC 모델 구성 후 SVC bagging 모델 성능 비교

# %%
SB_model = make_pipeline(
    StandardScaler(),
    SVC()
)

Sbagging_model = BaggingClassifier(SB_model, n_estimators=10,random_state=42)
# ''',max_samples=0.5,max_features=0.5'''

# %%
cross_val = cross_validate(
    estimator= SB_model,
    X=iris.data, y=iris.target,
    cv=cv
)
print('Average Fit Time : {}'.format(cross_val['fit_time'].mean()))
print('Average Score Time : {}'.format(cross_val['score_time'].mean()))
print('Average Test Score : {}'.format(cross_val['test_score'].mean()))

# %%
cross_val = cross_validate(
    estimator=Sbagging_model,
    X=iris.data,y=iris.target,
    cv=cv
)
print('Average Fit Time : {}'.format(cross_val['fit_time'].mean()))
print('Average Score Time : {}'.format(cross_val['score_time'].mean()))
print('Average Test Score : {}'.format(cross_val['test_score'].mean()))

# %% [markdown]
# 붓꽃 데이터에 SVC 모델과 SVC bagging 모델 성능 비교
# 
# random state를 지정 안하니 svcBagging 모델이 계속 값이 바뀜 
# svc모델은 maxsample과 maxfeature을 적용유무에 관계없이 0.9666...으로 값이 고정
# 하지만 svcbagging 모델은 maxsample과 maxfeature을 적용시 0.94,0.93,0.92  등 값이 0.96에 비해 낮게 나옴
# 완전한 데이터를 사용하면 0.966..,0.98,0.97,0.96 등 svc모델보다 보통은 높게 나오나 간혹 0.95로 값이 낮게 나오는 경우 존재 차후 random stata 적용후 확인 필요

# %%
cross_val = cross_validate(
    estimator=SB_model,
    X=wine.data,y=wine.target,
    cv=cv
)
print('Average Fit Time : {}'.format(cross_val['fit_time'].mean()))
print('Average Score Time : {}'.format(cross_val['score_time'].mean()))
print('Average Test Score : {}'.format(cross_val['test_score'].mean()))

# %%
cross_val = cross_validate(
    estimator=Sbagging_model,
    X=wine.data,y=wine.target,
    cv=cv
)
print('Average Fit Time : {}'.format(cross_val['fit_time'].mean()))
print('Average Score Time : {}'.format(cross_val['score_time'].mean()))
print('Average Test Score : {}'.format(cross_val['test_score'].mean()))

# %% [markdown]
# 와인 데이터에 SVC 모델과 SVC bagging 모델 성능 비교

# %%
cross_val = cross_validate(
    estimator=SB_model,
    X=wine.data,y=wine.target,
    cv=cv
)
print('Average Fit Time : {}'.format(cross_val['fit_time'].mean()))
print('Average Score Time : {}'.format(cross_val['score_time'].mean()))
print('Average Test Score : {}'.format(cross_val['test_score'].mean()))

# %%
cross_val = cross_validate(
    estimator=Sbagging_model,
    X=cancer.data,y=cancer.target,
    cv=cv
)
print('Average Fit Time : {}'.format(cross_val['fit_time'].mean()))
print('Average Score Time : {}'.format(cross_val['score_time'].mean()))
print('Average Test Score : {}'.format(cross_val['test_score'].mean()))

# %% [markdown]
# Breast cancer 데이터에 SVC 모델과 SVC bagging 모델 성능 비교

# %%
DB_model = make_pipeline(
    StandardScaler(),
    DecisionTreeClassifier(random_state=42)
)

Dbagging_model = BaggingClassifier(
    estimator=DB_model,
    n_estimators=10,
    random_state=42
)



# %% [markdown]
# Decision Tree모델과 Decision Tree bagging 모델 성능 비교

# %%
cross_val = cross_validate(
    estimator=DB_model,
    X=iris.data, y=iris.target,
    cv=cv
)
print('Average Fit Time : {}'.format(cross_val['fit_time'].mean()))
print('Average Score Time : {}'.format(cross_val['score_time'].mean()))
print('Average Test Score : {}'.format(cross_val['test_score'].mean()))

# %%
cross_val = cross_validate(
    estimator=Dbagging_model,
    X=iris.data, y=iris.target,
    cv=cv
)
print('Average Fit Time : {}'.format(cross_val['fit_time'].mean()))
print('Average Score Time : {}'.format(cross_val['score_time'].mean()))
print('Average Test Score : {}'.format(cross_val['test_score'].mean()))

# %% [markdown]
# 붓꽃 데이터에 DT 모델과 DT bagging모델 성능 비교

# %%
cross_val = cross_validate(
    estimator=DB_model,
    X=wine.data, y=wine.target,
    cv=cv
)
print('Average Fit Time : {}'.format(cross_val['fit_time'].mean()))
print('Average Score Time : {}'.format(cross_val['score_time'].mean()))
print('Average Test Score : {}'.format(cross_val['test_score'].mean()))

# %%
cross_val = cross_validate(
    estimator=Dbagging_model,
    X=wine.data, y=wine.target,
    cv=cv
)
print('Average Fit Time : {}'.format(cross_val['fit_time'].mean()))
print('Average Score Time : {}'.format(cross_val['score_time'].mean()))
print('Average Test Score : {}'.format(cross_val['test_score'].mean()))

# %% [markdown]
# 와인 데이터에 DT 모델과 DT bagging모델 성능 비교

# %%
cross_val = cross_validate(
    estimator=DB_model,
    X=cancer.data, y=cancer.target,
    cv=cv
)
print('Average Fit Time : {}'.format(cross_val['fit_time'].mean()))
print('Average Score Time : {}'.format(cross_val['score_time'].mean()))
print('Average Test Score : {}'.format(cross_val['test_score'].mean()))

# %%
cross_val = cross_validate(
    estimator=Dbagging_model,
    X=cancer.data, y=cancer.target,
    cv=cv
)
print('Average Fit Time : {}'.format(cross_val['fit_time'].mean()))
print('Average Score Time : {}'.format(cross_val['score_time'].mean()))
print('Average Test Score : {}'.format(cross_val['test_score'].mean()))

# %% [markdown]
# Breast cancer 데이터에 DT 모델과 DT bagging모델 성능 비교
# 
# 

# %% [markdown]
# 메모 :146 p == 시트 ex sheet
# classifier 평가

# %%
from sklearn.model_selection import KFold

california = fetch_california_housing()
diabetes = load_diabetes()
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# %%
RB_model = make_pipeline(
    StandardScaler(),
    KNeighborsRegressor()
)

# BaggingRegressor 적용 (랜덤 시드 고정)
Rbagging_model = BaggingRegressor(
    estimator=RB_model,   # scikit-learn 1.2 이상은 estimator= 사용
    n_estimators=10,
    max_samples=0.3,
    max_features=0.5,
    random_state=42       # ✅ 결과 재현을 위한 시드 고정
)

# %%
cross_val = cross_validate(
    estimator=RB_model,
    X=california.data, y=california.target,
    cv=cv,
    # scoring='r2'
)
print('Average Fit Time : {}'.format(cross_val['fit_time'].mean()))
print('Average Score Time : {}'.format(cross_val['score_time'].mean()))
print('Average Test Score : {}'.format(cross_val['test_score'].mean()))

# %%
cross_val = cross_validate(
    estimator=Rbagging_model,
    X=california.data, y=california.target,
    cv=cv
)
print('Average Fit Time : {}'.format(cross_val['fit_time'].mean()))
print('Average Score Time : {}'.format(cross_val['score_time'].mean()))
print('Average Test Score : {}'.format(cross_val['test_score'].mean()))

# %% [markdown]
# 캘리포니아 집값 데이터에 K Regressor과 K Regressor bagging으로 사용한 모델 성능

# %%
cross_val = cross_validate(
    estimator=RB_model,
    X=diabetes.data, y=diabetes.target,
    cv=cv,
    # scoring='r2'
)
print('Average Fit Time : {}'.format(cross_val['fit_time'].mean()))
print('Average Score Time : {}'.format(cross_val['score_time'].mean()))
print('Average Test Score : {}'.format(cross_val['test_score'].mean()))

# %%
cross_val = cross_validate(
    estimator=Rbagging_model,
    X=diabetes.data, y=diabetes.target,
    cv=cv,
    # scoring='r2'
)
print('Average Fit Time : {}'.format(cross_val['fit_time'].mean()))
print('Average Score Time : {}'.format(cross_val['score_time'].mean()))
print('Average Test Score : {}'.format(cross_val['test_score'].mean()))

# %% [markdown]
# 당뇨병 환자 데이터에 K Regressor과 K Regressor bagging으로 사용한 모델 성능

# %%
SVRB_model = make_pipeline(
    StandardScaler(),
    SVR()
)

SVRbagging_model = BaggingRegressor(
    estimator=SVRB_model,   # scikit-learn 1.2 이상은 estimator= 사용
    n_estimators=10,
    max_samples=0.3,
    max_features=0.5,
    random_state=42       # ✅ 결과 재현을 위한 시드 고정
)

# %% [markdown]
# ## 캘리포니아 데이터에 SVR과 SVR bagging 모델 성능

# %%
cross_val = cross_validate(
    estimator=SVRB_model,
    X=california.data, y=california.target,
    cv=cv,
    # scoring='r2'
)
print('Average Fit Time : {}'.format(cross_val['fit_time'].mean()))
print('Average Score Time : {}'.format(cross_val['score_time'].mean()))
print('Average Test Score : {}'.format(cross_val['test_score'].mean()))

# %%
cross_val = cross_validate(
    estimator=SVRbagging_model,
    X=california.data, y=california.target,
    cv=cv,
    # scoring='r2'
)
print('Average Fit Time : {}'.format(cross_val['fit_time'].mean()))
print('Average Score Time : {}'.format(cross_val['score_time'].mean()))
print('Average Test Score : {}'.format(cross_val['test_score'].mean()))

# %% [markdown]
# 캘리포니아 데이터에 SVR과 SVR bagging 모델 성능
# 최적의 max sample, max feature를 확인하기 위해 다양하게 시도해 볼 필요가 있다 !!

# %%
cross_val = cross_validate(
    estimator=SVRB_model,
    X=diabetes.data, y=diabetes.target,
    cv=cv,
    # scoring='r2'
)
print('Average Fit Time : {}'.format(cross_val['fit_time'].mean()))
print('Average Score Time : {}'.format(cross_val['score_time'].mean()))
print('Average Test Score : {}'.format(cross_val['test_score'].mean()))

# %%
cross_val = cross_validate(
    estimator=SVRbagging_model,
    X=diabetes.data, y=diabetes.target,
    cv=cv,
    # scoring='r2'
)
print('Average Fit Time : {}'.format(cross_val['fit_time'].mean()))
print('Average Score Time : {}'.format(cross_val['score_time'].mean()))
print('Average Test Score : {}'.format(cross_val['test_score'].mean()))

# %% [markdown]
# 당뇨병 데이터에 KNN Regressor과 KNN Regressor bagging 모델 성능 비교

# %%
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

# %%
RF_model = make_pipeline(
    StandardScaler(),
    RandomForestClassifier()
)

# %%
cross_val = cross_validate(
    estimator=RF_model,
    X=iris.data,y=iris.target,
    cv = cv_class
)
print('Average Fit Time : {}'.format(cross_val['fit_time'].mean()))
print('Average Score Time : {}'.format(cross_val['score_time'].mean()))
print('Average Test Score : {}'.format(cross_val['test_score'].mean()))

# %% [markdown]
# 붓꽃 데이터에 Random Forest 모델(RF_model) 성능 평가

# %%
cross_val = cross_validate(
    estimator=RF_model,
    X=wine.data,y=wine.target,
    cv = cv_class
)
print('Average Fit Time : {}'.format(cross_val['fit_time'].mean()))
print('Average Score Time : {}'.format(cross_val['score_time'].mean()))
print('Average Test Score : {}'.format(cross_val['test_score'].mean()))

# %% [markdown]
# 와인 데이터에 Random Forest 모델(RF_model) 성능 평가

# %%
cross_val = cross_validate(
    estimator=RF_model,
    X=cancer.data,y=cancer.target,
    cv = cv_class
)
print('Average Fit Time : {}'.format(cross_val['fit_time'].mean()))
print('Average Score Time : {}'.format(cross_val['score_time'].mean()))
print('Average Test Score : {}'.format(cross_val['test_score'].mean()))

# %% [markdown]
# 암환자 데이터에 Random Forest 모델(RF_model) 성능 평가

# %%
RFR_model = make_pipeline(
    StandardScaler(),
    RandomForestRegressor()
)

# %%
cross_val = cross_validate(
    estimator=RFR_model,
    X=california.data,y=california.target,
    cv = cv
)
print('Average Fit Time : {}'.format(cross_val['fit_time'].mean()))
print('Average Score Time : {}'.format(cross_val['score_time'].mean()))
print('Average Test Score : {}'.format(cross_val['test_score'].mean()))

# %% [markdown]
# 캘리포니아 데이터에 Random Forest 모델(RFR_model) 성능 평가

# %%
cross_val = cross_validate(
    estimator=RFR_model,
    X=diabetes.data,y=diabetes.target,
    cv = cv
)
print('Average Fit Time : {}'.format(cross_val['fit_time'].mean()))
print('Average Score Time : {}'.format(cross_val['score_time'].mean()))
print('Average Test Score : {}'.format(cross_val['test_score'].mean()))

# %% [markdown]
# 당뇨병 데이터에 Random Forest 모델(RFR_model) 성능 평가

# %%
ETC_model = make_pipeline(
    StandardScaler(),
    ExtraTreesClassifier()
)

# %%
cross_val = cross_validate(
    estimator=ETC_model,
    X=iris.data,y=iris.target,
    cv = cv_class
)
print('Average Fit Time : {}'.format(cross_val['fit_time'].mean()))
print('Average Score Time : {}'.format(cross_val['score_time'].mean()))
print('Average Test Score : {}'.format(cross_val['test_score'].mean()))

# %% [markdown]
# 붓꽃 데이터에 ExtraTree Classifier 모델(ETC_model) 성능 평가

# %%
cross_val = cross_validate(
    estimator=ETC_model,
    X=wine.data,y=wine.target,
    cv = cv_class
)
print('Average Fit Time : {}'.format(cross_val['fit_time'].mean()))
print('Average Score Time : {}'.format(cross_val['score_time'].mean()))
print('Average Test Score : {}'.format(cross_val['test_score'].mean()))

# %% [markdown]
# 와인 데이터에 ExtraTree Classifier 모델(ETC_model) 성능 평가

# %%
cross_val = cross_validate(
    estimator=ETC_model,
    X=cancer.data,y=cancer.target,
    cv = cv_class
)
print('Average Fit Time : {}'.format(cross_val['fit_time'].mean()))
print('Average Score Time : {}'.format(cross_val['score_time'].mean()))
print('Average Test Score : {}'.format(cross_val['test_score'].mean()))

# %% [markdown]
# Breast cancer 데이터에 ExtraTree Classifier 모델(ETC_model) 성능 평가

# %%
ETR_model = make_pipeline(
    StandardScaler(),
    ExtraTreesRegressor(random_state=42)
)

# %%
from sklearn.model_selection import cross_validate,train_test_split

X_train,X_test,y_train,y_test = train_test_split(california.data,california.target,train_size=0.3)

# %%
cross_val = cross_validate(
    estimator=ETR_model,
    X=california.data,y=california.target,
    cv = cv
)
print('Average Fit Time : {}'.format(cross_val['fit_time'].mean()))
print('Average Score Time : {}'.format(cross_val['score_time'].mean()))
print('Average Test Score : {}'.format(cross_val['test_score'].mean()))

# %% [markdown]
# 캘리포니아 데이터에 ExtraTree Regressor 모델(ETR_model) 환경 설정

# %%
cross_val = cross_validate(
    estimator=ETR_model,
    X=diabetes.data,y=diabetes.target,
    cv = cv
)
print('Average Fit Time : {}'.format(cross_val['fit_time'].mean()))
print('Average Score Time : {}'.format(cross_val['score_time'].mean()))
print('Average Test Score : {}'.format(cross_val['test_score'].mean()))

# %% [markdown]
# 당뇨병 데이터에 ExtraTree Regressor 모델(ETR_model) 환경 설정s

# %%
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor


# %%
ADA_model = make_pipeline(
    StandardScaler(),
    AdaBoostClassifier(random_state=42)
)

# %% [markdown]
# ADA Boost Classifier 모델(ADA_model) 환경 설정

# %%
cross_val = cross_validate(
    estimator=ADA_model,
    X=iris.data,y=iris.target,
    cv = cv_class
)
print('Average Fit Time : {}'.format(cross_val['fit_time'].mean()))
print('Average Score Time : {}'.format(cross_val['score_time'].mean()))
print('Average Test Score : {}'.format(cross_val['test_score'].mean()))

# %% [markdown]
# 붓꽃 데이터에 ADA Boost Classifier 모델(ADA_model) 환경 설정

# %%
cross_val = cross_validate(
    estimator=ADA_model,
    X=wine.data,y=wine.target,
    cv = cv_class
)
print('Average Fit Time : {}'.format(cross_val['fit_time'].mean()))
print('Average Score Time : {}'.format(cross_val['score_time'].mean()))
print('Average Test Score : {}'.format(cross_val['test_score'].mean()))

# %% [markdown]
# 와인 데이터에 ADA Boost Classifier 모델(ADA_model) 환경 설정

# %%
cross_val = cross_validate(
    estimator=ADA_model,
    X=cancer.data,y=cancer.target,
    cv = cv_class
)
print('Average Fit Time : {}'.format(cross_val['fit_time'].mean()))
print('Average Score Time : {}'.format(cross_val['score_time'].mean()))
print('Average Test Score : {}'.format(cross_val['test_score'].mean()))

# %% [markdown]
# Breast cancer 데이터에 ADA Boost Classifier 모델(ADA_model) 환경 설정

# %%
ADAR_model = make_pipeline(
    StandardScaler(),
    AdaBoostRegressor()
)

# %%
from sklearn.model_selection import cross_validate,train_test_split
X_train,X_test,y_train,y_test = train_test_split(california.data,california.target,train_size=0.3)

# %%
cross_val = cross_validate(
    estimator=ADAR_model,
    X=X_train,y=y_train,
    cv = cv
)
print('Average Fit Time : {}'.format(cross_val['fit_time'].mean()))
print('Average Score Time : {}'.format(cross_val['score_time'].mean()))
print('Average Test Score : {}'.format(cross_val['test_score'].mean()))

# %% [markdown]
# 캘리포니아 데이터에 ADA Boost Regressor 모델(ADAR_model) 성능 평가

# %%
cross_val = cross_validate(
    estimator=ADAR_model,
    X=diabetes.data,y=diabetes.target,
    cv = cv
)
print('Average Fit Time : {}'.format(cross_val['fit_time'].mean()))
print('Average Score Time : {}'.format(cross_val['score_time'].mean()))
print('Average Test Score : {}'.format(cross_val['test_score'].mean()))

# %% [markdown]
# 당뇨병 데이터에 ADA Boost Regressor 모델(ADAR_model) 성능 평가

# %%
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor

# %%
GRAC_model = make_pipeline(
    StandardScaler(),
    GradientBoostingClassifier()
)

# %%
cross_val = cross_validate(
    estimator=GRAC_model,
    X=iris.data,y=iris.target,
    cv = cv
)
print('Average Fit Time : {}'.format(cross_val['fit_time'].mean()))
print('Average Score Time : {}'.format(cross_val['score_time'].mean()))
print('Average Test Score : {}'.format(cross_val['test_score'].mean()))

# %% [markdown]
# Gradient Boosting Classifier 모델(GRAC_model) 환경 구성 및 성능평가(붓꽃)

# %%
cross_val = cross_validate(
    estimator=GRAC_model,
    X=wine.data,y=wine.target,
    cv = cv
)
print('Average Fit Time : {}'.format(cross_val['fit_time'].mean()))
print('Average Score Time : {}'.format(cross_val['score_time'].mean()))
print('Average Test Score : {}'.format(cross_val['test_score'].mean()))

# %%
cross_val = cross_validate(
    estimator=GRAC_model,
    X=cancer.data,y=cancer.target,
    cv = cv
)
print('Average Fit Time : {}'.format(cross_val['fit_time'].mean()))
print('Average Score Time : {}'.format(cross_val['score_time'].mean()))
print('Average Test Score : {}'.format(cross_val['test_score'].mean()))

# %% [markdown]
# Gradient Boosting Classifier 모델(GRAC_model) 성능평가(와인, Breast cancer )

# %%
GRAR_model = make_pipeline(
    StandardScaler(),
    GradientBoostingRegressor(random_state=42)
)

# %% [markdown]
# Gradient Boosting Regressor 모델(GRAR_model) 환경 구성

# %%
cross_val = cross_validate(
    estimator=GRAR_model,
    X=X_train,y=y_train,
    cv = cv
)
print('Average Fit Time : {}'.format(cross_val['fit_time'].mean()))
print('Average Score Time : {}'.format(cross_val['score_time'].mean()))
print('Average Test Score : {}'.format(cross_val['test_score'].mean()))

# %% [markdown]
# 캘리포니아 데이터에 Gradient Boosting Regressor 모델(GRAR_model) 성능 평가

# %%
cross_val = cross_validate(
    estimator=GRAR_model,
    X=diabetes.data,y=diabetes.target,
    cv = cv
)
print('Average Fit Time : {}'.format(cross_val['fit_time'].mean()))
print('Average Score Time : {}'.format(cross_val['score_time'].mean()))
print('Average Test Score : {}'.format(cross_val['test_score'].mean()))

# %% [markdown]
# 당뇨병 데이터에 Gradient Boosting Regressor 모델(GRAR_model) 성능 평가

# %%
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score


# %% [markdown]
# Ensemble voting을 위한 환경 설정
# Hard voting : 각 모델의 예측 결과 중 가장 많은 표를 얻은 클래스를 최종 예측으로 선택
# Soft voting : 모델들의 예측 확률 값을 고려하여 최종 예측 결과를 도출

# %%
model_s = SVC()
model_d = DecisionTreeClassifier()
model_r = RandomForestClassifier()

voting_model = VotingClassifier(
    estimators=[('svc', model_s), ('DecisionTree', model_d), ('RF', model_r)],
    voting='hard'
)


# %%
for model in (model_s, model_d, model_r, voting_model):
    model_name = str(type(model)).split('.')[-1][:-2]
    scores = cross_val_score(model, iris.data, iris.target, cv=5)
    print('Accuracy : %0.2f [%s]' % (scores.mean(), model_name))


# %% [markdown]
# Hard voting 기준시 다수결로 모델 선정

# %%
type(SVC())  
# 모델의 클래스 타입을 문자열로 변환
# 출력: <class 'sklearn.svm._classes.SVC'>

str(type(SVC())).split('.')  
# '.(점)'을 기준으로 나눠 리스트로 반환
# 출력: ["<class 'sklearn'", 'svm', '_classes', "SVC'>"]

str(type(SVC())).split('.')[-1]  
# 리스트에서 마지막 요소 (클래스 이름 포함


# %%
model_s = SVC(probability=True)  # 가중치 예측 확률 계산 가능하도록 설정
model_d = DecisionTreeClassifier()
model_r = RandomForestClassifier()

voting_model = VotingClassifier(
    estimators=[('svc', model_s), ('DecisionTree', model_d), ('forest', model_r)],
    voting='soft',
    weights=[1, 1, 5]  # 평균 시 각 모델에 대한 가중치 부여
)


# %%
for model in (model_s, model_d, model_r, voting_model):
    model_name = str(type(model)).split('.')[-1][:-2]
    scores = cross_val_score(model, iris.data, iris.target, cv=5)
    print('Accuracy : %0.2f [%s]' % (scores.mean(), model_name))


# %% [markdown]
# Soft voting 기준시 모델별 가중치를 부여하여 다수결로 모델 선정


