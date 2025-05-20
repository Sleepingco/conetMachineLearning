# %%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train1 = pd.read_csv('/kaggle/input/titanic/train.csv')
test1 = pd.read_csv('/kaggle/input/titanic/test.csv')

train = train1
test = test1

# %%
train.head()

# %%
test.head()

# %%
train.shape

# %%
train.info()

# %%
test.info()

# %%
train1.isnull().sum()

# %%
test1.isnull().sum()

# %%
feature_selected = 'Sex'
feature_selected = 'Pclass'
feature_selected = 'SibSp'
feature_selected = 'Parch'
feature_selected = 'Embarked'

F_survived = train[train['Survived']==1][feature_selected].value_counts()
F_dead = train[train['Survived']==0][feature_selected].value_counts()
F_df=pd.DataFrame([F_survived,F_dead])
F_df.index = ['Survived','Dead']
F_df.plot(kind='bar',stacked=True,figsize=(10,5))

# %%
train1.head()

# %%
TDF = pd.DataFrame(train1)

TDF1 = TDF.iloc[:,[0,1,2,4,5,6]]
TDF1.head()

# %%
TDF2 = TDF1.pivot_table('Survived', index='Sex',
                       columns='Pclass',aggfunc='mean')
TDF2

# %%
TDF2.plot(kind='bar',stacked=True,title='average survivors rates')

# %%
train = train1
train_test_data = [train,test]
for two_data in train_test_data:
    two_data['Title'] = two_data["Name"].str.extract(' ([A-Za-z]+)\.',expand =False)
    

# %%
train['Title'].value_counts()

# %%
test['Title'].value_counts()

# %%
train.Title.value_counts()

# %%
test.Title.value_counts()

# %%
pd.crosstab(train['Title'],train['Sex'])

# %%
pd.crosstab(test['Title'],test['Sex'])

# %%
train.groupby('Title')['Survived'].apply(lambda x:x.mean())

# %%
title_mapping = {'Mr':0,'Rev':0,'Don':0,'Capt':0,'Jonkheer':0,
                'Miss':1,'Ms':1,
                'Mrs':2,'Lady':2,'Dona':2,'Mme':2,'Countess':2,
                 'Master':3,'Dr':3,'Mlle':3,
                'Col':4,'Major':4,"Sir":4}
for two_data in train_test_data:
    two_data['Title'] = two_data['Title'].map(title_mapping)

# %%
train.head()

# %%
test.head()

# %%
train.drop("Name",axis = 1, inplace = True)
test.drop("Name",axis=1, inplace = True)

# %%
train.head()

# %%
test.head()

# %%
feature_selected = 'Title'

F_survived = train[train['Survived']==1][feature_selected].value_counts().sort_index()
F_dead = train[train['Survived']==0][feature_selected].value_counts().sort_index()
F_df = pd.DataFrame([F_survived, F_dead])
F_df.index = ['Survived','Dead']
F_df.plot(kind='bar',stacked=True,figsize=(10,5))

# %%
sex_mapping = {'male':0,'female':1}
for two_data in train_test_data:
    two_data['Sex']=two_data['Sex'].map(sex_mapping)

# %%
test.head()

# %%
train.info()

# %%
train['Age'].fillna(train.groupby('Title')['Age'].transform('mean'),inplace=True)
test['Age'].fillna(test.groupby('Title')['Age'].transform('mean'),inplace=True)

# %%
train.info()

# %%
train['Age'].describe()

# %%
train['Age'].value_counts()

# %%
facet = sns.FacetGrid(train,
                     hue='Survived',aspect=4)
facet.map(sns.kdeplot,'Age',shade=True)
facet.set(xlim = (0, train['Age'].max()))
facet.add_legend()

# %%
import matplotlib.pyplot as plt

facet = sns.FacetGrid(train,hue = 'Survived', aspect=4)
facet.map(sns.kdeplot, 'Age', shade = True)
facet.set(xlim=(0,train['Age'].max()))
facet.add_legend()

plt.xlim(0,18)

plt.show()

# %%
facet = sns.FacetGrid(train,hue='Survived',aspect=4)
facet.map(sns.kdeplot,'Age',shade=True)
facet.set(xlim=(0,train['Age'].max()))
facet.add_legend()

plt.xlim(18,35)
plt.show()

# %%
facet = sns.FacetGrid(train,hue='Survived',aspect=4)
facet.map(sns.kdeplot,'Age',shade=True)
facet.set(xlim=(0,train['Age'].max()))
facet.add_legend()

plt.xlim(35,45)
plt.show()

# %%
facet = sns.FacetGrid(train,hue='Survived',aspect=4)
facet.map(sns.kdeplot,'Age',shade=True)
facet.set(xlim=(0,train['Age'].max()))
facet.add_legend()

plt.xlim(45,60)
plt.show()

# %%
for two_data in train_test_data:
    two_data.loc[two_data['Age']<=18,'Age']=0
    two_data.loc[(two_data['Age']>18)&(two_data['Age']<=35),'Age']=1
    two_data.loc[(two_data['Age']>35)&(two_data['Age']<=45),'Age']=2
    two_data.loc[(two_data['Age']>45)&(two_data['Age']<=60),'Age']=3
    two_data.loc[(two_data['Age']>60),'Age']=4

# %%
train.head()

# %%
train.info()

# %%
feature_selected = 'Embarked'

F_survived = train[train['Survived']==1][feature_selected].value_counts()
F_dead = train[train['Survived']==0][feature_selected].value_counts()
F_df = pd.DataFrame([F_survived,F_dead])
F_df.index = ['Survived','Dead']
F_df.plot(kind='bar',stacked=True,figsize=(10,5))

# %%
Pclass1 = train[train['Pclass']==1]['Embarked'].value_counts()
Pclass2 = train[train['Pclass']==2]['Embarked'].value_counts()
Pclass3 = train[train['Pclass']==3]['Embarked'].value_counts()

df = pd.DataFrame([Pclass1,Pclass2,Pclass3])
df.index = ['1st class','2nd class', '3rd class']
df.plot(kind='bar',stacked=True,figsize=(10,5))

# %%
df

# %%
for two_data in train_test_data:
    two_data['Embarked'] = two_data['Embarked'].fillna('S')

# %%
embarked_mapping = {'S':0,'C':1,'Q':2}

for two_data in train_test_data:
    two_data['Embarked'] = two_data['Embarked'].map(embarked_mapping)

# %%
train.head()

# %%
train['Fare'].fillna(train.groupby('Pclass')['Fare'].transform('median'),inplace = True)
test['Fare'].fillna(test.groupby('Pclass')['Fare'].transform('median'),inplace = True)

test.info()

# %%
train['Fare'].max()

# %%
facet = sns.FacetGrid(train,hue='Survived',aspect=4)
facet.map(sns.kdeplot,'Fare',shade=True)
facet.set(xlim=(0,train['Fare'].max()))
facet.add_legend()

plt.xlim(0,300)
plt.show()

# %%
facet = sns.FacetGrid(train,hue='Survived',aspect=4)
facet.map(sns.kdeplot,'Fare',shade=True)
facet.set(xlim=(0,train['Fare'].max()))
facet.add_legend()

plt.xlim(0,30)
plt.show()

# %%


# %%
for two_data in train_test_data:
    two_data.loc[two_data['Fare']<=5,'Fare']=0
    two_data.loc[(two_data['Fare']>5)&(two_data['Fare']<=15),'Fare']=1
    two_data.loc[(two_data['Fare']>15)&(two_data['Fare']<=30),'Fare']=2
    two_data.loc[(two_data['Fare']>30)&(two_data['Fare']<=100),'Fare']=3
    two_data.loc[(two_data['Fare']>100),'Fare']=4
train.head()

# %%
train.info()

# %%
train['Cabin'].value_counts()

# %%
for two_data in train_test_data:
    two_data['Cabin'] = two_data['Cabin'].str[:1]
train['Cabin'].value_counts()

# %%
train[train['Pclass']==1]['Cabin'].value_counts()

# %%
train[train['Pclass']==2]['Cabin'].value_counts()

# %%
train[train['Pclass']==3]['Cabin'].value_counts()

# %%
Pclass1 = train[train['Pclass']==1]['Cabin'].value_counts()
Pclass2 = train[train['Pclass']==2]['Cabin'].value_counts()
Pclass3 = train[train['Pclass']==3]['Cabin'].value_counts()

df = pd.DataFrame([Pclass1,Pclass2,Pclass3])
df.index = ['1st class','2nd class', '3rd class']
df.plot(kind='bar',stacked=True,figsize=(10,5))

# %%
cabin_mapping = {'A':2,'B':2,'C':2,'T':2,
                'D':1,'G':1,
                'E':0,'F':0}
for two_data in train_test_data:
    two_data['Cabin'] = two_data['Cabin'].map(cabin_mapping)

# %%
train.head()

# %%
train['Cabin'].fillna(train.groupby('Pclass')['Cabin'].transform('median'),inplace = True)
test['Cabin'].fillna(test.groupby('Pclass')['Cabin'].transform('median'),inplace = True)

# %%
train.head()

# %%
train.info()

# %%
train['Familysize'] = train['SibSp'] + train['Parch']+1
test['Familysize'] = test['SibSp'] + test['Parch']+1

# %%
train.head()

# %%
facet = sns.FacetGrid(train,hue='Survived',aspect=4)
facet.map(sns.kdeplot,'Familysize',shade=True)
facet.set(xlim=(0,train['Familysize'].max()))
facet.add_legend()

plt.xlim(0)
plt.show()

# %%
train['Familysize'].value_counts()

# %%
for two_data in train_test_data:
    two_data.loc[two_data['Familysize']<=1,'Familysize']=0
    two_data.loc[(two_data['Familysize']>1)&(two_data['Familysize']<=2),'Familysize']=1
    two_data.loc[(two_data['Familysize']>2)&(two_data['Familysize']<=5),'Familysize']=2
    two_data.loc[two_data['Familysize']>5,'Familysize']=3


# %%
train.head()

# %%
train['Familysize'].value_counts()

# %%
# 제거할 불필요한 열 지정
feature_drop = ['Ticket', 'SibSp', 'Parch']

# 학습 데이터에서 지정한 열 제거
train = train.drop(feature_drop, axis=1)
train = train.drop('PassengerId', axis=1)

# 테스트 데이터에서도 동일한 열 제거
test = test.drop(feature_drop, axis=1)

# 입력(features)과 정답(labels) 분리
train_x = train.drop('Survived', axis=1)
train_y = train['Survived']




# %%
# 상위 5개 행 출력
train_x.head()

# %%
train_y.head()

# %%
test.head()

# %%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

k_fold =KFold(n_splits = 10, shuffle=True,random_state=0)

# %%
k_clf = KNeighborsClassifier(n_neighbors = 10)
scoring1 = 'accuracy'
score = cross_val_score(k_clf,train_x,train_y,cv=k_fold,scoring=scoring1)
print(round(np.mean(score)*100,2))

# %%
d_clf = DecisionTreeClassifier()
scoring1 = 'accuracy'
score = cross_val_score(d_clf,train_x,train_y,cv=k_fold,scoring=scoring1)
print(round(np.mean(score)*100,2))

# %%
r_clf = RandomForestClassifier(n_estimators = 15)
scoring1 = 'accuracy'
score = cross_val_score(r_clf,train_x,train_y,cv=k_fold,scoring=scoring1)
print(round(np.mean(score)*100,2))

# %%
g_clf = GaussianNB()
scoring1 = 'accuracy'
score = cross_val_score(g_clf,train_x,train_y,cv=k_fold,scoring=scoring1)
print(round(np.mean(score)*100,2))

# %%
s_clf = SVC()
scoring1 = 'accuracy'
score = cross_val_score(s_clf,train_x,train_y,cv=k_fold,scoring=scoring1)
print(round(np.mean(score)*100,2))

# %%
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(random_state=42)
score = cross_val_score(lr_model,train_x,train_y,cv=k_fold,scoring=scoring1)
print(round(np.mean(score)*100,2))

# %%
s_clf = SVC()
s_clf.fit(train_x,train_y)
test_x = test.drop('PassengerId',axis=1).copy()
prediction = s_clf.predict(test_x)

submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':prediction})
submission.to_csv('submission.csv',index=False)

# %%
test.head()

# %%
test_x.head()

# %%
prediction

# %%
submission.head()

# %%
from xgboost import XGBClassifier

x_clf = XGBClassifier(n_estimators = 10,random_state=123)
scoring1 = 'accuracy'
score = cross_val_score(x_clf,train_x,train_y,cv=k_fold,scoring=scoring1)
print(round(np.mean(score)*100,2))

# %%
from xgboost import XGBClassifier

x_model = XGBClassifier(n_estimators = 20,random_state=123)
x_model.fit(train_x,train_y)

pred = x_model.predict_proba(test_x)[:,1]
pred_label = np.where(pred>0.5,1,0)
submission = pd.DataFrame({'PassengerId':test['PassengerId'],
                          'Survived':pred_label})
submission.to_csv('submission_xgboost.csv',index=False)

# %%



