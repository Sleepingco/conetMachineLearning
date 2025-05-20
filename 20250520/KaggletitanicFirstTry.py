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

train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')

train_x = train.drop(['Survived'],axis=1)
train_y = train['Survived']

test_x = test.copy()
# test_x

# %%
train.info()

# %%
train_x.info()

# %%
train_y

# %%
from sklearn.preprocessing import LabelEncoder

train_x = train_x.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)
test_x = test_x.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)

for c in ['Sex','Embarked']:
    le = LabelEncoder()
    le.fit(train_x[c].fillna('NA'))

    train_x[c] = le.transform(train_x[c].fillna('NA'))
    test_x[c] = le.transform(test_x[c].fillna('NA'))

# %%
train_x.head()

# %%
test_x.head()

# %%
from xgboost import XGBClassifier

model = XGBClassifier(n_estimators = 20, random_state=60)
model.fit(train_x,train_y)

pred = model.predict_proba(test_x)[:,1]

pred_label = np.where(pred>0.5,1,0)

submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':pred_label})
submission.to_csv('submission_second.csv',index=False)

# %%
pred

# %%
pred_label

# %%
submission


