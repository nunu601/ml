import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns

train_data=pd.read_csv('input/train.csv')
print(train_data.head())
test_data=pd.read_csv('input/test.csv')
print(test_data.head())
women=train_data.loc[train_data.Sex == 'female']['Survived']
rate_women=sum(women)/len(women)
print('% of women who survived:',rate_women)
men=train_data.loc[train_data.Sex == 'male']['Survived']
rate_men=sum(men)/len(men)
print('% of men who survived:',rate_men)


y=train_data['Survived']
feature = ['Pclass','Sex','SibSp','Parch']

fig, saxis = plt.subplots(2, 3,figsize=(16,12))
sns.barplot(x = 'Embarked', y = 'Survived', data=train_data, ax = saxis[0,0])
plt.show()

all_data_na = (train_data.isnull().sum() / len(train_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio':all_data_na})
print(missing_data.head(20))



train_data['Age'].fillna(train_data['Age'].median(),inplace=True)
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0],inplace=True)
test_data['Age'].fillna(train_data['Age'].median(),inplace=True)
test_data['Embarked'].fillna(train_data['Embarked'].mode()[0],inplace=True)

train_data['Fare'].fillna(train_data['Fare'].median(),inplace=True)
test_data['Fare'].fillna(train_data['Fare'].mode()[0],inplace=True)


test_passaengerId = test_data.PassengerId

train_data.drop("Cabin",axis=1,inplace=True)
test_data.drop("Cabin",axis=1,inplace=True)
train_data.drop("Name",axis=1,inplace=True)
test_data.drop("Name",axis=1,inplace=True)
train_data.drop("PassengerId",axis=1,inplace=True)
test_data.drop("PassengerId",axis=1,inplace=True)
train_data.drop("Ticket",axis=1,inplace=True)
test_data.drop("Ticket",axis=1,inplace=True)
train_data.drop("Survived",axis=1,inplace=True)

all_data_na = (train_data.isnull().sum() / len(train_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio':all_data_na})
print(missing_data.head(20))


X = pd.get_dummies(train_data)
X_test = pd.get_dummies(test_data)

print('Train columns with null values: \n', X_test.isnull().sum())
model = RandomForestClassifier(n_estimators=100,max_depth=5,random_state=1)
model.fit(X,y)
y_forest = model.predict(X)

accuracy = accuracy_score(y,y_forest)
print('#########random forest classifier accuracy##############')
print(accuracy)
print(sum(abs(y_forest-y)) / len(y_forest))

predictions = model.predict(X_test)


output = pd.DataFrame({'PassengerId':test_passaengerId,'Survived':predictions})
output.to_csv('my_submission.csv',index=False)


other_params = {'eta': 0.3, 'n_estimators': 500, 'gamma': 0, 'max_depth': 6, 'min_child_weight': 1,
                'colsample_bytree': 1, 'colsample_bylevel': 1, 'subsample': 1, 'reg_lambda': 1, 'reg_alpha': 0,
                'seed': 33}
cv_params = {'n_estimators': np.linspace(100, 1000, 10, dtype=int)}

x_train,x_val,y_train,y_val = train_test_split(X,y,test_size=0.3,random_state=0)
model = XGBClassifier(learning_rate=0.1,

                       n_estimators=1000,         # 树的个数--1000棵树建立xgboost
                       max_depth=6,               # 树的深度
                       min_child_weight = 1,      # 叶子节点最小权重
                       gamma=0.,                  # 惩罚项中叶子结点个数前的参数
                       subsample=0.8,             # 随机选择80%样本建立决策树
                       colsample_btree=0.8,       # 随机选择80%特征建立决策树
                       objective='reg:logistic', # 指定损失函数
                       scale_pos_weight=1,        # 解决样本个数不平衡的问题
                       random_state=27            # 随机数

                       )

model.fit(x_train,y_train, eval_set=[(x_val,y_val)], eval_metric='auc',early_stopping_rounds=19,verbose=True)
y_val_pred = model.predict(x_val)
accuracy = accuracy_score(y_val,y_val_pred)

print('#########xgboost  classifier accuracy##############')



print(accuracy* 100.0)
print(sum(abs(y_forest-y)) / len(y_forest))


model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5), algorithm="SAMME", n_estimators=200, learning_rate=0.8)

model.fit(x_train,y_train)
y_val_pred = model.predict(x_val)
accuracy = accuracy_score(y_val,y_val_pred)


print('#########adaboost  classifier accuracy##############')
print(accuracy* 100.0)
print(sum(abs(y_forest-y)) / len(y_forest))