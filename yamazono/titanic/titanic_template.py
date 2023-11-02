import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import warnings
import pdb
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 150)

path = "/home/yamazono/DASeminar2023/yamazono/titanic/dataset/"
path_image = "/home/yamazono/DASeminar2023/yamazono/titanic/image/"

df = pd.read_csv(path + 'train.csv')
df_test = pd.read_csv(path + 'test.csv')


## -----------------------特徴量の可視化-------------------------------
# 生存率
f,ax=plt.subplots(1,2,figsize=(18,8), facecolor='gray')
df['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Survived')
ax[0].set_ylabel('')
sns.countplot(x='Survived',data=df,ax=ax[1])
ax[1].set_title('Survived')
plt.savefig(path_image + 'survived.png')

# カテゴリーごとの割合
f,ax=plt.subplots(1,2,figsize=(18,8), facecolor='gray')
df['Pclass'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'],ax=ax[0])
ax[0].set_title('Number of Passengers By Pclass')
ax[0].set_ylabel('Count')
sns.countplot(x='Pclass',hue='Survived',data=df,ax=ax[1])
ax[1].set_title('Pclass:Perished vs Survived')
plt.savefig(path_image + 'category.png')

# バイオリン図
f,ax=plt.subplots(1,2,figsize=(18,8), facecolor='gray')
sns.violinplot(x="Pclass",y="Age", hue="Survived", data=df,split=True,ax=ax[0])
ax[0].set_title('Pclass and Age vs Perished')
ax[0].set_yticks(range(0,110,10))
sns.violinplot(x="Sex",y="Age", hue="Survived", data=df,split=True,ax=ax[1])
ax[1].set_title('Sex and Age vs Perished')
ax[1].set_yticks(range(0,110,10))
plt.savefig(path_image + 'violin.png')

# ヒートマップ
sns.heatmap(df.corr(),annot=True,cmap='bwr',linewidths=0.2)
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.savefig(path_image + 'heatmap.png')

## ---------------------------------------------------------------


## ------------------データの前処理---------------------------------
# Ageの欠損値を中央値で補完
age_median = df['Age'].median()
df['Age'].fillna(age_median, inplace=True)
df_test['Age'].fillna(age_median, inplace=True)

# Cabinは多くの欠損値を含むため削除
df.drop('Cabin', axis=1, inplace=True)
df_test.drop('Cabin', axis=1, inplace=True)

# Embarkedの欠損値を最頻値で補完
embarked_mode = df['Embarked'].mode()[0]
df['Embarked'].fillna(embarked_mode, inplace=True)
df_test['Embarked'].fillna(embarked_mode, inplace=True)

# カテゴリ変数を数値にエンコーディング
label_encoder = LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])
df_test['Sex'] = label_encoder.transform(df_test['Sex'])

df['Embarked'] = label_encoder.fit_transform(df['Embarked'])
df_test['Embarked'] = label_encoder.transform(df_test['Embarked'])

# Fareカラムの平均値を計算
fare_mean = df_test['Fare'].mean()

# 欠損値を平均値で埋める
df_test['Fare'].fillna(fare_mean, inplace=True)

## -----------------------------------------------------------------


## ----------------データセットの作成---------------------------------

# 特徴量とターゲット変数を選択
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
target = 'Survived'

X = df[features]
y = df[target]
X_test = df_test[features]  # テストデータにはターゲット変数がない

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42)

## ------------------------------------------------------------------


##---------------- 様々なモデルの構築・調整----------------------------

## ランダムフォレスト
rfc = RandomForestClassifier(max_depth=10, min_samples_leaf=1, n_estimators=100, n_jobs=-1, random_state=42)
rfc.fit(X_train, y_train)
print('Random Forest')
print('Trian Score: {}'.format(round(rfc.score(X_train, y_train), 3)))
print(' Test Score: {}'.format(round(rfc.score(X_valid, y_valid), 3)))

### ロジスティック回帰モデル
lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)
print('Logistic Regression')
print('Train Score: {}'.format(round(lr.score(X_train, y_train), 3)))
print(' Test Score: {}'.format(round(lr.score(X_valid, y_valid), 3)))

### 多層パーセプトロンモデル
mlpc = MLPClassifier(hidden_layer_sizes=(100, 100, 10), random_state=0)
mlpc.fit(X_train, y_train)
print('Multilayer Perceptron')
print('Train Score: {}'.format(round(mlpc.score(X_train, y_train), 3)))
print(' Test Score: {}'.format(round(mlpc.score(X_valid, y_valid), 3)))

##--------------------------------------------------------------------

## モデルのアンサンブリング
rfc_pred = rfc.predict_proba(X_test)
lr_pred = lr.predict_proba(X_test)
mlpc_pred = mlpc.predict_proba(X_test)
pred_proba = (rfc_pred + lr_pred + mlpc_pred) / 3
pred = pred_proba.argmax(axis=1)

## 提出
submission = pd.read_csv(path + 'gender_submission.csv')
submission['Survived'] = pred
submission.to_csv(path + 'gender_submission.csv', index=False)
