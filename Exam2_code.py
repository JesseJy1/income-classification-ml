#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 09:06:59 2025

@author: shenyanxi
"""

import pandas as pd

df = pd.read_csv("salary.csv")
df.head()
print(df.shape)
print(df.dtypes)
print(df['>50k'].value_counts(dropna=False))


df[['age', 'years-of-education', 'hours-per-week']].describe()

for col in ['workclass', 'education', 'marital-status',
            'occupation', 'race', 'sex', 'native-country']:
    #print(col)
    print(df[col].value_counts().head().to_string())
    print()
    
df.isna().sum()
df = df.dropna()
df = df.drop(columns=['Unnamed: 0'])


df = df.dropna(subset=['>50k'])

df['target'] = (df['>50k'] == '>50k').astype(int)
df = df.drop(columns=['>50k'])

numeric_features = ['age', 'years-of-education', 'hours-per-week']
categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 
                        'race', 'sex', 'native-country']

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

preprocess = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numeric_features)])


from sklearn.model_selection import train_test_split

X = df[numeric_features + categorical_features]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    stratify = y, random_state =42)

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf = Pipeline(steps=[('preprocess', preprocess), ('model', rf_clf)])

clf.fit(X_train, y_train)


from sklearn.metrics import classification_report, confusion_matrix
y_pred = clf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, digits=3))


from sklearn.model_selection import RandomizedSearchCV
import numpy as np

param_distributions = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [None, 5, 10, 20],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4],
    'model__max_features': ['sqrt', 'log2'],
    'model__class_weight': [None, 'balanced']
}

search = RandomizedSearchCV(
    clf,
    param_distributions=param_distributions,
    n_iter=20,
    scoring='f1',           # 这里用F1作为调参目标
    cv=5,
    n_jobs=-1,
    random_state=42
)

search.fit(X_train, y_train)

print(search.best_params_)
print(search.best_score_)







