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
    print(col)
    print(df[col].value_counts().head())
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




