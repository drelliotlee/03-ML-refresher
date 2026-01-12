import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression

# 1st pipeline for cleaning, preprocessing, etc.
df_clean = (
    df_raw
    .pipe(cleaning)
    .pipe(cast_and_freeze_categories)
    .pipe(cast_numerics)
    .pipe(enforce_schema)
)

# HARD BOUNDARY
# x also called inputs, factors, features, predictors
# y also called outputs, response, target, labels
X = df_clean.drop(columns="features")
y = df_clean["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)