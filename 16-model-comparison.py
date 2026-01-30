import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Assume we have some arbitrary df with features and target
X_train, X_test = ...  # features
y_train, y_test = ...  # target

# ============================================================================
# 1. LINEAR REGRESSION (for continuous target)
# ============================================================================
# Math: ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
#       Minimizes: Σ(y - ŷ)²  (ordinary least squares)
# Pros: Interpretable
# Cons: Assumes linearity
# Use if data looks like a line

model = LinearRegression(
    fit_intercept=True,    # ← Whether to calculate intercept β₀ (almost always True)
    n_jobs=-1              # ← Use all CPU cores for parallelization
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ============================================================================
# 2. LOGISTIC REGRESSION (for binary/multiclass target)
# ============================================================================
# Math: P(y=1|x) = 1/(1 + e^(-z)) where z = β₀ + β₁x₁ + ... + βₙxₙ  (sigmoid)
#       Minimizes: -Σ[y·log(ŷ) + (1-y)·log(1-ŷ)]  (cross-entropy loss)
# Use if: data looks like points are separable by a line/plane boundary
#
# HYPERPARAMETERS TO TUNE:
#   penalty=['l1', 'l2'] 
#   C=[0.01, 0.1, 1, 10, 100]
#   solver=[,'liblinear','saga'] (also has 'lbfgs' and 'newton-cg')

model = LogisticRegression(
    penalty='l2',            # ← regularization type (l1 = lasso, l2 = ridge)
    C=1.0,                   # ← ↓C = ↑regularization
    solver='lbfgs',          # ← Optimization algorithm. Tune over solver=['lbfgs','liblinear','saga','newton-cg']
    max_iter=1000,           # ← Maximum iterations for convergence
    class_weight='balanced', # ← Auto-adjust weights for imbalanced classes
    multi_class='auto',      # ← 'ovr' (one-vs-rest) = separate logistic for each class, prediction is one with highest prob
    random_state=42          #   'multiclass' = output probability vector of being in each class
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# ============================================================================
# 3. XGBoost (Extreme Gradient Boosting)
# ============================================================================
# Intuition: Build trees sequentially, each correcting errors of previous trees
# Pros: Prevents overfitting
# Cons: Many hyperparameters to tune, slower to train
# Use when: Good on all tabular data, especially w mixed feature dtypes
#
# HYPERPARAMETERS TO TUNE:
#   n_estimators=[100, 200, 500]
#   learning_rate=[0.01, 0.05, 0.1, 0.2]
#   max_depth=[3, 5, 7, 10]
#   min_child_weight=[1, 3, 5]
#   subsample=[0.6, 0.8, 1.0]
#   colsample_bytree=[0.6, 0.8, 1.0]
#   gamma=[0, 0.1, 0.5, 1]
#   reg_alpha=[0, 0.1, 1]
#   reg_lambda=[1, 5, 10]

model = xgb.XGBClassifier(   # or XGBRegressor for regression
    n_estimators=100,        # ← Number of boosting rounds (trees to build)
    learning_rate=0.1,       # ← η: shrinkage applied to each tree (smaller = more robust but needs more trees)
    max_depth=6,             # ← Maximum depth of each tree (deeper = more complex, higher risk of overfit)
    min_child_weight=1,      # ← Minimum sum of instance weights needed in a child (higher = more conservative)
    subsample=0.8,           # ← Fraction of samples to use for each tree (prevents overfitting)
    colsample_bytree=0.8,    # ← Fraction of features to use for each tree (prevents overfitting)
    gamma=0,                 # ← γ: minimum loss reduction required to make split (higher = more conservative)
    reg_alpha=0,             # ← L1 regularization on weights (Lasso)
    reg_lambda=1,            # ← λ: L2 regularization on weights (Ridge)
    scale_pos_weight=1,      # ← Balance of positive/negative weights (for imbalanced data)
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# ============================================================================
# 4. LightGBM (Light Gradient Boosting Machine)
# ============================================================================
# Intuition: XGBoost, but faster because 1. leaf-wise growth 2. histogram-based splits
# Use when: Tabular data but speed is critical
#
# HYPERPARAMETERS TO TUNE:
#   n_estimators=[100, 200, 500]
#   learning_rate=[0.01, 0.05, 0.1, 0.2]
#   max_depth=[-1, 5, 10, 20]
#   num_leaves=[15, 31, 63, 127]
#   min_child_samples=[5, 10, 20, 50]
#   subsample=[0.6, 0.8, 1.0]
#   colsample_bytree=[0.6, 0.8, 1.0]
#   reg_alpha=[0, 0.1, 1]
#   reg_lambda=[0, 1, 5, 10]

model = lgb.LGBMClassifier(     # ← or LGBMRegressor for regression
    n_estimators=100,           # ← Number of boosting rounds
    learning_rate=0.1,          # ← Shrinkage rate (smaller = more robust)
    max_depth=-1,               # ← Maximum tree depth (-1 = no limit, uses num_leaves instead)
    num_leaves=31,              # ← Max number of leaves in one tree (2^max_depth for balanced tree)
    min_child_samples=20,       # ← Minimum number of samples needed in a leaf (higher = more conservative)
    subsample=0.8,              # ← Fraction of samples for each tree (also called bagging_fraction)
    colsample_bytree=0.8,       # ← Fraction of features for each tree (also called feature_fraction)
    reg_alpha=0,                # ← L1 regularization
    reg_lambda=0,               # ← L2 regularization
    class_weight='balanced',    # ← Handle imbalanced classes
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# ============================================================================
# 5. CatBoost (Categorical Boosting)
# ============================================================================
# Intuition: XGBoost, but better because
#  1. natively handles categorical encoding less prone to overfitting leakage
#     called "ordered boosting" AKA replace category with target mean of only rows above it
#  2. natively handles missing values 
#  3. inference is faster because all nodes on same level are split on same feature & threshold
#     called "symmetric trees". note: actually more expensive to train/fit
# Use when: Many categorical features
# 
# HYPERPARAMETERS TO TUNE:
#   iterations=[100, 200, 500]
#   learning_rate=[0.01, 0.05, 0.1, 0.2]
#   depth=[4, 6, 8, 10]
#   l2_leaf_reg=[1, 3, 5, 10]
#   bagging_temperature=[0, 0.5, 1, 2]
#   border_count=[32, 64, 128, 254]

model = cb.CatBoostClassifier(      # or CatBoostRegressor for regression
    iterations=100,                 # ← Number of boosting rounds (same as n_estimators)
    learning_rate=0.1,              # ← Shrinkage rate
    depth=6,                        # ← Maximum tree depth
    l2_leaf_reg=3,                  # ← L2 regularization coefficient (higher = more regularization)
    border_count=254,               # ← Number of splits for numerical features (higher = finer splits)
    bagging_temperature=1,          # ← Bayesian bootstrap intensity (0 = no bootstrap, higher = more aggressive)
    random_strength=1,              # ← Amount of randomness for scoring splits (higher = more random)
    auto_class_weights='Balanced',  # ← Automatically balance class weights: 'Balanced', 'SqrtBalanced', or None
    cat_features=None,              # ← Indices or names of categorical features (None = auto-detect)
    verbose=False,                  # ← Suppress training output
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)


## Support vector classification/regression (SVC/SVR) are not included here
## because they rarely outperform the models above on tabular data