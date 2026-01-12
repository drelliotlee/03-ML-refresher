import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

np.random.seed(42)

data = {
    'height': np.random.normal(170,10,1000),
    'minute': np.random.uniform(0,60,1000),
    'income': np.random.lognormal(11.5,0.8,1000),
    'binary_outcome': np.random.randint(0,2,1000)
}

df = pd.DataFrame(data)

X = df[['height', 'minute', 'income']]
y = df['binary_outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# BEST PRACTICE: Use ColumnTransformer to apply different scalers to different columns
ct = ColumnTransformer([
    ('standard', StandardScaler(), ['height']),  # Normally distributed data
    ('minmax', MinMaxScaler(), ['minute']),  # Uniformly distributed data
    ('robust', RobustScaler(), ['income'])  # Data with outliers
])
X_train_mixed = ct.fit_transform(X_train)
X_test_mixed = ct.transform(X_test)

# MODELS THAT REQUIRE/BENEFIT FROM SCALING:
# logistic regression, SVM, KNN, neural networks, PCA, K-Means
lr_scaled = LogisticRegression(max_iter=1000)
lr_scaled.fit(X_train_mixed, y_train)

# MODELS THAT DON'T NEED SCALING:
# Tree-based models (Decision Trees, Random Forest, XGBoost, LightGBM)
rf_scaled = RandomForestClassifier(random_state=42)
rf_scaled.fit(X_train_mixed, y_train)


