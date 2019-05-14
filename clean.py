import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from fancyimpute import KNN
from sklearn.neighbors import KNeighborsClassifier

X = np.array(
    [[0, 2.10, 1.45], [1, 1.18, 1.33], [0, 1.22, 1.27], [1, -0.21, -1.19]]
)


# X_with_nan = np.array([[np.nan, 0.87, 1.31], [np.nan, -0.67, -0.22]])

clf = KNeighborsClassifier(5, weights="distance")

# trained_model = clf.fit(X[:, 1:], X[:, 0])

imputed_values = trained_model.predict(X_with_nan[:, 1:])

# Join column of predicted class with their other features
X_with_imputed = np.hstack((imputed_values.reshape(-1, 1), X_with_nan[:, 1:]))

# Join two feature matrices
np.vstack((X_with_imputed, X))

mp_mean = SimpleImputer(missing_values=np.nan, strategy="knn")

rng = np.random.RandomState(0)

dataset = pd.read_csv("Data/train.csv")

y_full = dataset.SalePrice

X_full = dataset.drop("SalePrice", axis=1).select_dtypes(["number"])

n_samples = X_full.shape[0]
n_features = X_full.shape[1]

# Estimate the score on the entire dataset, with no missing values
estimator = RandomForestRegressor(random_state=0, n_estimators=100)
score = cross_val_score(estimator, X_full, y_full).mean()
print("Score with the entire dataset = %.2f" % score)

# Add missing values in 75% of the lines
missing_rate = 0.75
n_missing_samples = np.floor(n_samples * missing_rate)
missing_samples = np.hstack(
    (
        np.zeros(n_samples - n_missing_samples, dtype=np.bool),
        np.ones(n_missing_samples, dtype=np.bool),
    )
)

rng.shuffle(missing_samples)
missing_features = rng.randint(0, n_features, n_missing_samples)

# Estimate the score without the lines containing missing values
X_filtered = X_full[~missing_samples, :]
y_filtered = y_full[~missing_samples]

estimator = RandomForestRegressor(random_state=0, n_estimators=100)
score = cross_val_score(estimator, X_filtered, y_filtered).mean()

print("Score without the samples containing missing values = %.2f" % score)

# Estimate the score after imputation of the missing values
X_missing = X_full.copy()
X_missing[np.where(missing_samples)[0], missing_features] = 0
y_missing = y_full.copy()
estimator = Pipeline(
    [
        ("imputer", Imputer(missing_values=0, strategy="mean", axis=0)),
        ("forest", RandomForestRegressor(random_state=0, n_estimators=100)),
    ]
)
