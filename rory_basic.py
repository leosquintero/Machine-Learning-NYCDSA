import pandas as pd
import numpy as np
from sklearn import linear_model
from fancyimpute import KNN
import math
from sklearn.model_selection import train_test_split


def split_impute_and_dummify(
    df, target_variable="SalePrice", drop_target=True
):
    df = df[df[target_variable].notnull()]

    if drop_target:
        X_all = df.drop(target_variable, axis="columns")
    else:
        X_all = df

    y = df[target_variable]

    non_numerical = X_all.select_dtypes(["object"])
    dummies = pd.get_dummies(non_numerical, drop_first=True)

    numerical_X = X_all.select_dtypes(["number"])

    X_numerical_filled_knn = pd.DataFrame(
        data=KNN(k=math.floor(df.shape[0] ** 0.5)).fit_transform(numerical_X),
        columns=numerical_X.columns,
    )

    X = pd.merge(
        X_numerical_filled_knn, dummies, left_index=True, right_index=True
    )

    return train_test_split(X, y, random_state=42)


housing = pd.read_csv("Data/train_original.csv")

training_features, testing_features, training_target, testing_target = split_impute_and_dummify(
    housing, drop_target=False
)

corrs = np.abs(training_features.corr()["SalePrice"]).sort_values(
    ascending=False
)

RELEVANT_COL_COUNT = 100

columns = corrs[:RELEVANT_COL_COUNT].keys()

training_features = training_features[columns].drop(
    "SalePrice", axis="columns"
)

testing_features = testing_features[columns].drop("SalePrice", axis="columns")

model = linear_model.Ridge()

model.fit(training_features, training_target)

model.score(testing_features, testing_target)
