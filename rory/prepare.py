import pandas as pd
import math
from sklearn.model_selection import train_test_split
from fancyimpute import KNN
from sklearn.preprocessing import Imputer


imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)

imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])


def impute_dummify_and_split(
    df, target_variable="SalePrice", drop_target=True, drop_first=True
):
    if "Id" in df:
        df = df.drop("Id", axis="columns")

    df = df[df[target_variable].notnull()]

    # uncomment next line to dummify MSSubClass
    # df["MSSubClass"] = df.MSSubClass.astype(str)

    if drop_target:
        X_all = df.drop(target_variable, axis="columns")
    else:
        X_all = df

    y = df[target_variable]

    non_numerical = X_all.select_dtypes(["object"])

    dummies = pd.get_dummies(non_numerical, drop_first=drop_first)

    numerical_X = X_all.select_dtypes(["number"])

    X_numerical_filled_knn = pd.DataFrame(
        data=KNN(k=100, verbose=False).fit_transform(numerical_X),
        columns=numerical_X.columns,
    )

    X = pd.merge(
        X_numerical_filled_knn, dummies, left_index=True, right_index=True
    )

    return train_test_split(X, y, random_state=42)
