import pandas as pd
from sklearn import linear_model
from rory.prepare import impute_dummify_and_split


def main():
    housing = pd.read_csv("Data/train_original.csv")

    training_features, testing_features, training_target, testing_target = impute_dummify_and_split(
        housing
    )

    # RIDGE

    model = linear_model.Ridge()

    model.fit(training_features, training_target)

    ridge_score = model.score(testing_features, testing_target)
    # score = 0.880

    # LASSO

    model = linear_model.Lasso(alpha=20)

    model.fit(training_features, training_target)

    lasso_score = model.score(testing_features, testing_target)
    # score = 0.8717402605376614

    return [
        ("_1_basic", "ridge", "", ridge_score),
        ("_1_basic", "lasso", "", lasso_score),
    ]
