import numpy as np
import pandas as pd
from sklearn import linear_model
from rory.prepare import impute_dummify_and_split


housing = pd.read_csv("Data/train_original.csv")
housing.isna().sum()[housing.isna().sum() > 0]
housing.shape


def main():
    """ runs basic ridge and lasso regressions with varying alphas """

    housing = pd.read_csv("Data/train_original.csv")

    training_features, testing_features, training_target, testing_target = impute_dummify_and_split(
        housing
    )

    result = []

    # 2.1 alpha previously found to give best score
    for alpha in np.append(10 ** np.linspace(-1, 2, num=10), [2.1]):

        # RIDGE
        model = linear_model.Ridge(alpha=alpha)

        model.fit(training_features, training_target)

        ridge_score_train = model.score(training_features, training_target)
        ridge_score_test = model.score(testing_features, testing_target)

        # LASSO

        model = linear_model.Lasso(alpha=alpha)

        model.fit(training_features, training_target)

        lasso_score_train = model.score(training_features, training_target)
        lasso_score_test = model.score(testing_features, testing_target)

        result = result + [
            (
                "_1_basic",
                "Ridge",
                "alpha: {:.3f}".format(alpha),
                ridge_score_train,
                ridge_score_test,
            ),
            (
                "_1_basic",
                "Lasso",
                "alpha: {:.3f}".format(alpha),
                lasso_score_train,
                lasso_score_test,
            ),
        ]

    return sorted(result, key=lambda x: x[3], reverse=True)[:5]
