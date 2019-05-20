import numpy as np
import pandas as pd
from sklearn import linear_model
from rory.prepare import impute_dummify_and_split


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

        ridge_score = model.score(testing_features, testing_target)

        # LASSO

        model = linear_model.Lasso(alpha=alpha)

        model.fit(training_features, training_target)

        lasso_score = model.score(testing_features, testing_target)

        result = result + [
            ("_1_basic", "ridge", "alpha: {:.3f}".format(alpha), ridge_score),
            ("_1_basic", "lasso", "alpha: {:.3f}".format(alpha), lasso_score),
        ]

    return sorted(result, key=lambda x: x[3], reverse=True)[:5]
