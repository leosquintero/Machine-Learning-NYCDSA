import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from rory.prepare import impute_dummify_and_split


def main():
    """ runs basic ridge and lasso regressions with varying alphas """

    housing = pd.read_csv("Data/train_original.csv")

    housing["TotalSF"] = (
        housing["TotalBsmtSF"] + housing["X1stFlrSF"] + housing["X2ndFlrSF"],
    )

    housing.corr()

    training_features, testing_features, training_target, testing_target = impute_dummify_and_split(
        housing
    )

    result = []

    basic_by_alpha = pd.DataFrame(
        columns=["model", "alpha", "train_score", "test_score"]
    )

    # 2.1 alpha previously found to give best score
    for alpha in np.append(10 ** np.linspace(-1, 2, num=400), [2.1]):

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

        basic_by_alpha = basic_by_alpha.append(
            [
                {
                    "model": "Ridge",
                    "alpha": alpha,
                    "train_score": ridge_score_train,
                    "test_score": ridge_score_test,
                },
                {
                    "model": "Lasso",
                    "alpha": alpha,
                    "train_score": lasso_score_train,
                    "test_score": lasso_score_test,
                },
            ]
        )

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

    # return sorted(result, key=lambda x: x[3], reverse=True)[:5]
    return basic_by_alpha


# res = main()
# res = res.reset_index().drop("index", axis="columns")
#
# res = res.sort_values(by="test_score", ascending=False)
#
#
# def scatter_plot(df, x, y):
#     plt.figure(figsize=(16, 8))
#     plt.scatter(df[x], df[y], c="black")
#     plt.xlabel(x)
#     plt.ylabel(y)
#     plt.show()
#
#
# lasso_alphas = res[res.model == "Lasso"].drop("model", axis="columns")
# ridge_alphas = res[res.model == "Ridge"].drop("model", axis="columns")
#
#
# scatter_plot(lasso_alphas, "alpha", "test_score")
# scatter_plot(ridge_alphas, "alpha", "test_score")


housing = pd.read_csv("Data/train_original.csv")


housing.columns
len(housing["TotalBsmtSF"])
len(housing["1stFlrSF"])
len(housing["2ndFlrSF"])


housing["TotalSF"] = (
    housing["TotalBsmtSF"] + housing["1stFlrSF"] + housing["2ndFlrSF"]
)

housing.corr()
# import matplotlib.pyplot as plt
# import numpy as np
#
#
# def scatter_plot(df, x, y):
#     plt.figure(figsize=(16, 8))
#     plt.scatter(df[x], df[y], c="black")
#     plt.xlabel(x)
#     plt.ylabel(y)
#     plt.show()
