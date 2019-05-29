import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split


def main():
    """ runs Ridge and Lasso r cleaned code """

    housing = pd.read_csv("Data/train_relevant", header=0)

    y = housing["SalePrice"]

    training_features, testing_features, training_target, testing_target = train_test_split(
        housing.drop("SalePrice", axis="columns"), y, random_state=42
    )

    result = []

    # 1.467 alpha previously found to give best score
    for alpha in np.append(10 ** np.linspace(-10, 2, num=200), [2.1, 1.467]):

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
                "_4_r_cleaned",
                "Ridge",
                # "alpha: {:.3f}".format(alpha),
                alpha,
                ridge_score_train,
                ridge_score_test,
            ),
            (
                "_4_r_cleaned",
                "Lasso",
                # "alpha: {:.3f}".format(alpha),
                alpha,
                lasso_score_train,
                lasso_score_test,
            ),
        ]

    return sorted(result, key=lambda x: x[4], reverse=True)


scores = main()
scores

df = pd.DataFrame(
    map(lambda r: {"model": r[1], "alpha": r[2], "test_score": r[3]}, scores)
)


def scatter_plot(df, x, y):

    plt.figure(figsize=(16, 8))
    plt.scatter(df[x], df[y], c="black")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()


scatter_plot(df[df.model == "Lasso"], "alpha", "test_score")
scatter_plot(df[df.model == "Ridge"], "alpha", "test_score")
