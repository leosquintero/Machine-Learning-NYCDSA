from scipy.stats import pearsonr
from rory.prepare import impute_dummify_and_split
from sklearn import linear_model
import pandas as pd


def main():
    """ only uses features with a correlation p value above a certain limit """
    housing = pd.read_csv("Data/train_original.csv")
    housing["TotalSF"] = (
        housing["TotalBsmtSF"] + housing["1stFlrSF"] + housing["2ndFlrSF"]
    )
    training_features, testing_features, training_target, testing_target = impute_dummify_and_split(
        housing, drop_target=False
    )

    p_values = [
        (c, pearsonr(training_features["SalePrice"], training_features[c])[1])
        for c in training_features.columns
    ]

    p_value_limits = [0.05]

    result = []
    ps_and_cols = {}

    for p_value_limit in p_value_limits:

        high_ps = list(
            map(lambda t: t[0], sorted(p_values, key=lambda t1: t1[1])[:15])
        )

        print(training_features[high_ps].corr())

        columns = [p[0] for p in p_values if p[1] < p_value_limit]

        training_features_restricted = training_features[columns].drop(
            "SalePrice", axis="columns"
        )

        testing_features_restricted = testing_features[columns].drop(
            "SalePrice", axis="columns"
        )

        for model in (
            linear_model.Lasso(alpha=2.1),
            linear_model.Ridge(alpha=2.1),
        ):

            model.fit(training_features_restricted, training_target)

            train_score = model.score(
                training_features_restricted, training_target
            )

            test_score = model.score(
                testing_features_restricted, testing_target
            )

            name = str(model).split("(")[0]

            result = result + [
                (
                    "_2_restrict_features",
                    name,
                    "p value limit: {:.3f}, alpha: 2.1".format(p_value_limit),
                    train_score,
                    test_score,
                )
            ]

    print(ps_and_cols)
    return training_features[high_ps].corr()


main()
