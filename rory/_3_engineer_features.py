import numpy as np
import pandas as pd
from sklearn import linear_model
from rory.prepare import impute_dummify_and_split
from scipy.stats import pearsonr
import math


def main():

    housing = pd.read_csv("Data/train_original.csv")
    # Engineer features

    housing["MonthsSinceBuild"] = (
        housing["YrSold"] * 12 + housing["MoSold"]
    ) - housing["YearBuilt"] * 12

    housing["YearsSinceRemodel"] = housing["YrSold"] - housing["YearRemodAdd"]

    housing["totalInnerSqFt"] = housing["GrLivArea"] + housing["TotalBsmtSF"]

    housing["bathrooms"] = (
        housing["BsmtFullBath"]
        + housing["BsmtHalfBath"]
        + housing["FullBath"]
        + housing["HalfBath"]
    )

    # dummify year cols
    housing["YrSold"] = housing["YrSold"].astype(str)
    housing["YearBuilt"] = housing["YearBuilt"].astype(str)

    housing = housing.drop(
        [
            "YearRemodAdd",
            "MoSold",
            "YearBuilt",
            "GrLivArea",
            "GrLivArea",
            "TotalBsmtSF",
            "BsmtQual",
            "BsmtCond",
            "BsmtExposure",
            "BsmtFinType1",
            "BsmtFinSF1",
            "BsmtFinType2",
            "BsmtFinSF2",
            "BsmtUnfSF",
            "BsmtFullBath",
            "BsmtHalfBath",
            "FullBath",
            "HalfBath",
            "MiscFeature",
        ],
        axis="columns",
    )

    # set outliers to nan (appear to slightly lower score)

    def replace(val, mean, lower_bound, upper_bound):
        if val < lower_bound or val > upper_bound:
            return math.nan  # will be dropped or imputed later
        return val

    for col in housing.columns:
        if col != "SalePrice" and (
            housing[col].dtype == np.float64 or housing[col].dtype == np.int64
        ):
            std = housing[col].std()
            mean = housing[col].mean()
            lower_bound = mean - (std * 3)
            upper_bound = mean + (std * 3)

            housing[col] = housing[col].apply(
                lambda v: replace(v, mean, lower_bound, upper_bound)
            )

    # drop columns with high nan count (appears to slightly lower score)
    nan_counts = housing.isna().sum()

    ratio_of_allowed_nans = 0.3

    nan_counts < (housing.shape[0] * ratio_of_allowed_nans)

    cols_to_drop = nan_counts[
        nan_counts > (housing.shape[0] * ratio_of_allowed_nans)
    ].keys()

    housing = housing.drop(cols_to_drop, axis="columns")

    # prepare data
    training_features, testing_features, training_target, testing_target = impute_dummify_and_split(
        housing, drop_target=False
    )

    p_values = [
        (c, pearsonr(training_features["SalePrice"], training_features[c])[1])
        for c in training_features.columns
    ]

    p_value_limits = [0.001, 0.01, 0.1, 0.5]

    result = []

    for p_value_limit in p_value_limits:

        columns = [p[0] for p in p_values if p[1] < p_value_limit]

        training_features_restricted = training_features[columns].drop(
            "SalePrice", axis="columns"
        )

        testing_featuress_restricted = testing_features[columns].drop(
            "SalePrice", axis="columns"
        )

        for model in linear_model.Lasso(), linear_model.Ridge():

            model.fit(training_features_restricted, training_target)

            score = model.score(testing_featuress_restricted, testing_target)
            name = str(model).split("(")[0]
            result = result + [
                (
                    "_3_engineer_features",
                    name,
                    "p value limit: {:.3f}".format(p_value_limit),
                    score,
                )
            ]

    return result
