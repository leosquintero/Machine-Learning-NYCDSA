import numpy as np
import pandas as pd
from sklearn import linear_model
from rory.prepare import impute_dummify_and_split
from scipy.stats import pearsonr
from scipy import stats

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

# dummify year sold
housing["YrSold"] = housing["YrSold"].astype(str)

housing = housing.drop(
    [
        "YearRemodAdd",
        "YearBuilt",
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
len(housing.columns)
# set outliers to nan

#
# def replace(group, stds):
#     group[np.abs(group - group.mean()) > stds * group.std()] = np.nan
#     return group
#
#
# for col in housing.columns:
#     if housing[col].dtype == np.float64 or housing[col].dtype == np.int64:
#         housing[col].transform(lambda g: replace(g, 3))

training_features, testing_features, training_target, testing_target = impute_dummify_and_split(
    housing, drop_target=False
)


p_values = [
    (c, pearsonr(training_features["SalePrice"], training_features[c])[1])
    for c in training_features.columns
]

p_value_limits = [0.001, 0.01, 0.1, 0.5]

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
        print(
            "p value limit: {:.3f}, model: {}, score: {:.3f}".format(
                p_value_limit, name, score
            )
        )
