from scipy.stats import pearsonr
from rory.prepare import impute_dummify_and_split
from sklearn import linear_model
import pandas as pd

housing = pd.read_csv("Data/train_original.csv")

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

# p value limit: 0.001, model: Lasso, score: 0.877
# p value limit: 0.001, model: Ridge, score: 0.877
# p value limit: 0.010, model: Lasso, score: 0.875
# p value limit: 0.010, model: Ridge, score: 0.876
# p value limit: 0.100, model: Lasso, score: 0.885
# p value limit: 0.100, model: Ridge, score: 0.883
# p value limit: 0.500, model: Lasso, score: 0.879
# p value limit: 0.500, model: Ridge, score: 0.882

# Conclusion
# not much difference from basic but restricting features would be
# significantly faster on larger data sets
