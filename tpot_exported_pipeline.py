import pandas as pd
from sklearn import linear_model
from input_features import split_impute_and_dummify

housing = pd.read_csv("Data/train_original.csv")

training_features, testing_features, training_target, testing_target = split_impute_and_dummify(
    housing
)

model = linear_model.Ridge()

model.fit(training_features, training_target)

model.score(testing_features, testing_target)
