from rory import (
    _1_basic,
    _2_restrict_features,
    _3_engineer_features,
    _4_r_cleaned,
)

import pandas as pd
from functools import reduce

results_data = reduce(
    lambda r, m: r + m.main(),
    [_1_basic, _2_restrict_features, _3_engineer_features, _4_r_cleaned],
    [],
)

results_data

results = pd.DataFrame(
    data=results_data,
    columns=["file", "model", "notes", "train_score", "test_score"],
)
results["ml_train_score"] = 1 - results.train_score
results["ml_test_score"] = 1 - results.test_score

results.sort_values(by=["ml_test_score"])[:10].drop(
    ["ml_train_score", "ml_test_score"], axis="columns"
)
