from rory import _1_basic, _2_restrict_features, _3_engineer_features
import pandas as pd
from functools import reduce

results_data = reduce(
    lambda r, m: r + m.main(),
    [_1_basic, _2_restrict_features, _3_engineer_features],
    [],
)

results_data

results = pd.DataFrame(
    data=results_data, columns=["file", "model", "notes", "score"]
)
results["ml_score"] = 1 - results.score
results.sort_values(by=["ml_score"])