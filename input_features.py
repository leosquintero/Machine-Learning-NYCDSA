from tpot import TPOTClassifier
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from fancyimpute import KNN


all_features = [
    (
        "SalePrice",
        "- the property's sale price in dollars. This is the target variable that you're trying to predict.",
        True,
    ),
    ("MSSubClass", " The building class", True),
    ("MSZoning", " The general zoning classification", True),
    ("LotFrontage", " Linear feet of street connected to property", True),
    ("LotArea", " Lot size in square feet", True),
    ("Street", " Type of road access", True),
    ("Alley", " Type of alley access", True),
    ("LotShape", " General shape of property", True),
    ("LandContour", " Flatness of the property", True),
    ("Utilities", " Type of utilities available", True),
    ("LotConfig", " Lot configuration", True),
    ("LandSlope", " Slope of property", True),
    ("Neighborhood", " Physical locations within Ames city limits", True),
    ("Condition1", " Proximity to main road or railroad", True),
    (
        "Condition2",
        " Proximity to main road or railroad (if a second is present)",
        True,
    ),
    ("BldgType", " Type of dwelling", True),
    ("HouseStyle", " Style of dwelling", True),
    ("OverallQual", " Overall material and finish quality", True),
    ("OverallCond", " Overall condition rating", True),
    ("YearBuilt", " Original construction date", True),
    ("YearRemodAdd", " Remodel date", True),
    ("RoofStyle", " Type of roof", True),
    ("RoofMatl", " Roof material", True),
    ("Exterior1st", " Exterior covering on house", True),
    (
        "Exterior2nd",
        " Exterior covering on house (if more than one material)",
        True,
    ),
    ("MasVnrType", " Masonry veneer type", True),
    ("MasVnrArea", " Masonry veneer area in square feet", True),
    ("ExterQual", " Exterior material quality", True),
    ("ExterCond", " Present condition of the material on the exterior", True),
    ("Foundation", " Type of foundation", True),
    ("BsmtQual", " Height of the basement", True),
    ("BsmtCond", " General condition of the basement", True),
    ("BsmtExposure", " Walkout or garden level basement walls", True),
    ("BsmtFinType1", " Quality of basement finished area", True),
    ("BsmtFinSF1", " Type 1 finished square feet", True),
    ("BsmtFinType2", " Quality of second finished area (if present)", True),
    ("BsmtFinSF2", " Type 2 finished square feet", True),
    ("BsmtUnfSF", " Unfinished square feet of basement area", True),
    ("TotalBsmtSF", " Total square feet of basement area", True),
    ("Heating", " Type of heating", True),
    ("HeatingQC", " Heating quality and condition", True),
    ("CentralAir", " Central air conditioning", True),
    ("Electrical", " Electrical system", True),
    ("1stFlrSF", " First Floor square feet", True),
    ("2ndFlrSF", " Second floor square feet", True),
    ("LowQualFinSF", " Low quality finished square feet (all floors)", True),
    ("GrLivArea", " Above grade (ground) living area square feet", True),
    ("BsmtFullBath", " Basement full bathrooms", True),
    ("BsmtHalfBath", " Basement half bathrooms", True),
    ("FullBath", " Full bathrooms above grade", True),
    ("HalfBath", " Half baths above grade", True),
    ("KitchenQual", " Kitchen quality", True),
    (
        "TotRmsAbvGrd",
        " Total rooms above grade (does not include bathrooms)",
        True,
    ),
    ("Functional", " Home functionality rating", True),
    ("Fireplaces", " Number of fireplaces", True),
    ("FireplaceQu", " Fireplace quality", True),
    ("GarageType", " Garage location", True),
    ("GarageYrBlt", " Year garage was built", True),
    ("GarageFinish", " Interior finish of the garage", True),
    ("GarageCars", " Size of garage in car capacity", True),
    ("GarageArea", " Size of garage in square feet", True),
    ("GarageQual", " Garage quality", True),
    ("GarageCond", " Garage condition", True),
    ("PavedDrive", " Paved driveway", True),
    ("WoodDeckSF", " Wood deck area in square feet", True),
    ("OpenPorchSF", " Open porch area in square feet", True),
    ("EnclosedPorch", " Enclosed porch area in square feet", True),
    ("3SsnPorch", " Three season porch area in square feet", True),
    ("ScreenPorch", " Screen porch area in square feet", True),
    ("PoolArea", " Pool area in square feet", True),
    ("PoolQC", " Pool quality", True),
    ("Fence", " Fence quality", True),
    (
        "MiscFeature",
        " Miscellaneous feature not covered in other categories",
        True,
    ),
    ("MiscVal", " $Value of miscellaneous feature", True),
    ("MoSold", " Month Sold", True),
    ("YrSold", " Year Sold", True),
    ("SaleType", " Type of sale", True),
    ("SaleCondition", " Condition of sale", True),
]

input_features = [f[0] for f in all_features if f[2]]

housing = pd.read_csv("Data/train.csv")


def split_impute_and_dummify(
    df, target_variable="SalePrice", drop_target=True
):
    df = df[input_features]
    df = df[df[target_variable].notnull()]

    # X_all = df.drop(target_variable, axis="columns")
    if drop_target:
        X_all = df.drop(target_variable, axis="columns")
    else:
        X_all = df

    y = df[target_variable]

    non_numerical = X_all.select_dtypes(["object"])
    dummies = pd.get_dummies(non_numerical)

    numerical_X = X_all.select_dtypes(["number"])

    X_numerical_filled_knn = pd.DataFrame(
        data=KNN(k=math.floor(df.shape[0] ** 0.5)).fit_transform(numerical_X),
        columns=numerical_X.columns,
    )

    X = pd.merge(
        X_numerical_filled_knn, dummies, left_index=True, right_index=True
    )

    return train_test_split(X, y, random_state=42)


def main() -> None:

    X_train, X_test, y_train, y_test = split_impute_and_dummify(housing)

    pipeline_optimizer = TPOTClassifier(
        generations=4, population_size=16, cv=5, random_state=42, verbosity=2
    )

    pipeline_optimizer.fit(X_train, y_train)

    pipeline_optimizer.score(X_test, y_test)

    pipeline_optimizer.export("tpot_housing_pipeline_full.py")
