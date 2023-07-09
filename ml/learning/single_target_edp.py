import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.exceptions import ConvergenceWarning
import warnings
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet
import os
import sys

import data_treatment.dictionaries as dict
import plot.plot as plot


def single_target_model_regression(
    df,
    train_algorithms,
    train_matrix_sizes,
    test_algorithms,
    test_matrix_sizes,
    architecture,
):
    print("Training on:")
    print(train_algorithms, train_matrix_sizes)
    print("Testing on:")
    print(test_algorithms, test_matrix_sizes)

    # Split data
    train = df[
        (df["matrix_size"].isin(train_matrix_sizes))
        # & (df["algorithm"].isin(train_algorithms))
    ].copy()
    test = df[
        (df["matrix_size"].isin(test_matrix_sizes))
        # & (df["algorithm"].isin(test_algorithms))
    ].copy()

    # Features
    feature_cols = (
        # App metrics
        ["number_of_tasks_normalized"]  # Represents the algorithm type
        + ["matrix_size_normalized"]
        + ["tile_size_normalized"]
        + ["case_normalized"]
        # System metrics
        + ["frequency_normalized"]
        # Hardware metrics
        + dict.metrics
    )

    # Regression Models
    models = {
        "LR": (
            LinearRegression(),
            {"estimator__fit_intercept": [True, False]},
        ),
        "Ridge": (
            Ridge(),
            {"estimator__alpha": [0.1, 1.0, 10.0]},
        ),
        "Lasso": (
            Lasso(),
            {"estimator__alpha": [0.1, 1.0, 10.0]},
        ),
        "GB": (
            GradientBoostingRegressor(random_state=1),
            {
                "estimator__n_estimators": [50, 100, 150],
                "estimator__learning_rate": [0.01, 0.1, 1],
                "estimator__max_depth": [3, 5, 7],
            },
        ),
        "XGBoost": (
            xgb.XGBRegressor(objective="reg:squarederror", tree_method="hist"),
            {
                "estimator__n_estimators": [50, 100, 150],
                "estimator__learning_rate": [0.01, 0.1, 1],
                "estimator__max_depth": [3, 5, 7],
            },
        ),
        "CatBoost": (
            CatBoostRegressor(verbose=0),  # verbose=0 to disable training output
            {
                "estimator__n_estimators": [50, 100, 150],
                "estimator__learning_rate": [0.01, 0.1, 1],
                "estimator__depth": [3, 5, 7],
            },
        ),
        "RF": (
            RandomForestRegressor(),
            {
                "estimator__n_estimators": [50, 100, 150],
                "estimator__max_depth": [3, 5, 7],
            },
        ),
        "ExtraTrees": (
            ExtraTreesRegressor(),
            {
                "estimator__n_estimators": [50, 100, 150],
                "estimator__max_depth": [3, 5, 7],
            },
        ),
        "SVM": (
            SVR(),
            {
                "estimator__C": [0.1, 1, 10],
                "estimator__gamma": ["scale", "auto"],
            },
        ),
        "KNN": (
            KNeighborsRegressor(),
            {"estimator__n_neighbors": [3, 5, 7]},
        ),
        "ElasticNet": (
            ElasticNet(),
            {"estimator__alpha": [0.1, 1, 10]},
        ),
    }

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    all_predictions = pd.DataFrame()

    # Scale features
    # scaler = MinMaxScaler()
    scaler = RobustScaler()
    # scaler = StandardScaler()

    TOLERANCE = float(os.getenv('TOLERANCE'))
    if TOLERANCE == None:
        sys.exit("TOLERANCE not defined")

    TARGET = os.getenv('TARGET')
    if TARGET == None:
        sys.exit("TARGET not defined")
    
    for name, (model, params) in models.items():
        best_cases = pd.DataFrame()
        
        # Train
        pipeline = Pipeline([("scaler", scaler), ("estimator", model)])
        cv = KFold(n_splits=16)
        grid_search = GridSearchCV(
            pipeline, params, cv=cv, scoring="neg_mean_squared_error"
        )

        if TARGET == "edp":
            grid_search.fit(train[feature_cols], train["edp"])
        elif TARGET == "energy":
            grid_search.fit(train[feature_cols], train["energy"])
        else:
            sys.exit("TARGET option unknown")

        print(f"Best parameters for {name}: ", grid_search.best_params_)

        test_pred = grid_search.predict(test[feature_cols])

        test["predicted_value"] = test_pred

        # Get unique combinations of algorithm, matrix_size, and tile_size
        unique_combinations = test[["algorithm", "matrix_size", "tile_size"]].drop_duplicates()

        for _, row in unique_combinations.iterrows():
            # Filter rows based on the current combination and case 1
            case1_row = test[
                (test["algorithm"] == row["algorithm"])
                & (test["matrix_size"] == row["matrix_size"])
                & (test["tile_size"] == row["tile_size"])
                & (test["case"] == 1)
            ]

            if case1_row.empty:
                continue

            energy_case1 = case1_row["energy"].values[0]
            time_case1 = case1_row["time"].values[0] * TOLERANCE

            # Filter rows based on energy and time conditions
            filtered_test = test[
                (test["algorithm"] == row["algorithm"])
                & (test["matrix_size"] == row["matrix_size"])
                & (test["tile_size"] == row["tile_size"])
                & (test["energy"] < energy_case1)
                & (test["time"] < time_case1)
                & (test["case"] != 1)
            ]

        #     print(case1_row[["matrix_size", "tile_size", "case", "edp", "predicted_value", "time", "energy"]])
        #     print("Tolerance: time :", time_case1, "energy: ", energy_case1)
        #     print(filtered_test[["matrix_size", "tile_size", "case", "edp", "predicted_value", "time", "energy"]])

        # exit(0)

            # Find case with the minimum predicted value
            min_target = filtered_test["predicted_value"].min()

            best_case = filtered_test[filtered_test["predicted_value"] == min_target]
            best_cases = pd.concat([best_cases, best_case])

        # Keep certain columns
        columns_to_keep = [
            "algorithm",
            "matrix_size",
            "tile_size",
            "case",
            "predicted_value",
            "edp",
            "time",
            "energy",
            "normalized_time",
            "normalized_energy",
        ]
        best_cases = best_cases[columns_to_keep]
        best_cases = best_cases.reset_index(drop=True)

        best_cases["model"] = name
        all_predictions = pd.concat([all_predictions, best_cases])

        print(f"Predicted best cases for {name}:")
        print(best_cases)

    # plot.plot_predictions(all_predictions, df, architecture)
    plot.plot_best_predictions(all_predictions, df, architecture)
