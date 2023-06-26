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
from lightgbm import LGBMRegressor

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

    # Preprocess data
    # train["total_energy"] = train[["PKG1", "PKG2", "DRAM1", "DRAM2"]].sum(axis=1)
    # test["total_energy"] = test[["PKG1", "PKG2", "DRAM1", "DRAM2"]].sum(axis=1)

    # # Compute the targets as energy * time for both train and test data
    # train["target"] = train["total_energy"] * train["time"]
    # test["target"] = test["total_energy"] * test["time"]
    # train["target"] = train["edp"]
    # test["target"] = test["edp"]

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
        # "XGBoost": (
        #     xgb.XGBRegressor(objective="reg:squarederror", tree_method="hist"),
        #     {
        #         "estimator__n_estimators": [50, 100, 150],
        #         "estimator__learning_rate": [0.01, 0.1, 1],
        #         "estimator__max_depth": [3, 5, 7],
        #     },
        # ),
        "CatBoost": (
            CatBoostRegressor(verbose=0),  # verbose=0 to disable training output
            {
                "estimator__n_estimators": [50, 100, 150],
                "estimator__learning_rate": [0.01, 0.1, 1],
                "estimator__depth": [3, 5, 7],
            },
        ),
    }

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    all_predictions = pd.DataFrame()

    # Scale features
    # scaler = MinMaxScaler()
    scaler = RobustScaler()
    # scaler = StandardScaler()
    for name, (model, params) in models.items():
        best_cases = pd.DataFrame()
        # Train
        pipeline = Pipeline([("scaler", scaler), ("estimator", model)])
        cv = KFold(n_splits=16)
        grid_search = GridSearchCV(
            pipeline, params, cv=cv, scoring="neg_mean_squared_error"
        )
        grid_search.fit(train[feature_cols], train["edp"])

        print(f"Best parameters for {name}: ", grid_search.best_params_)
        # print(f"Best score for {name}     : ", -grid_search.best_score_)

        best_model = grid_search.best_estimator_
        if hasattr(best_model[-1], "feature_importances_"):
            importance = best_model[-1].feature_importances_
            print(f"Feature importance for {name}:")
            for i, j in enumerate(importance):
                print(f"{feature_cols[i]}: {j}")

        # Evaluate
        test_pred = grid_search.predict(test[feature_cols])

        # Predict
        test["predicted_value"] = test_pred

        min_targets = (
            test.groupby(["algorithm", "matrix_size", "tile_size"])["predicted_value"]
            .min()
            .reset_index()
        )

        best_cases = pd.merge(
            test,
            min_targets,
            on=[
                "algorithm",
                "matrix_size",
                "tile_size",
                "predicted_value",
            ],
        )

        columns_to_keep = [
            "algorithm",
            "matrix_size",
            "tile_size",
            "case",
            "predicted_value",
            "edp",
            "time",
            "PKG1",
            "PKG2",
            "DRAM1",
            "DRAM2",
        ]
        best_cases = best_cases[columns_to_keep]

        best_cases["model"] = name
        all_predictions = pd.concat([all_predictions, best_cases])

        print(f"Predicted best cases for {name}:")
        print(best_cases)

    # plot.plot_predictions(all_predictions, df, architecture)
    plot.plot_best_predictions(all_predictions, df, architecture)
