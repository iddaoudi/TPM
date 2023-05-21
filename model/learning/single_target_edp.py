from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import warnings

import data_treatment.dictionaries as dict


def single_target_model_regression(
    df, train_algorithms, train_matrix_sizes, test_algorithms, test_matrix_sizes
):
    print("Training on:")
    print(train_algorithms, train_matrix_sizes)
    print("Testing on:")
    print(test_algorithms, test_matrix_sizes)

    # Split data
    train = df[
        (df["matrix_size"].isin(train_matrix_sizes))
        & (df["algorithm"].isin(train_algorithms))
    ].copy()
    test = df[
        (df["matrix_size"].isin(test_matrix_sizes))
        & (df["algorithm"].isin(test_algorithms))
    ].copy()

    # Preprocess data
    train["total_energy"] = train[["PKG1", "PKG2", "DRAM1", "DRAM2"]].sum(axis=1)
    test["total_energy"] = test[["PKG1", "PKG2", "DRAM1", "DRAM2"]].sum(axis=1)

    # Compute the targets as energy * time for both train and test data
    train["target"] = train["total_energy"] * train["time"]
    test["target"] = test["total_energy"] * test["time"]

    # Features
    feature_cols = (
        # App metrics
        ["matrix_size"]
        + ["tile_size"]
        + [f"task{i}_weight" for i in range(1, 5)]
        + ["case"]
        # System metrics
        + [f"task{i}" for i in range(1, 5)]
        # Hardware metrics
        + [f"task{i}_{metric}" for i in range(1, 5) for metric in dict.metrics]
    )

    # Regression Models
    models = {
        "linear_regression": (
            LinearRegression(),
            {"estimator__fit_intercept": [True, False]},
        ),
        "ridge": (
            Ridge(),
            {"estimator__alpha": [0.1, 1.0, 10.0]},
        ),
        "lasso": (
            Lasso(),
            {"estimator__alpha": [0.1, 1.0, 10.0]},
        ),
        "svr": (
            SVR(),
            {
                "estimator__C": [0.01, 0.1, 1, 10, 100],
                "estimator__gamma": ["scale", "auto"],
            },
        ),
        "random_forest_regressor": (
            RandomForestRegressor(),
            {
                "estimator__n_estimators": [50, 100, 150],
                "estimator__max_depth": [3, 5, 7],
                "estimator__min_samples_split": [2, 5, 10],
            },
        ),
        "gradient_boosting_regressor": (
            GradientBoostingRegressor(random_state=1),
            {
                "estimator__n_estimators": [50, 100, 150],
                "estimator__learning_rate": [0.01, 0.1, 1],
                "estimator__max_depth": [3, 5, 7],
            },
        ),
    }

    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    # Scale features
    scaler = StandardScaler()
    for name, (model, params) in models.items():
        # Train
        pipeline = Pipeline([("scaler", scaler), ("estimator", model)])
        cv = KFold(n_splits=5)
        grid_search = GridSearchCV(
            pipeline, params, cv=cv, scoring="neg_mean_squared_error"
        )
        grid_search.fit(train[feature_cols], train["target"])

        print(f"Best parameters for {name}: ", grid_search.best_params_)
        print(
            f"Best score for {name}     : ", -grid_search.best_score_
        )  # The score is negated as GridSearchCV uses negative MSE

        # Evaluate
        test_pred = grid_search.predict(test[feature_cols])

        # Predict
        test["predicted_value"] = test_pred
        best_cases = test.loc[
            test.groupby(["algorithm", "matrix_size", "tile_size"])[
                "predicted_value"
            ].idxmin()
        ]
        columns_to_keep = [
            "algorithm",
            "matrix_size",
            "tile_size",
            "case",
            "target",
            "task1",
            "task2",
            "task3",
            "task4",
            "predicted_value",
        ]
        best_cases = best_cases[columns_to_keep]
        print(f"Predicted best cases for {name}:")
        print(best_cases)
