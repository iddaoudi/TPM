import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import (
    cross_val_score,
    GridSearchCV,
    StratifiedShuffleSplit,
)
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize


def identify_best_case(df):
    acceptable_cases = df[
        df["time"] <= 1.05 * df.loc[df["case"] == 1, "time"].values[0]
    ]
    best_case = acceptable_cases.loc[acceptable_cases["total_energy"].idxmin()]
    return best_case.name


def two_targets_model(df):
    # Split data
    unique_matrix_sizes = df["matrix_size"].unique()
    train_matrix_size = unique_matrix_sizes[0]
    test_matrix_size = unique_matrix_sizes[1]

    train = df[df["matrix_size"] == train_matrix_size].copy()
    test = df[df["matrix_size"] == test_matrix_size].copy()

    # Preprocess data
    train["total_energy"] = train[["PKG1", "PKG2", "DRAM1", "DRAM2"]].sum(axis=1)
    test["total_energy"] = test[["PKG1", "PKG2", "DRAM1", "DRAM2"]].sum(axis=1)

    # Compute the targets for each combination of matrix_size and tile_size, which are the good cases
    # Initialize target column to 0 for all cases
    train["target"] = 0
    test["target"] = 0

    for matrix_size in train["matrix_size"].unique():
        for tile_size in train["tile_size"].unique():
            train_subset = train[
                (train["matrix_size"] == matrix_size)
                & (train["tile_size"] == tile_size)
            ]
            train_case1 = train_subset[train_subset["case"] == 1]
            # Set target to 1 for "good" cases
            train.loc[
                (train["matrix_size"] == matrix_size)
                & (train["tile_size"] == tile_size)
                & (train_subset["total_energy"] < train_case1["total_energy"].values[0])
                & (train_subset["time"] <= 1.05 * train_case1["time"].values[0]),
                "target",
            ] = 1
            # Set target to 2 for "best" case
            best_case_index = identify_best_case(train_subset)
            train.loc[best_case_index, "target"] = 2

    for matrix_size in test["matrix_size"].unique():
        for tile_size in test["tile_size"].unique():
            test_subset = test[
                (test["matrix_size"] == matrix_size) & (test["tile_size"] == tile_size)
            ]
            test_case1 = test_subset[test_subset["case"] == 1]
            # Set target to 1 for "good" cases
            test.loc[
                (test["matrix_size"] == matrix_size)
                & (test["tile_size"] == tile_size)
                & (test_subset["total_energy"] < test_case1["total_energy"].values[0])
                & (test_subset["time"] <= 1.05 * test_case1["time"].values[0]),
                "target",
            ] = 1
            # Set target to 2 for "best" case
            best_case_index = identify_best_case(test_subset)
            test.loc[best_case_index, "target"] = 2

    # Features
    # feature_cols = ['tile_size'] + [f'task{i}_{metric}' for i in range(1, 5) for metric in ['mem_boundness', 'arithm_intensity', 'ilp', 'l3_cache_ratio']]
    feature_cols = (
        ["tile_size"]
        + [f"task{i}" for i in range(1, 5)]
        + [
            f"task{i}_{metric}"
            for i in range(1, 5)
            for metric in ["mem_boundness", "arithm_intensity", "ilp", "l3_cache_ratio"]
        ]
    )

    # Models
    models = {
        "logistic_regression": (
            LogisticRegression(),
            {"estimator__C": [0.01, 0.1, 1, 10, 100]},
        ),
        "svm": (
            SVC(probability=True),
            {
                "estimator__C": [0.01, 0.1, 1, 10, 100],
                "estimator__gamma": ["scale", "auto"],
            },
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=1),
            {
                "estimator__n_estimators": [50, 100, 150],
                "estimator__learning_rate": [0.01, 0.1, 1],
                "estimator__max_depth": [3, 5, 7],
            },
        ),
    }

    # Scale features
    scaler = StandardScaler()
    for name, (model, params) in models.items():
        # Train
        pipeline = Pipeline([("scaler", scaler), ("estimator", model)])
        cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
        grid_search = GridSearchCV(
            pipeline, params, cv=cv, scoring="roc_auc_ovo"
        )  # one-vs-one ROC AUC for multi-class problems
        grid_search.fit(train[feature_cols], train["target"])
        print(f"Best parameters for {name}: ", grid_search.best_params_)

        # Evaluate
        test_pred_proba = grid_search.predict_proba(test[feature_cols])
        test_pred_class = grid_search.predict(test[feature_cols])

        # Compute ROC AUC for each class
        test_target_bin = label_binarize(test["target"], classes=[0, 1, 2])
        for i in range(3):  # we have 3 classes: 0, 1, and 2
            print(
                f"Test AUC-ROC for {name}, class {i}: ",
                roc_auc_score(test_target_bin[:, i], test_pred_proba[:, i]),
            )

        # Predict
        test["predicted_class"] = test_pred_class
        best_cases = test.loc[
            test.groupby(["matrix_size", "tile_size"])["predicted_class"].idxmax()
        ]
        columns_to_keep = [
            "matrix_size",
            "tile_size",
            "case",
            "target",
            "task1",
            "task2",
            "task3",
            "task4",
            "predicted_class",
        ]
        best_cases = best_cases[columns_to_keep]
        print(f"Predicted best cases for {name}:")
        print(best_cases)

    # Print feature importance if the model has 'feature_importances_' attribute
    # if hasattr(grid_search.best_estimator_.named_steps['estimator'], 'feature_importances_'):
    #     importances = grid_search.best_estimator_.named_steps['estimator'].feature_importances_
    #     features = feature_cols

    #     print(f'Feature importance for {name}:')
    #     for feature, importance in zip(features, importances):
    #         print(f'Feature   : {feature}')
    #         print(f'Importance: {importance}')

    print("********************************************************")
