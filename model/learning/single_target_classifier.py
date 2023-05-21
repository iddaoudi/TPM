from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import warnings
from sklearn.exceptions import ConvergenceWarning

import data_treatment.dictionaries as dict


def single_target_model(
    df, train_algorithms, train_matrix_sizes, test_algorithms, test_matrix_sizes
):
    print("Training on:")
    print(train_algorithms, train_matrix_sizes)
    print("Testing on:")
    print(test_algorithms, test_matrix_sizes)

    # Split data FIXME according to matrix_size
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

    # Compute the targets for each combination of matrix_size and tile_size
    for matrix_size in train["matrix_size"].unique():
        for tile_size in train["tile_size"].unique():
            train_subset = train[
                (train["matrix_size"] == matrix_size)
                & (train["tile_size"] == tile_size)
            ]
            train_case1 = train_subset[train_subset["case"] == 1]
            train.loc[
                (train["matrix_size"] == matrix_size)
                & (train["tile_size"] == tile_size),
                "target",
            ] = (
                train_subset["total_energy"] < train_case1["total_energy"].values[0]
            ) & (
                train_subset["time"] <= 1.05 * train_case1["time"].values[0]
            )

    for matrix_size in test["matrix_size"].unique():
        for tile_size in test["tile_size"].unique():
            test_subset = test[
                (test["matrix_size"] == matrix_size) & (test["tile_size"] == tile_size)
            ]
            test_case1 = test_subset[test_subset["case"] == 1]
            test.loc[
                (test["matrix_size"] == matrix_size) & (test["tile_size"] == tile_size),
                "target",
            ] = (test_subset["total_energy"] < test_case1["total_energy"].values[0]) & (
                test_subset["time"] <= 1.05 * test_case1["time"].values[0]
            )

    train["target"] = train["target"].astype(bool)
    test["target"] = test["target"].astype(bool)

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
        "random_forest": (
            RandomForestClassifier(),
            {
                "estimator__n_estimators": [50, 100, 150],
                "estimator__max_depth": [3, 5, 7],
                "estimator__min_samples_split": [2, 5, 10],
            },
        ),
        "knn": (
            KNeighborsClassifier(),
            {
                "estimator__n_neighbors": [3, 5, 7, 9],
                "estimator__weights": ["uniform", "distance"],
                "estimator__algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            },
        ),
        "mlp": (
            MLPClassifier(),
            {
                "estimator__hidden_layer_sizes": [(50,), (100,)],
                "estimator__activation": ["tanh", "relu"],
                "estimator__solver": ["sgd", "adam"],
                "estimator__alpha": [0.0001, 0.05],
                "estimator__learning_rate": ["constant", "adaptive"],
            },
        ),
    }

    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    # Scale features
    scaler = StandardScaler()
    for name, (model, params) in models.items():
        # Train
        pipeline = Pipeline([("scaler", scaler), ("estimator", model)])
        cv = StratifiedKFold(n_splits=5)
        grid_search = GridSearchCV(pipeline, params, cv=cv, scoring="roc_auc")
        grid_search.fit(train[feature_cols], train["target"])
        # print(f"Best parameters for {name}: ", grid_search.best_params_)
        print(f"Best score for {name}     : ", grid_search.best_score_)

        # Evaluate
        test_pred = grid_search.predict_proba(test[feature_cols])[:, 1]

        # Predict
        test["predicted_probability"] = test_pred
        best_cases = test.loc[
            test.groupby(["algorithm", "matrix_size", "tile_size"])[
                "predicted_probability"
            ].idxmax()
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
            "predicted_probability",
        ]
        best_cases = best_cases[columns_to_keep]
        print(f"Predicted best cases for {name}:")
        print(best_cases)

        # Print feature importance if the model has 'feature_importances_' attribute
        feature_importance = 0
        if feature_importance:
            if hasattr(
                grid_search.best_estimator_.named_steps["estimator"],
                "feature_importances_",
            ):
                importances = grid_search.best_estimator_.named_steps[
                    "estimator"
                ].feature_importances_
                features = feature_cols

                print(f"Feature importance for {name}:")
                for feature, importance in zip(features, importances):
                    print(f"Feature   : {feature}")
                    print(f"Importance: {importance}")

        print("********************************************************")
