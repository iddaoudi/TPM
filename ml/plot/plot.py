import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.legend import Legend
import os
import sys

marker_symbols = {
    1: "o",
    2: "s",
    3: "^",
    4: "d",
    5: "*",
    6: "p",
    7: "x",
    8: "h",
    9: "+",
    10: "v",
    11: ">",
    12: "<",
    13: "H",
    14: "D",
    15: "P",
    16: "X",
}

color_dict = {
    "LR": "blue",
    "Ridge": "green",
    "Lasso": "red",
    "GB": "purple",
    "XGBoost": "orange",
    "CatBoost": "magenta",
}


def plot_predictions(predictions, df, architecture):
    models = predictions["model"].unique()
    algorithms = predictions["algorithm"].unique()
    matrix_sizes = predictions["matrix_size"].unique()
    tile_sizes = predictions["tile_size"].unique()
    width = 0.1  # the width of the bars

    df["time_energy_product"] = df["time"] * df[["PKG1", "PKG2", "DRAM1", "DRAM2"]].sum(
        axis=1
    )

    for algorithm in algorithms:
        for matrix_size in matrix_sizes:
            fig, ax = plt.subplots(figsize=(10, 7))  # single ax
            fig.suptitle(f"Algorithm: {algorithm} - Matrix Size: {matrix_size}")

            for idx, model in enumerate(models):  # iterate over models instead of tiles
                model_predictions = predictions[
                    (predictions["model"] == model)
                    & (predictions["algorithm"] == algorithm)
                    & (predictions["matrix_size"] == matrix_size)
                ]
                bar_positions = (
                    np.arange(len(tile_sizes)) + idx * width
                )  # new bar position for each model

                for tile_size_pos, tile_size in zip(bar_positions, tile_sizes):
                    tile_predictions = model_predictions[
                        model_predictions["tile_size"] == tile_size
                    ]

                    # Extract the default case
                    default_case = df[
                        (df["case"] == 1)
                        & (df["algorithm"] == algorithm)
                        & (df["matrix_size"] == matrix_size)
                        & (df["tile_size"] == tile_size)
                    ]
                    default_case_value = default_case["time_energy_product"].values[0]

                    # Calculate the improvement percentage over the default case
                    best_case_value = tile_predictions["time"] * tile_predictions[
                        ["PKG1", "PKG2", "DRAM1", "DRAM2"]
                    ].sum(axis=1)
                    if not best_case_value.empty:
                        improvement_percentage = (
                            (default_case_value - best_case_value) / default_case_value
                        ) * 100
                        bar = ax.bar(
                            tile_size_pos,  # plot position
                            improvement_percentage,
                            width=width,  # set bar width
                            color=color_dict[model],  # use color from color_dict
                        )

                # Add the legend for the model just once
                bar.set_label(model)

            ax.set_title(f"Tile Size Improvement")
            ax.set_ylabel("Improvement % over Default Case")
            ax.set_xticks(np.arange(len(tile_sizes)) + width * len(models) / 2)
            ax.set_xticklabels(tile_sizes)
            ax.legend()  # add this line
            fig.tight_layout()
            plt.savefig(
                f"predictions_{architecture}_{algorithm}_{matrix_size}_robust.png"
            )
            plt.show()


def plot_best_predictions(predictions, df, architecture):
    models = predictions["model"].unique()
    algorithms = predictions["algorithm"].unique()
    matrix_sizes = predictions["matrix_size"].unique()
    tile_sizes = predictions["tile_size"].unique()
    width = 0.35 / len(tile_sizes)  # Adjust width of the bars

    # Use seaborn styles
    sns.set_theme()
    plt.figure(figsize=(10, 7))  # Adjust as needed
    fig, ax = plt.subplots()

    # Increase font size
    plt.rcParams.update({"font.size": 14})

    # Generate color palette
    palette = sns.color_palette("hls", len(models))

    color_dict = dict(zip(models, palette))

    # Predefined hatch patterns
    hatch_patterns = ["///", "\\\\\\", "|||", "-", "+", "x", "o", "O", ".", "*"]
    tile_hatch_dict = dict(zip(tile_sizes, hatch_patterns))

    legend_patches = {}

    TOLERANCE = float(os.getenv('TOLERANCE'))
    if TOLERANCE == None:
        print("TOLERANCE not defined")
        exit(0)
    
    COMPARE = os.getenv('COMPARE')
    if COMPARE == None:
        sys.exit("COMPARE not defined")

    for algorithm in algorithms:
        for idx, matrix_size in enumerate(matrix_sizes):
            offset = (len(tile_sizes) - 1) / 2 * width

            for tile_pos, tile_size in enumerate(tile_sizes):
                best_improvement = None
                best_model = None

                for model in models:
                    model_predictions = predictions[
                        (predictions["model"] == model)
                        & (predictions["algorithm"] == algorithm)
                        & (predictions["matrix_size"] == matrix_size)
                        & (predictions["tile_size"] == tile_size)
                    ]

                    default_case = df[
                        (df["case"] == 1)
                        & (df["algorithm"] == algorithm)
                        & (df["matrix_size"] == matrix_size)
                        & (df["tile_size"] == tile_size)
                    ]
                    default_case_edp = default_case["edp"].values[0]
                    default_case_energy = default_case["energy"].values[0]

                    best_case_edp = model_predictions["edp"]
                    best_case_energy = model_predictions["energy"]

                    if COMPARE == "edp":
                        if not best_case_edp.empty:
                            current_improvement = (
                                (default_case_edp - best_case_edp.iloc[0])
                                / default_case_edp
                            ) * 100

                            if (
                                best_improvement is None
                                or current_improvement > best_improvement
                            ):
                                best_improvement = current_improvement
                                best_model = model
                    elif COMPARE == "energy":
                        if not best_case_energy.empty:
                            current_improvement = (
                                (default_case_energy - best_case_energy.iloc[0])
                                / default_case_energy
                            ) * 100

                            if (
                                best_improvement is None
                                or current_improvement > best_improvement
                            ):
                                best_improvement = current_improvement
                                best_model = model
                    else:
                        sys.exit("COMPARE option unknown")

                if best_improvement is not None:
                    bar_pos = idx - offset + tile_pos * width  # calculate bar position
                    bar = ax.bar(
                        bar_pos,
                        best_improvement,
                        width=width,
                        color=color_dict[best_model],
                        hatch=tile_hatch_dict[tile_size],
                    )  # Add hatch pattern based on tile size
                    legend_patches[best_model] = Patch(
                        color=color_dict[best_model]
                    )  # Create patch for legend

    ax.set_title("Best improvement for each matrix and tile sizes", pad=20)
    ax.set_ylabel("Improvement % over Default Case", labelpad=15)
    ax.set_xlabel("Matrix Size", labelpad=15)
    ax.set_xticks(np.arange(len(matrix_sizes)))
    ax.set_xticklabels(matrix_sizes)

    legend1 = ax.legend(
        legend_patches.values(),
        legend_patches.keys(),
        title="Models",
        title_fontsize="13",
        loc="upper left",
    )
    ax.add_artist(legend1)

    legend_patches_tile = [
        Patch(facecolor="gray", hatch=tile_hatch_dict[size], edgecolor="black")
        for size in tile_sizes
    ]
    legend2 = Legend(
        ax,
        legend_patches_tile,
        labels=tile_sizes,
        title="Tile Sizes",
        loc="upper right",
        frameon=True,
    )
    ax.add_artist(legend2)

    ax.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()

    TARGET = os.getenv('TARGET')
    if TARGET == None:
        sys.exit("TARGET not defined")

    plt.savefig(f"misc/results/graphs/predictions/predictions_{COMPARE}_{architecture}_{algorithm}_{TOLERANCE}_{TARGET}.png")
    # plt.show()
